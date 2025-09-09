import torch
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
from model.seec import SeecNet, SeecLoss
from datasets.dataset import ImgMaskDataset
from datasets.transform import TrainTransform, EvalTransform
import os
from pathlib import Path
from utils.func import set_seed, configure_optimizers
from utils.engine import train_epoch, eval_epoch


def get_args_parser():
    parser = argparse.ArgumentParser(description="Train a model")

    #  Training parameters
    parser.add_argument("--resume", default="", help="Folder that contains checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--aux_lr", type=float, default=0.001, help="Auxiliary learning rate")
    parser.add_argument("--num_epochs", type=int, default=1500, help="Number of epochs")
    parser.add_argument(
        "--lr_reduce_patience",
        type=int,
        default=30,
        help="Patience for learning rate reduction",
    )
    parser.add_argument(
        "--lr_reduce_factor",
        type=float,
        default=0.9,
        help="Learning rate reduction factor",
    )
    parser.add_argument("--multistep", action="store_true", help="Use MultiStepLR scheduler")
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=[350, 390, 430, 470, 510, 550, 590],
        help="Milestones for MultiStepLR",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="Gamma for MultiStepLR")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--p_hflip", type=float, default=0.5, help="Probability of horizontal flip")
    parser.add_argument("--p_vflip", type=float, default=0.5, help="Probability of vertical flip")
    parser.add_argument(
        "--train_path",
        type=str,
        default="./data/DIV2K_train_p128",
        help="Path to the training data",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="./data/DIV2K_valid_p128",
        help="Path to the validation data",
    )
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Number of batches to preload")

    # Model parameters
    parser.add_argument("--no_seg", action="store_true", help="Disable segmentation")
    parser.add_argument("--no_multichannel_lmm", action="store_true", help="Disable multi-channel LMM")
    parser.add_argument("--num_ch", type=int, default=192, help="Number of channels for the network")
    parser.add_argument("--prior_ch", type=int, default=256, help="Number of prior channels")
    parser.add_argument("--context_ch", type=int, default=256, help="Number of context channels")
    parser.add_argument("--ep_ch", type=int, default=256, help="Number of ep channels")
    parser.add_argument(
        "--num_mix",
        type=int,
        default=5,
        help="Number of mixtures for the entropy model",
    )
    return parser.parse_args()


def train(args):

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    PATCH_SZ = 64

    set_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    transform_train = TrainTransform(PATCH_SZ, args.p_hflip, args.p_vflip)
    transform_val = EvalTransform(PATCH_SZ)

    train_dataset = ImgMaskDataset(args.train_path, transform=transform_train)
    val_dataset = ImgMaskDataset(args.val_path, transform=transform_val)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )
    model = SeecNet(
        num_ch=args.num_ch,
        prior_ch=args.prior_ch,
        context_ch=args.context_ch,
        ep_ch=args.ep_ch,
        num_mix=args.num_mix,
        noseg=args.no_seg,
        no_multichannel_lmm=args.no_multichannel_lmm,
    ).to(device)

    criterion = SeecLoss()
    optimizer, aux_optimizer = configure_optimizers(model, args.lr, args.aux_lr)
    if args.multistep:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        args.num_epochs = 600
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_reduce_factor, patience=args.lr_reduce_patience
        )
    out_dir = os.path.join(
        args.output_dir,
        "logs",
        f"seec_seg_{not args.no_seg}_multichannel_{not args.no_multichannel_lmm}",
        time.strftime("%Hh%Mm%Ss_on_%b_%d"),
    )

    if args.resume:
        out_dir = os.path.dirname(args.resume).replace("ckp", "logs")
        ckp = torch.load(args.resume)
        model.load_state_dict(ckp["model"])
        start_epoch = ckp["epoch"] + 1
        optimizer.load_state_dict(ckp["optimizer_state_dict"])
        aux_optimizer.load_state_dict(ckp["aux_optimizer_state_dict"])
        scheduler.load_state_dict(ckp["scheduler"])
        train_step = ckp["step"]
        best_bpp = ckp["best_bpp"]
    else:
        start_epoch = 0
        train_step = 0
        best_bpp = 1e10
    ckp_dir = out_dir.replace("logs", "ckp")
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    writer = SummaryWriter(log_dir=out_dir)
    writer.add_text("args", str(args))
    try:

        for epoch in range(start_epoch, args.num_epochs):
            train_step = train_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                aux_optimizer,
                writer,
                train_step,
                clip_grad=args.clip_grad,
            )

            val_loss, val_x_bpp, val_latent_bpp, val_z_bpp, val_y_bpp, val_aux_loss = eval_epoch(
                model, criterion, val_loader, epoch, writer
            )
            if args.multistep:
                scheduler.step()
            else:
                scheduler.step(val_loss)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "aux_optimizer_state_dict": aux_optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": train_step,
                    "best_bpp": best_bpp,
                },
                os.path.join(ckp_dir, "model.pt"),
            )
            if val_loss < best_bpp:
                best_bpp = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                    },
                    os.path.join(ckp_dir, "best_model.pt"),
                )

            print("Epoch: {}, bpp:{}".format(epoch + 1, val_loss))

            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch == args.num_epochs - 1:
            print("Training stopped because reached maximum")

        writer.close()
    except KeyboardInterrupt:
        print("Exiting from training early because of KeyboardInterrupt")


if __name__ == "__main__":

    args = get_args_parser()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
