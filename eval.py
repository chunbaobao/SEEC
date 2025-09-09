import torch
from PIL import Image
from torchvision import transforms
from model.seec import SeecNet, SeecLoss
import os
from model_hub.models.birefnet import BiRefNet
from encode import compress
from decode import decompress
from utils.func import get_md5, AverageMeter, check_state_dict
from utils.func import img2patch, extract_mask, coding_table_3p


def config_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        help="Path to the model checkpoint",
    )

    parser.add_argument(
        "--birefnet_ckpt",
        type=str,
        default="model_hub/BiRefNet-general-epoch_244.pth",
        help="Path to the BiRefNet checkpoint",
    )
    parser.add_argument("--imgdir", type=str, nargs="+", help="Directory containing images to encode")
    # parser.add_argument("--dec", action="store_true", help="If set, decode the images")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory to cache results")
    parser.add_argument("--dryrun", action="store_true", help="Run dry run for testing purposes")
    parser.add_argument(
        "--segtype",
        type=str,
        choices=["norm", "random", "wrong"],
        default="norm",
        help="Mask type for segmentation",
    )
    return parser.parse_args()


def main():
    args = config_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    birefnet = BiRefNet(bb_pretrained=False)
    state_dict = torch.load(args.birefnet_ckpt, map_location="cpu")
    state_dict = check_state_dict(state_dict)
    birefnet.load_state_dict(state_dict)
    birefnet.to(device)
    birefnet.eval()
    birefnet.half()

    model = SeecNet.from_state_dict(torch.load(args.ckpt)["model"]).to(device)

    model.seg_img_compressor.update(force=True)
    model.eval()
    COT = coding_table_3p(patch_sz=64).to(device)

    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    if args.imgdir is None:
        return

    if isinstance(args.imgdir, str):
        args.imgdir = [args.imgdir]

    if not args.dryrun:

        for imgdir in args.imgdir:
            bpp = AverageMeter()
            enc_time = AverageMeter()
            dec_time = AverageMeter()

            for path in os.listdir(imgdir):
                img_path = os.path.join(imgdir, path)

                cache_key = get_md5(args.ckpt, img_path, args.segtype)
                if cache_key in os.listdir(args.cache_dir):
                    enc_results, dec_results = torch.load(os.path.join(args.cache_dir, cache_key))
                else:
                    latent_code, x_stream, seg_bin, img_shape, enc_results = compress(
                        model, birefnet, img_path, COT=COT, segtype=args.segtype
                    )
                    img, dec_results = decompress(model, latent_code, x_stream, img_shape, seg_bin, COT)
                    original_img = Image.open(img_path).convert("RGB")
                    original_img = transforms.PILToTensor()(original_img)
                    # assert torch.all(img == original_img), "Decoded image does not match the original image"
                    torch.save((enc_results, dec_results), os.path.join(args.cache_dir, cache_key))
                bpp.update(enc_results["bpp"])
                enc_time.update(enc_results["enc_time"])
                dec_time.update(dec_results["dec_time"])

            print(f"Results for {imgdir}:")
            print(f"Average BPP: {bpp.avg:.2f}")
            print(f"Average Encoding Time: {enc_time.avg:.2f} seconds")
            print(f"Average Decoding Time: {dec_time.avg:.2f} seconds")
    else:
        criterion = SeecLoss()
        for imgdir in args.imgdir:
            Nll = AverageMeter()
            X_bpp = AverageMeter()
            for path in os.listdir(imgdir):
                img_path = os.path.join(imgdir, path)

                img = Image.open(img_path).convert("RGB")
                seg = extract_mask(birefnet, img, args.segtype)
                img = transforms.ToTensor()(img).to(device)
                x = img2patch(img, patch_sz=64).to(device)
                seg = img2patch(seg, patch_sz=64).to(device)
                nll = 0
                x_bpp = 0
                for i in range(x.size(0)):
                    x_split = x[i].unsqueeze(0)
                    seg_split = seg[i].unsqueeze(0)
                    out = model(x_split, seg_split)
                    output = criterion(x_split, out)
                    x_bpp += output["x_bpp"].item() * x_split.numel() / x_split.shape[1]
                    nll += output["loss"].item() * x_split.numel() / x_split.shape[1]
                nll = nll / img.size(1) / img.size(2)
                x_bpp = x_bpp / img.size(1) / img.size(2)
                Nll.update(nll)
                X_bpp.update(x_bpp)
            print(f"Results for {imgdir}:")
            print(f"Average X bpp: {X_bpp.avg:.4f}")
            print(f"Average NLL: {Nll.avg:.4f}")


if __name__ == "__main__":
    main()
