import torch


def train_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, writer, train_step, clip_grad=None):
    model.train()
    device = next(model.parameters()).device
    for x, seg in train_dataloader:
        x = x.to(device)
        seg = seg.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out = model(x, seg)
        output = criterion(x, out)
        if torch.isnan(output["loss"]):
            print("Nan detected")
            raise ValueError("Nan detected")
        output["loss"].backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        aux_loss = model.seg_img_compressor.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        train_step += 1
        if train_step % 100 == 0:
            writer.add_scalar("train/loss", output["loss"].item(), train_step)
            writer.add_scalar("train/x_bpp", output["x_bpp"].item(), train_step)
            writer.add_scalar("train/latent_bpp", output["latent_bpp"].item(), train_step)
            writer.add_scalar("train/z_bpp", output["z_bpp"].item(), train_step)
            writer.add_scalar("train/y_bpp", output["y_bpp"].item(), train_step)
            writer.add_scalar("train/aux_loss", aux_loss.item(), train_step)

    return train_step


def eval_epoch(model, criterion, eval_dataloader, epoch, writer):
    model.eval()
    device = next(model.parameters()).device

    loss = 0
    x_bpp = 0
    latent_bpp = 0
    z_bpp = 0
    y_bpp = 0
    val_size = 0
    aux_loss = []

    with torch.no_grad():
        for x, seg in eval_dataloader:
            x = x.to(device)
            seg = seg.to(device)
            out = model(x, seg)
            output = criterion(x, out)

            N = x.shape[0]

            loss += output["loss"].item() * N
            x_bpp += output["x_bpp"].item() * N
            latent_bpp += output["latent_bpp"].item() * N
            z_bpp += output["z_bpp"].item() * N
            y_bpp += output["y_bpp"].item() * N
            aux_loss.append(model.seg_img_compressor.aux_loss().item())

            val_size += N

        loss /= val_size
        x_bpp /= val_size
        latent_bpp /= val_size
        z_bpp /= val_size
        y_bpp /= val_size
        aux_loss = sum(aux_loss) / len(aux_loss)

        writer.add_scalar("val/loss", loss, epoch)
        writer.add_scalar("val/x_bpp", x_bpp, epoch)
        writer.add_scalar("val/latent_bpp", latent_bpp, epoch)
        writer.add_scalar("val/z_bpp", z_bpp, epoch)
        writer.add_scalar("val/y_bpp", y_bpp, epoch)
        writer.add_scalar("val/aux_loss", aux_loss, epoch)

    return loss, x_bpp, latent_bpp, z_bpp, y_bpp, aux_loss


def eval_epoch(model, criterion, eval_dataloader, epoch, writer):
    model.eval()
    device = next(model.parameters()).device

    loss = 0
    x_bpp = 0
    latent_bpp = 0
    z_bpp = 0
    y_bpp = 0
    val_size = 0
    aux_loss = []

    with torch.no_grad():
        for x, seg in eval_dataloader:
            x = x.to(device)
            seg = seg.to(device)
            out = model(x, seg)
            output = criterion(x, out)

            N = x.shape[0]

            loss += output["loss"].item() * N
            x_bpp += output["x_bpp"].item() * N
            latent_bpp += output["latent_bpp"].item() * N
            z_bpp += output["z_bpp"].item() * N
            y_bpp += output["y_bpp"].item() * N
            aux_loss.append(model.seg_img_compressor.aux_loss().item())

            val_size += N

        loss /= val_size
        x_bpp /= val_size
        latent_bpp /= val_size
        z_bpp /= val_size
        y_bpp /= val_size
        aux_loss = sum(aux_loss) / len(aux_loss)

        writer.add_scalar("val/loss", loss, epoch)
        writer.add_scalar("val/x_bpp", x_bpp, epoch)
        writer.add_scalar("val/latent_bpp", latent_bpp, epoch)
        writer.add_scalar("val/z_bpp", z_bpp, epoch)
        writer.add_scalar("val/y_bpp", y_bpp, epoch)
        writer.add_scalar("val/aux_loss", aux_loss, epoch)

    return loss, x_bpp, latent_bpp, z_bpp, y_bpp, aux_loss
