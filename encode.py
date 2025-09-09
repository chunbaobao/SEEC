import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import torchac
from model.seec import SeecNet
import pickle
import os
from utils.func import img2patch, check_state_dict, extract_mask, Timer, coding_table_3p
from model_hub.models.birefnet import BiRefNet
import imagecodecs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
patch_sz = 64
norm_scale = 1.0 / 255.0 * 2.0
half = 0.5 * norm_scale
mix_num = 5
mix_num2 = mix_num * 2
samples = torch.arange(0, 256, dtype=torch.float32).to(device)
samples = samples * norm_scale


def compress(model: SeecNet, birefnet: BiRefNet, input_path: str, COT, segtype: str = "norm"):

    x_stream = []
    results = {}

    img = Image.open(input_path).convert("RGB")

    with Timer(results, "seg_extract_time"):
        torch.cuda.synchronize()
        seg = extract_mask(birefnet, img, segtype)

    with Timer(results, "compress_time"):
        hw = img.size[0] * img.size[1]

        img = transforms.PILToTensor()(img).to(device).unsqueeze(0)
        img_shape = img.shape[-2:]
        is_padding = img_shape[0] % patch_sz != 0 or img_shape[1] % patch_sz != 0
        x = img2patch(img, patch_sz=patch_sz).to(device)
        seg_patch = img2patch(seg, patch_sz=patch_sz).to(device)
        if is_padding:
            code_flag = img2patch(torch.ones_like(seg), patch_sz=patch_sz).to(device)
        with torch.no_grad():

            model.eval()

            latent_code = model.seg_img_compressor.compress(x / 255.0)
            prior_total = model.seg_img_compressor.decompress(**latent_code)["prior"]

            context_total = model.mask_conv(x * norm_scale)
            B = x.shape[0]
            max_step = torch.max(COT)
            for i in range(max_step):
                h_idx, w_idx = torch.nonzero(COT == i + 1, as_tuple=True)
                context = context_total[:, :, h_idx, w_idx].unsqueeze(3)
                prior = prior_total[:, :, h_idx, w_idx].unsqueeze(3)
                x_crop = x[:, :, h_idx, w_idx].unsqueeze(3)
                seg_crop = seg_patch[:, :, h_idx, w_idx].unsqueeze(3)
                fusion_context = model.fusion(torch.cat([prior, context], dim=1))
                lmm_params = model.ep(fusion_context, seg_crop)
                mu, log_sigma, coeffs, weights = torch.split(lmm_params, 15, dim=1)
                if model.no_multichannel_lmm:
                    weights = weights.reshape(B, 1, mix_num, -1, 1)
                    weights = weights.repeat(1, 3, 1, 1, 1)
                else:
                    weights = weights.reshape(B, 3, mix_num, -1, 1)
                coeffs = torch.tanh(coeffs)

                for c in range(3):
                    if c == 0:
                        mu_c = mu[:, :mix_num, :, :].permute(0, 2, 1, 3)

                    elif c == 1:
                        mu_c = (
                            mu[:, mix_num:mix_num2, :, :] + (x_crop[:, 0:1, :] * norm_scale) * coeffs[:, :mix_num, :, :]
                        )
                        mu_c = mu_c.permute(0, 2, 1, 3)
                    else:
                        mu_c = (
                            mu[:, mix_num2:, :, :]
                            + (x_crop[:, 0:1, :] * norm_scale) * coeffs[:, mix_num:mix_num2, :, :]
                            + (x_crop[:, 1:2, :] * norm_scale) * coeffs[:, mix_num2:, :, :]
                        )
                        mu_c = mu_c.permute(0, 2, 1, 3)
                    samples_centered = samples - mu_c
                    inv_sigma = torch.exp(-log_sigma[:, c * mix_num : (c + 1) * mix_num, :, :].permute(0, 2, 1, 3))
                    plus_in = inv_sigma * (samples_centered + half)
                    cdf_plus = torch.sigmoid(plus_in)
                    min_in = inv_sigma * (samples_centered - half)
                    cdf_min = torch.sigmoid(min_in)
                    cdf_delta = cdf_plus - cdf_min
                    one_minus_cdf_min = torch.exp(-F.softplus(min_in))
                    cdf_plus = torch.exp(plus_in - F.softplus(plus_in))

                    samples2 = samples - torch.zeros_like(mu_c)
                    cdf_delta = torch.where(
                        samples2 - half < 0.001,
                        cdf_plus,
                        torch.where(samples2 + half > 1.999, one_minus_cdf_min, cdf_delta),
                    )

                    weights_c = weights[:, c, :, :, :].permute(0, 2, 1, 3)
                    m = torch.amax(weights_c, 2, keepdim=True)
                    weights_c = torch.exp(
                        weights_c - m - torch.log(torch.sum(torch.exp(weights_c - m), 2, keepdim=True))
                    )
                    pmf = torch.sum(cdf_delta * weights_c, dim=2)

                    pmf = pmf.clamp_(1.0 / 64800, 1.0)
                    pmf = pmf / torch.sum(pmf, dim=2, keepdim=True)
                    cdf = torch.cumsum(pmf, dim=2).clamp_(0.0, 1.0)
                    cdf = F.pad(cdf, (1, 0))
                    symbol = x_crop[:, c].short().reshape(B, -1)
                    if is_padding:
                        cdf = cdf[code_flag[:, :, h_idx, w_idx].squeeze(1).bool() == 1]
                        symbol = symbol[code_flag[:, :, h_idx, w_idx].squeeze(1).bool() == 1]
                    stream = torchac.encode_float_cdf(
                        cdf.cpu(), symbol.cpu(), needs_normalization=False, check_input_bounds=False
                    )
                    x_stream.append(stream)

        # print("compress time (min):", (time_end - time_start) / 60)

    with Timer(results, "seg_enc_time"):
        seg_bin = imagecodecs.jpegxl_encode(transforms.ToPILImage()(seg.squeeze(0).cpu().byte()))
        torch.cuda.synchronize()

    y_len = sum(len(latent_code["strings"][0][i]) for i in range(len(latent_code["strings"][0])))
    z_len = sum(len(latent_code["strings"][1][i]) for i in range(len(latent_code["strings"][1])))

    x_len = sum([len(x_stream[i]) for i in range(len(x_stream))])

    results["z_bpp"] = z_len * 8 / hw
    results["y_bpp"] = y_len * 8 / hw
    results["x_bpp"] = x_len * 8 / hw
    results["seg_bpp"] = len(seg_bin) * 8 / hw
    results["lantent_bpp"] = (y_len + z_len) * 8 / hw

    results["bpp"] = (
        results["lantent_bpp"] + results["x_bpp"] + results["seg_bpp"] + 6 * 2 * 8 / hw
    )  # lantent stream + x stream + z_shape + x_shape
    results["enc_time"] = results["compress_time"] + results["seg_extract_time"] + results["seg_enc_time"]

    return latent_code, x_stream, seg_bin, img_shape, results


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
    parser.add_argument("--input", "--i", type=str, help="Directory containing images to encode")
    parser.add_argument("--output", "--o", type=str, default="tmp/temp", help="Output path to save the results")

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
    COT = coding_table_3p(patch_sz=64).to(device)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    latent_code, x_stream, seg_bin, img_shape, results = compress(model, birefnet, args.input, COT, args.segtype)
    print("Results:", results)
    print("Compression completed. Results saved to:", args.output)
    with open(args.output, "wb") as f:
        pickle.dump((latent_code, seg_bin, x_stream, img_shape), f)


if __name__ == "__main__":
    main()
