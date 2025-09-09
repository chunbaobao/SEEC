import numpy as np
from PIL import Image
import os
import tqdm
import torch
from model_hub.models.birefnet import BiRefNet
import argparse
from utils.func import check_state_dict, extract_mask, check_path


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare data for BiRefNet")
    parser.add_argument(
        "--birefnet_ckpt",
        type=str,
        default="model_hub/BiRefNet-general-epoch_244.pth",
        help="Path to the BiRefNet checkpoint",
    )
    parser.add_argument("--div_path_train", type=str, help="Path to the DIV2K training dataset")
    parser.add_argument("--div_path_val", type=str, help="Path to the DIV2K validation dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Path to save the processed data",
    )
    return parser.parse_args()


args = parse_args()

PATH_TO_WEIGHT = args.birefnet_ckpt

birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load(PATH_TO_WEIGHT, map_location="cpu")
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)
torch.set_float32_matmul_precision(["high", "highest"][0])
birefnet.to("cuda")
birefnet.eval()
birefnet.half()


div_path_train = args.div_path_train
div_path_val = args.div_path_val
data_path = args.output
paths = [div_path_train, div_path_val]

patch_size = 128
counter = 1
for path in paths:
    img_path = os.path.join(data_path, path.split("/")[-1].replace("HR", f"p{patch_size}"), "images")
    mask_path = os.path.join(data_path, path.split("/")[-1].replace("HR", f"p{patch_size}"), "masks")
    check_path(img_path)
    check_path(mask_path)

    files = sorted(os.listdir(path))
    print(path)

    for file in tqdm.tqdm(files):
        file_path = os.path.join(path, file)
        I = Image.open(file_path)
        mask = extract_mask(birefnet, I, threshold=0.5).squeeze(0)
        w, h = I.size
        im = np.array(I)
        ma = np.array(mask)
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                if i + patch_size <= h and j + patch_size <= w:
                    im_p = im[i : i + patch_size, j : j + patch_size, :]
                    ma_p = ma[i : i + patch_size, j : j + patch_size]

                    I_p = Image.fromarray(im_p)
                    M_p = Image.fromarray(ma_p)
                    filename = f"{counter:06d}.png"
                    I_p.save(
                        os.path.join(img_path, filename),
                        compress_level=0,
                    )
                    M_p.save(
                        os.path.join(mask_path, filename),
                        compress_level=0,
                    )
                    counter += 1
print("Total patches:", counter - 1)
