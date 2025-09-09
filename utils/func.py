import numpy as np
import torch.nn.functional as F
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import time
import hashlib
import torch.optim as optim


def configure_optimizers(model, lr, aux_lr):

    parameters = {n for n, p in model.named_parameters() if not n.endswith(".quantiles") and p.requires_grad}
    aux_parameters = {n for n, p in model.named_parameters() if n.endswith(".quantiles") and p.requires_grad}
    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_lr,
    )

    return optimizer, aux_optimizer


def img2patch(img, patch_sz):
    if img.dim() == 3:
        img = img.unsqueeze(0)
    B, C, H, W = img.shape
    pad_h = (patch_sz - H % patch_sz) % patch_sz
    pad_w = (patch_sz - W % patch_sz) % patch_sz
    img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0)
    patches = img_pad.unfold(2, patch_sz, patch_sz).unfold(3, patch_sz, patch_sz)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(-1, C, patch_sz, patch_sz)

    return patches


def patch2img(patch, img_sz):
    C = patch.shape[1]
    patch_sz = patch.shape[2]
    H, W = img_sz
    pad_h = (patch_sz - H % patch_sz) % patch_sz
    pad_w = (patch_sz - W % patch_sz) % patch_sz
    rows = (H + pad_h) // patch_sz
    cols = (W + pad_w) // patch_sz
    patch = patch.view(-1, rows, cols, C, patch_sz, patch_sz)
    patch = patch.permute(0, 3, 1, 4, 2, 5).contiguous()
    img = patch.view(-1, C, H + pad_h, W + pad_w)
    img = img[:, :, :H, :W]
    return img


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def coding_table_3p(patch_sz=64):
    COT = torch.zeros(patch_sz, patch_sz, dtype=torch.int64)
    for i in range(patch_sz):
        start = 2 * i + 1
        COT[i, :] = torch.arange(start, start + patch_sz)
    return COT


def check_state_dict(state_dict, unwanted_prefixes=["module.", "_orig_mod."]):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict


def extract_mask(birefnet, image: Image.Image, segtype="norm", threshold=0.5):

    if segtype == "random":
        return torch.randint(0, 2, (1, image.size[1], image.size[0]), dtype=torch.uint8)

    image_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    input_images = transform_image(image).unsqueeze(0).to("cuda").half()
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    mask = transforms.ToTensor()(mask)
    mask = (mask > threshold).to(torch.uint8)  # Thresholding to create binary mask
    if segtype == "wrong":
        mask = 1 - mask
    return mask


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Timer:
    def __init__(self, results_dict, key):
        self.results = results_dict
        self.key = key

    def __enter__(self):
        self.start = time.time()
        return self  # Optional

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.results[self.key] = time.time() - self.start


def get_md5(*paths):
    key = ""
    for path in paths:
        key += path
    return hashlib.md5(key.encode()).hexdigest()


class AverageMeter:

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
