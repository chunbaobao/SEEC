import torch
import torchvision.transforms as transforms
import random
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from utils.func import img2patch


class RandomCropWithSeg:
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, seg):
        rect = transforms.RandomCrop.get_params(img, output_size=self.size)
        img = TF.crop(img, *rect)
        seg = TF.crop(seg, *rect)
        return img, seg


class RandomHorizontalFlipWithSeg:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, seg):
        if random.random() < self.p:
            img = TF.hflip(img)
            seg = TF.hflip(seg)
        return img, seg


class RandomVerticalFlipWithSeg:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, seg):
        if random.random() < self.p:
            img = TF.vflip(img)
            seg = TF.vflip(seg)
        return img, seg


class TrainTransform:
    def __init__(self, crop_size=64, p_flip=0.5, p_vflip=0.5):
        self.crop = RandomCropWithSeg(crop_size)
        self.hflip = RandomHorizontalFlipWithSeg(p_flip)
        self.vflip = RandomVerticalFlipWithSeg(p_vflip)
        self.to_tensor = transforms.ToTensor()
        self.pil_to_tensor = transforms.PILToTensor()
        # self.brightness = transforms.ColorJitter(brightness=0.3)

    def __call__(self, img, seg):
        img, seg = self.crop(img, seg)
        img, seg = self.hflip(img, seg)
        img, seg = self.vflip(img, seg)
        img = self.to_tensor(img)
        # img = self.brightness(img)
        seg = self.pil_to_tensor(seg)
        return img, seg.long()


class EvalTransform:
    def __init__(self, crop_size=64):
        self.crop = transforms.CenterCrop(crop_size)
        self.to_tensor = transforms.ToTensor()
        self.pil_to_tensor = transforms.PILToTensor()

    def __call__(self, img, seg):
        img = self.crop(img)
        seg = self.crop(seg)
        img = self.to_tensor(img)
        seg = self.pil_to_tensor(seg)
        return img, seg.long()


class TestTransform:
    def __init__(self, only_back_front):
        self.pil_to_tensor = transforms.PILToTensor()
        self.only_back_front = only_back_front

    def __call__(self, img, seg=None):
        img = self.pil_to_tensor(img)
        if seg:
            seg = self.pil_to_tensor(seg)
        else:
            seg = torch.zeros(img.shape[1:]).unsqueeze(0)
        if self.only_back_front:
            seg = seg > 0
        return img, seg.long()


class DryRunTransform:
    def __init__(self, only_back_front, patch_sz=64, fix_seg=None):
        self.to_tensor = transforms.ToTensor()
        self.pil_to_tensor = transforms.PILToTensor()
        self.only_back_front = only_back_front
        self.patch_sz = patch_sz
        self.fix_seg = fix_seg

    def __call__(self, img, seg):
        img = self.to_tensor(img)
        seg = self.pil_to_tensor(seg)
        img = img2patch(img, self.patch_sz)
        seg = img2patch(seg, self.patch_sz)
        if self.only_back_front:
            seg = seg > 0
        if self.fix_seg is not None:
            seg = torch.ones_like(seg) * self.fix_seg

        return img, seg.long()


base_transform = transforms.Compose(
    [
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
    ]
)

base_transform_eval = transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor()])
