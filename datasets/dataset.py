import os
from torch.utils.data import Dataset
import PIL.Image as Image


class ImgMaskDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_path = os.path.join(root, "images")
        self.mask_path = os.path.join(root, "masks")
        self.imgs = os.listdir(self.img_path)
        self.imgs.sort()
        self.masks = os.listdir(self.mask_path)
        self.masks.sort()

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.imgs[index])
        mask_path = os.path.join(self.mask_path, self.masks[index])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)
