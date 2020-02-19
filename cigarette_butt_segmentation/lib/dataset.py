import json
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from lib.utils import get_mask


class BasicDataset(Dataset):
    def __init__(self, imgs_path, img_format, transform=None, augument=None, scale=1):

        self.path = imgs_path
        self.images = listdir(f"{self.path}/images")
        self.annotations = json.load(open(f"{self.path}/coco_annotations.json", "r"))
        self.img_format = img_format
        self.transform = transform
        self.augument = augument

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):

        img_id = int(self.images[i].split(".")[0])

        opened_img = Image.open(f"{self.path}/images/{img_id:08}.{self.img_format}")

        if opened_img.mode == 'RGBA':
            opened_img = opened_img.convert('RGB')

        image = np.asarray(opened_img)
        mask = get_mask(img_id, self.annotations)

        if self.augument:
            augmented = self.augument(image=image, mask=mask)
            # Access augmented image and mask
            image = augmented['image']
            mask = augmented['mask']

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask)

        return image, mask


class TestDataset(Dataset):
    def __init__(self, imgs_path, img_format, null_number, transform=None):

        self.path = imgs_path
        self.images = listdir(f"{self.path}")
        self.img_format = img_format
        self.transform = transform
        self.null_number = null_number

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):

        img_id = int(self.images[i].split(".")[0])
        opened_img = Image.open(f"{self.path}/{img_id:0{self.null_number}}.{self.img_format}")

        if opened_img.mode == 'RGBA':
            opened_img = opened_img.convert('RGB')

        if self.transform:
            opened_img = self.transform(opened_img)

        return img_id, opened_img
