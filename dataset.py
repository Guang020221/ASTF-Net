# coding: utf-8
import os
import cv2
import torch
import random
import torch.utils.data as data
import numpy as np
from option import args


def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        img = np.flipud(np.rot90(img))
    elif mode == 2:
        img = np.flipud(img)
    elif mode == 3:
        img = np.rot90(img, k=3)
    elif mode == 4:
        img = np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        img = np.rot90(img)
    elif mode == 6:
        img = np.rot90(img, k=2)
    elif mode == 7:
        img = np.flipud(np.rot90(img, k=3))

    return np.ascontiguousarray(img)


class Trainset(data.Dataset):
    def __init__(self, transform):
        super(Trainset, self).__init__()
        self.dir_prefix = args.dir_train
        self.lr_over = os.listdir(os.path.join(self.dir_prefix, 'over'))
        self.lr_over.sort()

        self.lr_under = os.listdir(os.path.join(self.dir_prefix, 'under'))
        self.lr_under.sort()

        self.patch_size = args.patch_size
        self.transform = transform

    def __len__(self):
        assert len(self.lr_over) == len(self.lr_under)
        return len(self.lr_over)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.dir_prefix, 'over', self.lr_over[idx])
        img6_path = os.path.join(self.dir_prefix, 'under', self.lr_under[idx])

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img6 = cv2.imread(img6_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None:
            print(f"Error loading image: {img1_path}")
            return None
        if img6 is None:
            print(f"Error loading image: {img6_path}")
            return None

        img1, img6 = self.get_patch(img1, img6)

        img1 = img1.astype(np.float32) / 255.0
        img6 = img6.astype(np.float32) / 255.0

        mode = random.randint(0, 7)
        img1 = augment_img(img1, mode=mode)
        img6 = augment_img(img6, mode=mode)

        img1 = np.ascontiguousarray(img1)
        img6 = np.ascontiguousarray(img6)

        img1 = torch.from_numpy(img1).unsqueeze(0).float()
        img6 = torch.from_numpy(img6).unsqueeze(0).float()

        seq = [img1, img6]
        return seq

    def get_patch(self, l_over, l_under):
        lh, lw = l_over.shape[:2]
        l_stride = self.patch_size

        x = random.randint(0, lw - l_stride)
        y = random.randint(0, lh - l_stride)

        l_over = l_over[y:y + l_stride, x:x + l_stride]
        l_under = l_under[y:y + l_stride, x:x + l_stride]

        return l_over, l_under


class TestData(data.Dataset):
    def __init__(self, transform):
        super(TestData, self).__init__()
        self.dir_prefix = args.dir_test
        self.lr_over = os.listdir(os.path.join(self.dir_prefix, 'testA'))
        self.lr_under = os.listdir(os.path.join(self.dir_prefix, 'testB'))
        self.lr_over.sort()
        self.lr_under.sort()
        self.transform = transform

    def __len__(self):
        assert len(self.lr_over) == len(self.lr_under)
        return len(self.lr_over)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.dir_prefix, 'testA', self.lr_over[idx])
        img6_path = os.path.join(self.dir_prefix, 'testB', self.lr_under[idx])

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img6 = cv2.imread(img6_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None:
            print(f"Error loading image: {img1_path}")
            return None
        if img6 is None:
            print(f"Error loading image: {img6_path}")
            return None

        img1 = img1.astype(np.float32) / 255.0
        img6 = img6.astype(np.float32) / 255.0

        img1 = np.ascontiguousarray(img1)
        img6 = np.ascontiguousarray(img6)

        img1 = torch.from_numpy(img1).unsqueeze(0).float()
        img6 = torch.from_numpy(img6).unsqueeze(0).float()

        seq = [img1, img6, img1, img6]
        return seq