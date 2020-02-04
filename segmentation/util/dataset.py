from __future__ import print_function, division
import os
import glob
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from PIL import Image
from cv2 import cv2

import math
import hdf5storage
from enum import Enum
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()


class NerveUSDataset(Dataset):
    def __init__(self, root, train=True):
        """
        Args:
            root (string): Directory of folder which contains subfolder(train, mask)
            train (boolean): If train data, True. If test data, False.
        """
        super().__init__()
        self.root = root
        self.items = []
        self.train = train

        if train == True:
            self.items = os.listdir(root + 'train/image/')
        elif train == False:
            self.items = os.listdir(root + 'test/image/')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if not (0 <= idx <  len(self.items)):
            raise IndexError("Index out of bound")

        if self.train == True:
            targ_img_path = os.path.join(self.root, 'train/', 'image/', self.items[idx])
            targ_mask_path = os.path.join(self.root, 'train/', 'mask/', self.items[idx].split('.')[0] + '_mask' + '.tif')
        elif self.train == False:
            targ_img_path = os.path.join(self.root, 'test/', 'image/', self.items[idx])
            targ_mask_path = os.path.join(self.root, 'test/', 'mask/', self.items[idx].split('.')[0] + '_mask' + '.tif')


        image_data = cv2.imread(targ_img_path)
        # image_data = image_data.transpose(2, 0, 1)
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        mask_data = cv2.imread(targ_mask_path, 0)
        mask_data = np.expand_dims(mask_data, axis=-1)

        image_with_metadata = {
            "image": ToPILImage()(image_data),
            "mask": ToPILImage(mode='L')(mask_data)
        }

        return image_with_metadata


class NerveUSDatasetMask(NerveUSDataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train)
        self.transform = transform

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        sample = (item["image"], item["mask"])
        return sample if self.transform is None else self.transform(*sample)