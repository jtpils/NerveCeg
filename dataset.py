import os
import pandas as pd
import numpy as np
from PIL import Image
from cv2 import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils


class NerveClassificationDataset(Dataset):
    def __init__(self, root='./data/', train=True, transform=None):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform

        self.items = []
        self.label = []

        if train == True:
            trc = pd.read_csv(root + 'train_label.csv')
            self.items = trc['image']
            self.label = trc['label']
        elif train == False:
            trc = pd.read_csv(root + 'test_label.csv')
            self.items = trc['image']
            self.label = trc['label']

    def __getitem__(self, index):
        if not (0 <= index < len(self.items)):
            raise IndexError("Index out of bound")

        if self.train == True:
            src = cv2.imread('./data/train/image/' + str(self.items[index]) + '.tif')
            label = int(self.label[index])
        elif self.train == False:
            src = cv2.imread(os.path.join('./data/test/image', self.items[index]) + '.tif')
            label = int(self.label[index])

        sample = (src, label)

        return sample if self.transform is None else self.transform(*sample)

    def __len__(self):
        return len(self.items)
