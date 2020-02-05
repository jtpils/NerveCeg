from cv2 import cv2

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils

from segmentation.util.dataset import NerveUSDataset, NerveUSDatasetMask
from segmentation.util.transform import real_preprocessing, crop_square
from classification.transform import pred
from segmentation.util.trainer import to_np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)


class RealDataset(Dataset):
    def __init__(self, root='./data/', transforms=None):
        super().__init__()
        self.root = root
        self.transform = transforms

        self.items = []

        for i in range(1, 5509):
            self.items.append(str(i))

    def __getitem__(self, index):
        if not (0 <= index < len(self.items)):
            raise IndexError("Index out of bound")

        src = cv2.imread('./data/real_test/' + str(self.items[index]) + '.tif')

        sample = (src, index)
        print(sample)

        return sample if self.transform is None else self.transform(*sample)

    def __len__(self):
        return len(self.items)
        

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_real_test = RealDataset(root=target_folder, transforms=real_preprocessing)

    with torch.no_grad():
        model = torch.load('save_models/25.pt')
        model = model.cuda(device)
        model.eval()
        cls_model = torch.load('cls/final.pt')

        # for i in range(1, 5509):
        #     X, y = ds_real_test.__getitem__(i)
        #     X = X.view(1, 3, 400, 400).cuda(device)
        #     values, indices = pred(cls_model(X))
        #     print(to_np(indices)[0])
        #     cls_res = to_np(indices)[0]
        #     if cls_res == 1:
        #         # torchvision.utils.save_image(X, './realimage/'+str(i)+'_X.png')
        #         # torchvision.utils.save_image(y, './realimage/'+str(i)+'_y.png')
        #         y_pred = model(X)
        #         torchvision.utils.save_image(y_pred, './realimage/'+str(i)+'_ypred.png')
        #     else:
        #         # torchvision.utils.save_image(X, './realimage/'+str(i)+'_X.png')
        #         # torchvision.utils.save_image(y, './realimage/'+str(i)+'_y.png')
        #         y_pred = torch.zeros_like(X)
        #         torchvision.utils.save_image(y_pred, './realimage/'+str(i)+'_ypred.png')

        
        X, y = ds_real_test.__getitem__(5507)
        X = X.view(1, 3, 400, 400).cuda(device)
        values, indices = pred(cls_model(X))
        print(to_np(indices)[0])
        cls_res = to_np(indices)[0]
        if cls_res == 1:
                # torchvision.utils.save_image(X, './realimage/'+str(i)+'_X.png')
                # torchvision.utils.save_image(y, './realimage/'+str(i)+'_y.png')
            y_pred = model(X)
            torchvision.utils.save_image(y_pred, './realimage/'+str(5507)+'_ypred.png')
        else:
                # torchvision.utils.save_image(X, './realimage/'+str(i)+'_X.png')
                # torchvision.utils.save_image(y, './realimage/'+str(i)+'_y.png')
            y_pred = torch.zeros_like(X)
            torchvision.utils.save_image(y_pred, './realimage/'+str(5507)+'_ypred.png')


    