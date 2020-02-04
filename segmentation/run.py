import random

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from segmentation.util.dataset import NerveUSDataset, NerveUSDatasetMask
from segmetnation.util.transform import preprocessing


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_test = NerveUSDatasetMask(root=target_folder, train=False, transform=preprocessing)

    model = torch.load('ckpt_10.pt')
    model = model.cuda()
    model.eval()

    pick = []
    for i in range(1):
        pick.append(random.randrange(0, 1000, 1))

    for i in pick:
        X, y = ds_test.__getitem__(i)
        torchvision.utils.save_image(X, './image/'+str(i)+'_X.png')
        torchvision.utils.save_image(y, './image/'+str(i)+'_y.png')
        X = X.view(1, 3, 400, 400).cuda()
        y_pred = model(X)
        torchvision.utils.save_image(y_pred, './image/'+str(i)+'_ypred.png')
