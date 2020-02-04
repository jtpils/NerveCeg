import random

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, ToPILImage

from segmentation.util.dataset import NerveUSDataset, NerveUSDatasetMask
from segmentation.util.transform import preprocessing, pred_proc, crop_square
from segmentation.util.trainer import to_np




device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_test = NerveUSDatasetMask(root=target_folder, train=False, transform=preprocessing)
    
    with torch.no_grad():
        model = torch.load('save_models/1.pt')
        model = model.cuda(device)
        model.eval()
        cls_model = torch.load_state_dict(torch.load('ckpt_10.pt'))
        pick = []
        for i in range(1):
            pick.append(random.randrange(0, 1000, 1))

        for i in pick:
            X, y = ds_test.__getitem__(i)
            X_t = ToTensor()(crop_square(ToPILImage()(X), 380))
            X_t = X_t.view(1, 3, 380, 380).cuda(device)
            _, indices = pred_proc(cls_model(X_t))
            if to_np(indices)[0] == 1:
                torchvision.utils.save_image(X, './image/'+str(i)+'_X.png')
                torchvision.utils.save_image(y, './image/'+str(i)+'_y.png')
                y_pred = model(X)
                torchvision.utils.save_image(y_pred, './image/'+str(i)+'_ypred.png')
            else:
                torchvision.utils.save_image(X, './image/'+str(i)+'_X.png')
                torchvision.utils.save_image(y, './image/'+str(i)+'_y.png')
                y_pred = torch.zeros_like(y)
                torchvision.utils.save_image(y_pred, './image/'+str(i)+'_ypred.png')

