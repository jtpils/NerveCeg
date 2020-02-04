import os
import random

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import PIL
from cv2 import cv2

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp

from segmentation.util.dataset import NerveUSDataset, NerveUSDatasetMask
from segmentation.util.trainer import Trainer
from segmentation.util.transform import preprocessing
from segmentation.model.models import U_Net, R2U_Net, AttU_Net, R2AttU_Net
from segmentation.etc.loss import dice_loss
from segmentation.etc.metric import dice_coeff
from segmentation.vis.vis import plot_fit, dataset_first_n


torch.manual_seed(0)
num_epochs = 35
batch_size = 5
learning_rate = 0.001
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    ds_train = NerveUSDatasetMask(root=target_folder, train=True, transform=preprocessing)
    ds_test = NerveUSDatasetMask(root=target_folder, train=False, transform=preprocessing)
    
    # dataset_first_n(ds_train, 4)
    
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=True)

    model = U_Net()
    # model = torch.nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = dice_loss
    success_metric = dice_coeff
    trainer = Trainer(model, criterion, optimizer, dice_coeff, device, None)
    fit = trainer.fit(dl_train, dl_test, num_epochs=num_epochs, checkpoints='save_models/' + model.__class__.__name__)
    torch.save('final.pt')

    plot_fit(fit)

    loss_fn_name = "dice_loss"
    best_score = str(fit.best_score)
    print(f"Best loss score(loss function = {loss_fn_name}): {best_score}")

    plt.plot(fit.train_acc)
    plt.plot(fit.test_acc)
    plt.legend(['train score', 'test score'])
    plt.show()
    plt.savefig('train_n_test_score.png')


