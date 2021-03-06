import random

import torch
import torch.nn as nn
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import NerveClassificationDataset
from transform import preprocessing, pred
from efficientnet_pytorch import EfficientNet
from trainer import to_np

from CNNVis.misc_functions import save_class_activation_images, convert_to_grayscale, save_gradient_images
from CNNVis.gradcam import GradCam
from CNNVis.vanilla_backprop import VanillaBackprop


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated(device=device)
    ngpus_per_node = torch.cuda.device_count()
    target_folder = './data/'

    efficient_list = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']

    ds_test = NerveClassificationDataset(root=target_folder, train=False, transform=preprocessing)
    # model = EfficientNet.from_name(efficient_list[4])
    model = torch.load('final.pt')
    # model = model.cuda()
    model.eval()

    pick = []
    for i in range(1):
        pick.append(random.randrange(0, 1000, 1))

    for i in pick:
        X, y = ds_test.__getitem__(i)
        print()
        X = X.view(1, 3, 400, 400).cuda('cuda:1')
        values, indices = pred(model(X))
        
        ''' for models with features
        grad_cam = GradCam(model, target_layer=11)
        # Generate cam mask
        cam = grad_cam.generate_cam(X, y)
        # Save mask
        save_class_activation_images(X, cam, str(i)+'grad_cam')
        print('Grad cam completed')
        
        # Vanilla backprop
        VBP = VanillaBackprop(model)
        # Generate gradients
        vanilla_grads = VBP.generate_gradients(X, y)
        # Save colored gradients
        save_gradient_images(vanilla_grads, str(i) + '_Vanilla_BP_color')
        # Convert to grayscale
        grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
        # Save grayscale gradients
        save_gradient_images(grayscale_vanilla_grads, str(i) + '_Vanilla_BP_gray')
        print('Vanilla backprop completed')
        '''

        # print("y_pred: " + str(y_pred))
        print("raw: ", end='')
        print(values, indices)
        print("y_pred:", str(to_np(indices)[0]))
        print("y: " + str(y))
        