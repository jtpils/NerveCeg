import cv2
from PIL import Image
import torch
from torchvision.transforms import transforms


def crop_square(im, size):
    width, height = im.size

    new_width, new_height = size, size

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    im = im.crop((left, top, right, bottom))

    return im

def crop_square_cv(im, tsize):
    size = im.shape

    new_width, new_height = tsize, tsize

    left = (size[0] - new_width)/2
    top = (size[1] - new_height)/2
    right = (size[0] + new_width)/2
    bottom = (size[1] + new_height)/2
    
    im = im[int(left):int(right), int(top):int(bottom)]

    return im

def normalize(x,  mean=500, std=None):
    mean_tensor = torch.ones_like(x) * mean
    x -= mean_tensor
    if std:
        x /= std
    return x

def pred_proc(pred):
    values, indices = pred.max(1)

    return values, indices

def preprocessing(image, mask):
    image, mask = crop_square(image, 400), crop_square(mask, 400)
    mask_transformer = transforms.Compose([
        transforms.ToTensor()
        
    ])

    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: normalize(x))
    ])

    return image_transformer(image).float(), mask_transformer(mask).float()

def real_preprocessing(image, index):
    image = crop_square_cv(image, 400)

    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: normalize(x))
    ])

    return image_transformer(image).float(), index
