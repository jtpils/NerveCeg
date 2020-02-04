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
