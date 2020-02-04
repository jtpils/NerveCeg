import cv2
from PIL import Image
import torch
from torchvision.transforms import transforms


def crop_square(im):
    width, height = im.size

    new_width, new_height = 400, 400

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

def preprocessing(image, mask):
    image, mask = crop_square(image), crop_square(mask)
    mask_transformer = transforms.Compose([
        transforms.ToTensor()
        
    ])

    image_transformer = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: normalize(x))
    ])

    return image_transformer(image).float(), mask_transformer(mask).float()
