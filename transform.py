import torchvision.transforms as transforms
from cv2 import cv2


def crop_square(im):
    size = im.shape

    new_width, new_height = 380, 380

    left = (size[0] - new_width)/2
    top = (size[1] - new_height)/2
    right = (size[0] + new_width)/2
    bottom = (size[1] + new_height)/2
    print(left, top, right, bottom)
    im = im[int(left):int(right), int(top):int(bottom)]

    cv2.imshow('123', im)
    cv2.waitKey()

    return im

def preprocessing(image, label):
    image = image / 255.0
    image = crop_square(image)
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    
    return image_transform(image).float(), label