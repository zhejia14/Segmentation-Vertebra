from torchvision import transforms
import torch
from PIL import Image, ImageEnhance
import random

class AdjustSharpness:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        img = transforms.ToTensor()(img)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(self.factor)
        return transforms.ToTensor()(img)


def augment_Sharpness(img, mask):
    img = AdjustSharpness(factor=5.0)(img)

    return img, mask

def augment_HorizontalFlip(img, mask):
    img = transforms.RandomHorizontalFlip(p=1)(img)
    mask = transforms.RandomHorizontalFlip(p=1)(mask)

    return img, mask


def augment_VerticalFlip(img, mask):
    img = transforms.RandomVerticalFlip(p=1)(img)
    mask = transforms.RandomVerticalFlip(p=1)(mask)

    return img, mask


def augment_Contrast(img, mask):
    img = transforms.ColorJitter(contrast=3.5)(img)

    return(img, mask)


def augment_RandomFlip(img, mask):
    if torch.rand(1) > 0.5:
        img, mask = augment_HorizontalFlip(img, mask)
    else:
        img, mask = augment_VerticalFlip(img, mask)
    return img, mask

def Random_augment(img, mask):
    augmentations = [
    augment_Sharpness,
    augment_HorizontalFlip,
    augment_VerticalFlip,
    augment_Contrast,
    ]

    augmentation = random.choice(augmentations)
    img, mask = augmentation(img, mask)
    
    return img, mask
