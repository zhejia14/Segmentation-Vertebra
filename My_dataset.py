from torch.utils.data import Dataset
from pycocotools.coco import COCO
import os
import cv2 as cv
import shutil
from imutils import paths


class CustomDataset(Dataset):
    def __init__(self, imgs_path, mask_path, transform=None, target_trnsform=None):
        # self.imgs_path = imgs_path
        # self.mask_path = mask_path
        # self.ann = COCO(annfile_path)
        # self.ids = list(self.ann.imgs.keys())
        # self.classes = self.ann.cats.items()
        self.imgs_path = sorted(list(paths.list_images(imgs_path)))
        self.mask_path = sorted(list(paths.list_images(mask_path)))
        self.transform = transform
        self.target_transform = target_trnsform

    def __len__(self):
        # return len(self.ids)
        return len(self.imgs_path)

    def __getitem__(self, idx):
        # img_id = self.ids[idx]
        # ann_ids = self.ann.getAnnIds(imgIds=img_id)
        # target = self.ann.loadAnns(ann_ids)
        # img_name = self.ann.loadImgs(img_id)[0]['file_name']
        # img_path = os.path.join(self.imgs_path, img_name)
        # mask_path = os.path.join(self.mask_path, img_name)

        img_path = self.imgs_path[idx]
        mask_path = self.mask_path[idx]

        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        # if self.target_transform is not None:
        #    target = self.target_transform(target)

        return (img, mask)

class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, augment_transform):
        self.original_dataset = original_dataset
        self.augment_transform = augment_transform
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        img, mask = self.original_dataset[idx]
        img, mask = self.augment_transform(img, mask)

        return (img, mask)
