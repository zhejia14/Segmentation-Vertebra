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

        img = cv.imread(img_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        mask = cv.imread(mask_path, 0)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        # if self.target_transform is not None:
        #    target = self.target_transform(target)

        return (img, mask)


def combine_fold(fold_a_path, fold_b_path, fold_a_num, fold_b_num):
    if not os.path.exists(fold_a_path):
        print("Fold one is not exit.")
        exit()
    if not os.path.exists(fold_b_path):
        print("Fold two is not exit.")
        exit()

    combine_path = "./dataset/combine"+str(fold_a_num)+str(fold_b_num)

    if not os.path.exists(combine_path):
        os.makedirs(combine_path)

    for img in os.listdir(os.path.join(fold_a_path, "image")):
        src = os.path.join(fold_a_path, "image", img)
        prefix_dst = combine_path+"/image/"
        if not os.path.exists(prefix_dst):
            os.makedirs(prefix_dst)
        dst = os.path.join(prefix_dst, img)
        shutil.copy(src, dst)

    for img in os.listdir(os.path.join(fold_a_path, "label")):
        src = os.path.join(fold_a_path, "label", img)
        prefix_dst = combine_path+"/label/"
        if not os.path.exists(prefix_dst):
            os.makedirs(prefix_dst)
        dst = os.path.join(prefix_dst, img)
        shutil.copy(src, dst)

    for img in os.listdir(os.path.join(fold_b_path, "image")):
        src = os.path.join(fold_b_path, "image", img)
        dst = os.path.join(combine_path, "image", img)
        shutil.copy(src, dst)

    for img in os.listdir(os.path.join(fold_b_path, "label")):
        src = os.path.join(fold_b_path, "label", img)
        dst = os.path.join(combine_path, "label", img)
        shutil.copy(src, dst)

    return combine_path
