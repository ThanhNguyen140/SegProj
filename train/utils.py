from torch.utils.data import Dataset
import os
import numpy as np
from train.augmentation import Augment


def extract_data(path):
    files = [file for file in os.listdir(path) if "npz" in file]
    arrays = [np.load(os.path.join(path, file)) for file in files]
    imgs = [array["img"] for array in arrays]
    imgs = np.stack(imgs, axis=0)
    labels = [array["label"] for array in arrays]
    labels = np.stack(labels, axis=0)
    return imgs, labels


class LoadData(Dataset):
    def __init__(self, imgs, labels, mode="train"):
        if mode == "train":
            augment = Augment(imgs, labels)
            aug_imgs, aug_labels = augment()
            self.cubes, self.masks = self.get_data(aug_imgs, aug_labels)
        else:
            self.cubes, self.masks = self.get_data(imgs, labels)

    def get_data(self, imgs, labels):
        x_center, y_center, z_center = 88, 135, 69
        xmin, xmax = x_center - 16, x_center + 16
        ymin, ymax = y_center - 16, y_center + 16
        zmin, zmax = z_center - 16, z_center + 16
        cubes = imgs[:, xmin:xmax, ymin:ymax, zmin:zmax]
        masks = labels[:, xmin:xmax, ymin:ymax, zmin:zmax]

        return cubes, masks

    def __len__(self):
        return len(self.cubes)

    def __getitem__(self, idx):
        img = self.cubes[idx, :, :, :]
        mask = self.masks[idx, :, :, :]
        return img, mask
