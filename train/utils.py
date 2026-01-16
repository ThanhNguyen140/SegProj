from torch.utils.data import Dataset
import os
import numpy as np
import logging
from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    RandAdjustContrastd, RandGaussianSmoothd, EnsureTyped,Rand3DElasticd, RandShiftIntensityd
)

def extract_data(path, debug = False):
    files = [file for file in os.listdir(path) if "npz" in file]
    if not debug: 
        arrays = [np.load(os.path.join(path, file)) for file in files]
    else:
        arrays = [np.load(os.path.join(path, file)) for file in files[:50]]
    imgs = [array["img"] for array in arrays]
    imgs = np.stack(imgs, axis=0)
    labels = [array["label"] for array in arrays]
    labels = np.stack(labels, axis=0)
    #print(imgs.shape, labels.shape)
    return imgs, labels


class LoadData(Dataset):
    def __init__(self, imgs, labels, mode="train"):
        self.mode = mode
        # Define MONAI augmentation pipeline
        self.transform = Compose([
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.3),
            RandRotate90d(keys=['image', 'label'],prob=0.4,spatial_axes=(0, 1)),
            EnsureTyped(keys=["image", "label"]),
        self.get_data(imgs,labels)

    def get_data(self, imgs, labels):
        #x_center, y_center, z_center = 88, 135, 69
        #xmin, xmax = x_center - int(self.w/2), x_center + int(self.w/2)
        #ymin, ymax = y_center - int(self.w/2), y_center + int(self.w/2)
        #zmin, zmax = z_center - 8, z_center + 8
        #imgs = imgs[:,:, xmin:xmax, ymin:ymax, zmin:zmax]
        #labels = labels[:,:, xmin:xmax, ymin:ymax, zmin:zmax]
        if self.mode == "train":
            cubes, masks = imgs[:,:8,:,:,:], labels[:,:8,:,:,:]
            #print('Train:',cubes.shape)
            cubes = np.concatenate([arr for arr in cubes], axis = 0)
            masks = np.concatenate([arr for arr in masks], axis = 0)
            sample = {'image':cubes,'label':masks}
            transformed = self.transform(sample)
            cubes, masks = transformed['image'], transformed['label']
        else:
            cubes, masks = imgs[:,0,:,:,:], labels[:,0,:,:,:]
        has_label = masks.reshape(masks.shape[0],-1).sum(axis = 1) > 0
        self.cubes = cubes[has_label]
        self.masks = masks[has_label]
        logging.info(f'Data shape:{self.cubes.shape, }')
        return self.cubes, self.masks

    def __len__(self):
        return len(self.cubes)

    def __getitem__(self, idx):
        img = self.cubes[idx, :, :, :]
        mask = self.masks[idx, :, :, :]
        return img, mask
