from torch.utils.data import Dataset
import os
import numpy as np
import logging
from tqdm import tqdm
import nibabel as nib

class DataCustomer(Dataset):
    def __init__(self, img_path, names, x_centers, y_centers, z_centers):
        self.img_path = img_path
        self.names = names
        self.x_centers = x_centers
        self.y_centers = y_centers
        self.z_centers = z_centers
        self.get_data()
        
    def get_data(self):
        patches = []
        logging.info('Loading data...')
        for idx, name in tqdm(enumerate(self.names)):
            data = nib.load(os.path.join(self.img_path,f'{name}.nii.gz'))
            img = data.get_fdata()
            normalized = (img - img.mean()) / img.std()
            x, y, z = int(self.x_centers[idx]), int(self.y_centers[idx]), int(self.z_centers[idx])
            wx = 32
            wy = 16
            wz = 8
            patch = normalized[x-wx:x+wx, y-wy:y+wy, z-wz:z+wz]
            patches.append(patch)
        self.patches = np.stack(patches)
        return self.names, self.patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        name = self.names[idx]
        patch = self.patches[idx,:,:,:]
        return name, patch