import numpy as np
import cv2
import nibabel as nib
from tools import label_props
import os

def convert_rgb(img):
    new_img = []
    img_trans = img.transpose()
    for n in range(img_trans.shape[0]):
        slice = img_trans[n,:,:]
        normalized = cv2.normalize(slice.astype(np.float32),None,alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        rgb = cv2.cvtColor(normalized,cv2.COLOR_GRAY2RGB)
        new_img.append(rgb)
    new_img = np.array(new_img).astype(np.uint8)
    return new_img

def extract_data(ground_truth):
    bbox_list = []
    masks = []
    trans = ground_truth.transpose()
    for n in range(trans.shape[0]):
        slice_gt = trans[n,:,:]
        if len(np.unique(slice_gt)) == 1:
            bbox = [0,0,0,0]
        else:
            result = label_props(slice_gt)
            bbox = result['bbox']
        bbox_list.append(bbox)
        masks.append(slice_gt)
    masks = np.array(masks)
    return bbox_list, masks




        
    