from train.utils import extract_data, LoadData
from train.model import UNet3D
from train.trainer import TrainEpoch
import os
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import logging
import numpy as np
import sys
import torch

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    batch_size = sys.argv[3]
    lr = sys.argv[4]
    logging.basicConfig(
        filename= os.path.join(output_dir,'train.log'),
        filemode="a",                  
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'The model is running with batch size of {batch_size}')
    logging.info(f'The model is running with learning rate of {lr}')

    #path = "./data/kopf_SHIP/training_data"
    imgs, labels = extract_data(input_path)

    # Define KFold split data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Test codes
    #imgs = imgs[:5]
    #labels = labels[:5]
    # Loop through the folds
    precision = []
    recall = []
    dice = []
    iou = []
    n = 1
    for fold, (train_idx, val_idx) in enumerate(kf.split(imgs)):
        n += 1
        logging.info(f'Running fold {n}')
        output_folder = f'{output_dir}/fold0{n}'
        os.mkdir(output_folder)
        train_imgs, train_labels = imgs[train_idx], labels[train_idx]
        val_imgs, val_labels = imgs[val_idx], labels[val_idx]
        train_loader = LoadData(train_imgs, train_labels, mode="train")
        val_loader = LoadData(val_imgs, val_labels, mode="val")
        train_dataset = DataLoader(train_loader, batch_size = batch_size, shuffle = True)
        val_dataset = DataLoader(val_loader, batch_size = batch_size, shuffle = False)
        path = ''
        tr = TrainEpoch(output_folder, train_dataset, val_dataset, lr = lr, UNet3D, device = device, epoch_num = 100)
        tr.train_epoch()
        precision.append(tr.epoch_val_dice_score[-1])
        recall.append(tr.epoch_val_recall_score[-1])
        dice.append(tr.epoch_val_dice_score[-1])
        iou.append(tr.epoch_val_iou_score[-1])
        logging.info(f'Finishing fold {n}')

    # Results of cross validation
    logging.info(f'Precision: {np.array(precision).mean()}')
    logging.info(f'Recall: {np.array(recall).mean()}')
    logging.info(f'Dice score: {np.array(dice).mean()}')
    logging.info(f'IoU score: {np.array(iou).mean()}')