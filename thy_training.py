from thy_train.utils import extract_data, LoadData
from thy_train.model import UNet3D
from thy_train.trainer import TrainEpoch
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
    batch_size = int(sys.argv[3])
    lr = float(sys.argv[4])
    init_features = int(sys.argv[5])
    #window = int(sys.argv[6])
    report_every = int(sys.argv[6])
    os.mkdir(output_dir)
    logging.basicConfig(
        filename= os.path.join(output_dir,'train.log'),
        filemode="a",                  
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'The model is training with batch size of {batch_size}, report every {report_every} epochs')
    logging.info(f'The model has init_features of {init_features}')
    logging.info(f'The model is training with learning rate of {lr}')
    
    #path = "./data/kopf_SHIP/training_data"
    imgs, labels = extract_data(input_path)
    #print(imgs.shape)

    # Define KFold split data
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Loop through the folds
    precision = []
    recall = []
    dice = []
    iou = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(imgs, labels)):
        #print(len(train_idx), len(val_idx))
        logging.info(f'Running fold {fold+1}')
        output_folder = f'{output_dir}/fold{fold+1}'
        # Create new folder for each fold
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        # Split data to train and validation data
        train_imgs, train_labels = imgs[train_idx], labels[train_idx]
        val_imgs, val_labels = imgs[val_idx], labels[val_idx]
        # Input data to DataLoader for augmentation
        train_loader = LoadData(train_imgs, train_labels, mode="train")
        val_loader = LoadData(val_imgs, val_labels, mode="val")
        # Initiate model
        model = UNet3D(init_features = init_features)
        # Use DataLoader for batching data
        train_dataset = DataLoader(train_loader, batch_size = batch_size, shuffle = True)
        val_dataset = DataLoader(val_loader, batch_size = batch_size, shuffle = False)
        tr = TrainEpoch(
            output_folder, 
            train_dataset, 
            val_dataset, 
            lr = lr, 
            model = model, 
            device = device, 
            epoch_num = 150, 
            report_every = report_every,
        )
        tr.train_epoch()
        precision.append(tr.epoch_val_precision_score[-1])
        recall.append(tr.epoch_val_recall_score[-1])
        dice.append(tr.epoch_val_dice_score[-1])
        iou.append(tr.epoch_val_iou_score[-1])
        logging.info(f'Finishing fold {fold+1}')
    # Results of cross validation
    logging.info(f'Final Precision: {np.array(precision).mean()}')
    logging.info(f'Final Recall: {np.array(recall).mean()}')
    logging.info(f'Final Dice score: {np.array(dice).mean()}')
    logging.info(f'Final IoU score: {np.array(iou).mean()}')
