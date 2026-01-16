import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pitseg.utils import DataCustomer
from pitseg.pitseg import load_model
import os
from pitseg.monte_carlo_dropout import MCDropout
import sys

if __name__ == '__main__':
    # Define paths
    path = './data/pituitary/kopf_SHIP/SHIP_2025_50_D_S2'
    df_path = './data/pituitary/kopf_SHIP/pitvol_S2_new_model_7.csv'
    outpath = './data/pituitary/Monte_Carlo_Dropouts/SHIP_2025_50_D_S2'

    # Read .csv file
    df = pd.read_csv(df_path)
    
    # Load image data from .nii.gz files
    data_customer = DataCustomer(path,df.zz_nr,df.x_centers,df.y_centers, df.z_centers)
    batches = DataLoader(data_customer, batch_size = 50)
    
    # Load model
    ckpt = './pitseg/checkpoint/pitseg_0.3.7.pt'
    model = load_model(ckpt)
    
    # Run Monte Carlo Droputs
    mcdropout = MCDropout(model, batches, outpath, T = 50)
    mcdropout.run()