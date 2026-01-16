from pitseg.model import UNet3D
import torch
import numpy as np
import logging
import os
import pickle
from tqdm import tqdm
import pandas as pd

class MCDropout:
    def __init__(self, model, dataloader, outpath, T = 50):
        self.model = model
        # Set model to evaluation mode
        self.model.eval()
        for m in self.model.modules():
            # Set dropout layers to active
            if isinstance(m, torch.nn.Dropout3d):
                m.train()

        self.T = T
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.outpath = outpath

    def save_stats(self,names, median_stdevs, mean_stdevs):
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        df = pd.DataFrame({
            'id':names,
            'median_std':median_stdevs,
            'mean_std':mean_stdevs,
        })
        df.to_csv(os.path.join(self.outpath,'stats.csv'))

    def save_mean_preds(self, names, mean_preds):
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        sub_dir = os.path.join(self.outpath,'mean_preds')
        if not os.path.isdir(sub_dir):
            os.mkdir(sub_dir)
        for name, mean_pred in zip(names, mean_preds):
            np.save(os.path.join(sub_dir,str(name.item())), mean_pred)
        

    @torch.no_grad()
    def run(self):
        identity = []
        median_stdevs = []
        mean_stdevs = []
        for _, batch in tqdm(enumerate(self.dataloader)):
            preds = []
            names, patches = batch
            patches = patches.unsqueeze(1).to(torch.float32)
            patches = patches.to(self.device)
            # Loop over T times
            for t in range(self.T):
                logits = self.model(patches)
                logits = logits.squeeze(1)
                probs = torch.sigmoid(logits)
                preds.append(probs)

                probs.cpu()
                patches.cpu()
                logits.cpu()
                
            preds = torch.stack(preds)
            # Calculate mean and std per batch over T times
            mean_preds = torch.mean(preds, axis = 0).cpu().numpy()
            std =  torch.std(preds, axis = 0).flatten(1)
            # Save the mean_preds to files
            self.save_mean_preds(names, mean_preds)
            median_std = torch.median(std, dim =-1).values.cpu().numpy()
            mean_std = torch.mean(std, dim =-1).cpu().numpy()
            std.cpu()
            median_stdevs.append(median_std)
            mean_stdevs.append(mean_std)
            identity.append(names)
        identity = np.concatenate(identity, axis = 0)
        median_stdevs = np.concatenate(median_stdevs, axis = 0)
        mean_stdevs = np.concatenate(mean_stdevs, axis = 0)
        self.save_stats(identity, median_stdevs, mean_stdevs)

        

                
                
        
        
                