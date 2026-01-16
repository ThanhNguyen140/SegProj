from pitseg.model import UNet3D
import torch
import numpy as np
from monai.transforms import RemoveSmallObjects
from monai.transforms import FillHoles

#ckpt_path = '../thyseg/checkpoint/thyseg112_0.1.0.pt'
def load_model(ckpt_path):
    model = UNet3D(init_features = 64)
    checkpoint = ckpt_path
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    return model

class Segmentor:
    def __init__(self, model, x_center = 128, y_center = 112, z_center = 46):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        #print(f'The model is running on {self.device}')
        self.model.eval()
        self.xw = 32
        self.yw = 16
        self.zw = 8
        self.x_center = x_center
        self.y_center = y_center
        self.z_center = z_center
        self.default = (x_center, y_center, z_center)

    def preprocess(self, imgs):
        imgs_tensor = torch.tensor(imgs)
        if len(imgs.shape) < 4:
            imgs_tensor = imgs_tensor.unsqueeze(0)
            #print('Processing a single image')
        #else:
            #print('Processing images in a batch')
        self.shape = imgs_tensor.shape
        flattened = imgs_tensor.flatten(1)
        normalized = (flattened - flattened.mean(axis = -1)) / flattened.std(axis = -1)
        imgs_tensor = normalized.view(self.shape)
        imgs_tensor = imgs_tensor.unsqueeze(1)
        return imgs_tensor.to(torch.float32)

    def patching(self,imgs_tensor):
        self.xmin, self.xmax = int(self.x_center - self.xw), int(self.x_center + self.xw)
        self.ymin, self.ymax = int(self.y_center - self.yw), int(self.y_center + self.yw)
        self.zmin, self.zmax = int(self.z_center - self.zw), int(self.z_center + self.zw)
        cubes = imgs_tensor[:,:,self.xmin : self.xmax, self.ymin : self.ymax, self.zmin : self.zmax]
        return cubes

    def calculate_centroid(self, mask: torch.Tensor):
        """
        Compute the centroid (x, y, z) of 3D masks for each batch.
        
        Args:
            mask (torch.Tensor): shape (B, X, Y, Z)
        
        Returns:
            torch.Tensor: shape (B, 3) with centroids in (x, y, z) order
        """
        B, x, y, z = mask.shape
    
        # create coordinate grids
        x_coords = torch.arange(x, device=mask.device, dtype=torch.float32)
        y_coords = torch.arange(y, device=mask.device, dtype=torch.float32)
        z_coords = torch.arange(z, device=mask.device, dtype=torch.float32)
    
        # broadcast to shape (x, y, z)
        grid_x, grid_y, grid_z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    
        # flatten and expand for batch
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1, -1)
        grid_z = grid_z.unsqueeze(0).expand(B, -1, -1, -1)
    
        # ensure mask is float
        mask = mask.float()
    
        # sum over voxels
        total_mass = mask.sum(dim=(1, 2, 3), keepdim=True) + 1e-8  # avoid divide-by-zero
    
        cx = (mask * grid_x).sum(dim=(1, 2, 3), keepdim=True) / total_mass
        cy = (mask * grid_y).sum(dim=(1, 2, 3), keepdim=True) / total_mass
        cz = (mask * grid_z).sum(dim=(1, 2, 3), keepdim=True) / total_mass
    
        return torch.cat([cx, cy, cz], dim=-1).squeeze(1,2)

    def __reset(self):
        self.x_center, self.y_center, self.z_center = self.default

    @torch.no_grad()
    def predict(self, imgs, threshold = 0.5):
        # Normalize images before inputting to the models
        imgs_tensor = self.preprocess(imgs)
        # Universal patching
        cubes = self.patching(imgs_tensor)
        cubes = cubes.to(self.device)
        # First prediction based on universal patching
        logits = self.model(cubes)
        masks, masks_prob = self.postprocess(logits, threshold)
        new_centroids = self.calculate_centroid(masks)
        #print(new_centroids, new_centroids.shape)
        new_masks = []
        new_masks_prob = [] 
        # Switch self.shape[0], switching to producing mask individually for each images with different patching
        self.shape = imgs_tensor[0].shape
        #print(self.shape)
        for idx in range(new_centroids.shape[0]):
            # Change centroids to fit to the ROI of each individual image
            self.x_center, self.y_center, self.z_center = new_centroids[idx]
            if (self.x_center - self.xw < 0) or (self.y_center - self.yw < 0) or (self.z_center - self.zw < 0):
                self.__reset()
            elif (self.x_center + self.xw < self.shape[0]-1) or (self.y_center + self.yw < self.shape[1]-1) or (self.z_center + self.zw < self.shape[2]-1):
                self.__reset()
            new_cubes = self.patching(imgs_tensor[idx].unsqueeze(0))
            #print(new_cubes.shape)
            new_cubes = new_cubes.to(self.device)
            new_logits = self.model(new_cubes)
            m, m_prob = self.postprocess(new_logits, threshold)
            m = m.cpu().numpy()
            m_prob = m_prob.cpu().numpy()
            new_logits.cpu().numpy()
            new_masks.append(m)
            new_masks_prob.append(m_prob)
        new_masks = np.concatenate(new_masks, axis = 0)
        new_masks_prob = np.concatenate(new_masks_prob, axis = 0)
        #print(new_masks.shape)
        if new_masks.shape[0] == 1:
            new_masks = new_masks.squeeze(0)
            new_masks_prob = new_masks_prob.squeeze(0)
        self.__reset()
        return new_masks, new_masks_prob

    def postprocess(self, logits, threshold):
        logits = logits.squeeze(1)
        probs = torch.sigmoid(logits)
        pred = torch.where(probs > threshold, 1, 0).int()
        pred = RemoveSmallObjects(min_size=50)(pred)
        pred = FillHoles(applied_labels=[1])(pred)

        pred = pred.cpu()
        logits = logits.cpu()
        probs = probs.cpu()
        
        masks = torch.zeros(self.shape, dtype = torch.uint8, device = torch.device('cpu'))
        masks_prob = torch.zeros(self.shape, dtype = torch.float32, device = torch.device('cpu'))

        masks[:,self.xmin : self.xmax, self.ymin : self.ymax, self.zmin : self.zmax] = pred
        masks_prob[:,self.xmin : self.xmax, self.ymin : self.ymax, self.zmin : self.zmax] = probs
        return masks, masks_prob
        
            
        