import numpy as np
import matplotlib.pyplot as plt
import cv2 

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box:np.array, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle(x0,y0), w, h, edgecolor = 'red', facecolor = (0,0,0,0), lw = 2)

def label_props(labels:np.array):
    """Extract area and bounding box of the objects from the labels

    Args:
        labels (np.array): an array of labeled pixels. 0 indicates the background

    Returns:
        results (dict): area annotates the number of pixels in the objects, 
        bbox defines the bounding box around objects with [x_min, y_min, x_max, y_max]
    """
    mask = labels.astype(np.int8)
    _, _, stats, _  = cv2.connectedComponentsWithStats(mask)
    objects = stats[1:,:]
    coords = objects[:,:-1]
    bbox = np.array((coords[:,0], coords[:,1], coords[:,0] + coords[:,2], coords[:,1] + coords[:,3])).transpose()
    results = {'area':objects[:,-1],
               'bbox':bbox}
    return results
