import torch
import os
import PIL.Image
import torch.utils.data
import cv2
import numpy as np
import json
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, imageList, transform=None, random_hflip=False):
        super(XYDataset, self).__init__()
        self.image_path = open(imageList,"r").readlines()
        self.transform = transform
        self.random_hflip = random_hflip
        
    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):
        image_path = self.image_path[idx]
        annotation_path =  os.path.join(os.path.dirname(image_path),os.path.basename(image_path).replace(".png",'.json'))
        ann = json.load(open(annotation_path, 'r'))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = PIL.Image.fromarray(image)
        width = image.width
        height = image.height
        if self.transform is not None:
            image = self.transform(image)
        #TODO: check the bug, X and Y can't read
        
        x = 2.0 * (ann['x'] / width - 0.5) # -1 left, +1 right
        y = 2.0 * (ann['y'] / height - 0.5) # -1 top, +1 bottom
        
        if self.random_hflip and float(np.random.random(1)) > 0.5:
            image = torch.from_numpy(image.numpy()[..., ::-1].copy())
            x = -x
            
        return image, torch.Tensor([x, y])

class HeatmapGenerator():
    def __init__(self, shape, std):
        self.shape = shape
        self.std = std
        self.idx0 = torch.linspace(-1.0, 1.0, self.shape[0]).reshape(self.shape[0], 1)
        self.idx1 = torch.linspace(-1.0, 1.0, self.shape[1]).reshape(1, self.shape[1])
        self.std = std
        
    def generate_heatmap(self, xy):
        x = xy[0]
        y = xy[1]
        heatmap = torch.zeros(self.shape)
        heatmap -= (self.idx0 - y)**2 / (self.std**2)
        heatmap -= (self.idx1 - x)**2 / (self.std**2)
        heatmap = torch.exp(heatmap)
        return heatmap