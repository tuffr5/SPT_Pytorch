import torch
import numpy as np

class ParticleDataset(torch.utils.data.Dataset):
    def __init__(self, folder, img_file, label_file):
        self.data = np.float32(np.load(folder + img_file))
        self.label = np.float32(np.load(folder + label_file))
        
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        
        img = torch.from_numpy(self.data[index, ...])
        label = torch.from_numpy(self.label[index, ...])
        
        
        return img, label