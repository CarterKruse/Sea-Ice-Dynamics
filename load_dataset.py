r'''
PyTorch Dataset
==================
Creating a PyTorch dataset object from the './_data' folder.

Link: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
'''

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class DensityFieldDataset(Dataset):
    def __init__(self, data_directory, time_interval = 1):
        self.time_interval = time_interval
        self.files = [os.path.join(data_directory, file) for file in os.listdir(data_directory) if file.startswith('density') and file.endswith('.npy')]

        self.data = [] # shape: (100 * x - y, 3, 100, 100)

        # load data, create sequences
        for file in self.files:
            fields = np.load(file)
            self.data.extend([
                fields[i: i + 3 * self.time_interval: self.time_interval]
                for i in range(len(fields) - 2 * self.time_interval)
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]

        # density fields
        x_train = torch.tensor(sequence[:2], dtype = torch.float32)
        y_train = torch.tensor(sequence[2], dtype = torch.float32)

        return x_train, y_train  

data_directory = './_data'
time_interval = 3

dataset = DensityFieldDataset(data_directory, time_interval)
print(dataset.__getitem__(0))
