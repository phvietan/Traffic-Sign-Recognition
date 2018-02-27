import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from get_num_train import *

class TrafficSignDataset(Dataset):
    def __getitem__(self, idx):
        folders = ["%05d" % class_id for class_id in range(43)]
        for folder in folders:
            csv_name = os.path.join(self.root_dir, folder, "GT-"+folder+".csv")
            landmarks_frame = pd.read_csv(csv_name)
            if (idx < len(landmarks_frame)):
                img_name = str(landmarks_frame.loc[idx,"Filename"])
                Y_result = landmarks_frame.loc[idx,"ClassId"]
                image = cv2.imread(os.path.join(self.root_dir, folder, img_name))
                image = cv2.resize(image, (self.n_row, self.m_row))
                sample = {'image': image, 'landmarks': Y_result}
                return sample
            else:
                idx -= len(landmarks_frame)
                
    def __init__(self):
        self.root_dir = os.path.join("..", "Images")
        self.num_class = 43
        self.n_row = self.m_row = 64
    
    def __len__(self):
        return get_num_train()
