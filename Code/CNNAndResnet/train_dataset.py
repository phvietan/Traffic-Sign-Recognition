# import os to use os.path.join
# import panda to read csv file
# import cv2 to resize images to shape (64, 64)
# import torch.utils.data to use dataloader of Pytorch
import os
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

# create dataloader class for train set
class TrafficSignTrain(Dataset):               
    def __init__(self):
        # store csv directory
        csv_dir = os.path.join("..", "..", "Dataset", "train.csv")
        # read csv and store to train
        self.csv = pd.read_csv(csv_dir)
    
    def __getitem__(self, idx):
        # get the directory of the image, but it is not really correct, so we fix it by joining 2 more '..'
        img_dir = str(self.csv.loc[idx,"Directory"])
        img_dir = os.path.join("..", "..", "Dataset", img_dir)
        # get the classId
        classId = self.csv.loc[idx,"Class"]
        # get the image
        image = cv2.imread(img_dir)
        # because the color system of cv2 is BGR, but i want it to be RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize
        image = cv2.resize(image, (64, 64))
        # normalize the image
        image = image / 256
        # transpose from (64x64x3) to (3x64x64) because of Pytorch convention
        image = np.transpose(image, (2, 0, 1))
        # return result
        sample = {'image': image, 'classId': classId }
        return sample
  
    def __len__(self):
        return len(self.csv)