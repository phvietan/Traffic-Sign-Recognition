import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    # max_pool here I only use 2x2 with strade 2

    def __init__(self):
        super(Net, self).__init__()
        #using "same" with relu then max pool from 64x64x3 to 32x32x8
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        #using "same" with relu then max pool from 32x32x8 to 16x16x16
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        #using "same" with relu then max pool from 16x16x16 to 8x8x32
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        #using "same" with relu then max pool from 8x8x32 to 4x4x64
        self.conv4 = nn.Conv2d(32, 64, 5, padding=2)

        #Fully connected layer
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 43)

    def forward(self, X):
        # use max pool with 2x2 and strade 2
        X = F.max_pool2d(F.relu(self.conv1(X)), (2,2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2,2))
        X = F.max_pool2d(F.relu(self.conv3(X)), (2,2))
        X = F.max_pool2d(F.relu(self.conv4(X)), (2,2))

        # flatten the image
        X = X.view(-1, self.num_flat_features(X))

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return X

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
