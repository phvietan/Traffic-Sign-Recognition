import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        #using "same" 2 times with relu then max pool from 64x64x3 to 32x32x9
        self.conv1 = nn.Conv2d(3, 6, 9, padding=4)
        self.conv2 = nn.Conv2d(6, 9, 9, padding=4)
        #using "same" 2 times with relu then max pool from 32x32x9 to 16x16x15
        self.conv3 = nn.Conv2d(9, 12, 7, padding=3)
        self.conv4 = nn.Conv2d(12, 15, 7, padding=3)
        #using "same" 2 times with relu then max pool from 16x16x15 to 8x8x21
        self.conv5 = nn.Conv2d(15, 18, 5, padding=2)
        self.conv6 = nn.Conv2d(18, 21, 5, padding=2)
        #using "same" with maxpool with relu from 8x8x21 to 8x8x27 then to 4x4x27
        self.conv7 = nn.Conv2d(21, 24, 3, padding=1)
        self.conv8 = nn.Conv2d(24, 27, 3, padding=1)

        #Fully connected layer
        self.fc1 = nn.Linear(432, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 64)
        self.fc7 = nn.Linear(64, 64)
        self.fc8 = nn.Linear(64, 43)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2,2))
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(F.relu(self.conv4(X)), (2,2))
        X = F.relu(self.conv5(X))
        X = F.max_pool2d(F.relu(self.conv6(X)), (2,2))
        X = F.relu(self.conv7(X))
        X = F.max_pool2d(F.relu(self.conv8(X)), (2,2))

        X = X.view(-1, self.num_flat_features(X))

        # dropout = nn.Dropout(p=0.2)
        #
        # X1 = dropout(F.relu(self.fc1(X))) #432 to 200
        # X2 = dropout(F.relu(self.fc2(X1))) #200 to 200
        # X3 = dropout(F.relu(self.fc3(X1 + X2))) #200 to 128
        # X4 = dropout(F.relu(self.fc4(X3))) #128 to 128
        # X5 = dropout(F.relu(self.fc5(X3 + X4))) #128 to 64
        # X6 = dropout(F.relu(self.fc6(X5))) #64 to 64
        # X7 = dropout(F.relu(self.fc7(X5 + X6))) #64 to 64
        # X8 = self.fc8(X7)

        X1 = F.relu(self.fc1(X)) #432 to 200
        X2 = F.relu(self.fc2(X1)) #200 to 200
        X3 = F.relu(self.fc3(X1 + X2)) #200 to 128
        X4 = F.relu(self.fc4(X3)) #128 to 128
        X5 = F.relu(self.fc5(X3 + X4)) #128 to 64
        X6 = F.relu(self.fc6(X5)) #64 to 64
        X7 = F.relu(self.fc7(X5 + X6)) #64 to 64
        X8 = self.fc8(X7)

        return X8

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
