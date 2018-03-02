import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
#         a picture with resolution 64x64x3 fully connected to a hidden layer with 64 neurons
        self.fc1 = nn.Linear(64*64*3, 64)
        # Hidden layer with 64 neurons fully connected with 43 classes
        self.fc2 = nn.Linear(64, 43)

    def forward(self, X):
#         flatten the input X
        X = X.view(-1, self.num_flat_features(X))
#         forward
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
