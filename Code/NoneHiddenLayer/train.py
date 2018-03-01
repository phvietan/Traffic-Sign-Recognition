# import class for train_dataset 
from train_dataset import *
# import the neural network
from neural_network import *
# method for training
from train_method import *

# retrieve the train_dataset
train_set = TrafficSignTrain()
print("Size of train set is:", len(train_set), "images")

hist = train_data("../checkpoint", "NoneHiddenlayer", resume = False, num_epochs = 3)
# after it has done training, it will store the net into the checkpoint named "NoneHiddenlayer"
