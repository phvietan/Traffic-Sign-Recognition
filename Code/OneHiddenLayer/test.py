# import class for test_dataset and train_dataset
from test_dataset import *
from train_dataset import *
# import the neural network
from neural_network import *
# method for testing
from evaluate import *

# retrieve test set and train set
test_set = TrafficSignTest()
train_set = TrafficSignTrain()

print("Size of test set is:", len(test_set), "images")
print("Size of train set is:", len(train_set), "images")

print("Test result:")
evaluate("../checkpoint", "OneHiddenlayer", test_set, showHistory = True)
print("-------------------------------")
print("Train result:")
evaluate("../checkpoint", "OneHiddenlayer", train_set)
