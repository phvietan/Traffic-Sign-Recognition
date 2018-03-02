import torch
import time
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from neural_network import *

def evaluate_net(net, dataset):
    # initialize some parameters
    loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    # Switch to evaluation mode.
    net.eval()
    # We need the batch_size to be equal 1 for testing
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i_batch, sample_batched in enumerate(dataloader):
        input = sample_batched['image']
        target = sample_batched['classId']
        # convert the image and classId into Variable
        input = Variable(input.float())
        target = Variable(target)
        # calculate and update the loss
        outputs = net(input)
        loss += criterion(outputs, target).data[0]
        # get the maximum predicted
        score, predicted = torch.max(outputs, 1)
        # check if label and predict is the same
        now = (target.data == predicted.data).sum()
        # update the params
        correct += now
        total += target.size(0)
    acc = correct / total
    loss /= total

    return loss, acc

def plotHistory(hist):
    # draw diagram of the history of loss
    print("History of loss after each epochs:", hist)
    plt.plot(hist)
    plt.ylabel('Loss of training set')
    plt.xlabel('Number of epoches')
    plt.show()

def evaluate(checkpoint_path, model_name, dataset, showHistory = False):
    # inital the network
    net = Net()

    # retrieve params from checkpoint
    name_checkpoint = model_name + ".chkpt"
    checkpoint = torch.load(os.path.join(checkpoint_path, name_checkpoint))
    hist = checkpoint["hist"]
    net.load_state_dict(checkpoint["net"])

    if (showHistory):
        plotHistory(hist)

    # evaluate and check for loss, accuracy and run-time per image
    start = time.time()
    loss, acc = evaluate_net(net, dataset)
    now = time.time()

    print("Loss = %.6f" % (loss))
    print("Acc = %.6f percent" % (acc*100))
    print("Time per image: %.6f seconds: " % ((now-start)/len(dataset)))
