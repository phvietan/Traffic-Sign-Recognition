import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

def get_outputs(net, instances):
    instances = Variable(instances)
    outputs = net(instances)
    return outputs

def evaluate(net, data_X, data_Y):
    loss = 0
    correct = 0
    total = 0
    
    m = data_X.shape[0]

    # Switch to evaluation mode.
    net.eval()

    for i in range(m):
        outputs = get_outputs(net, torch.from_numpy(data_X[i:i+1]).float())
        labels = Variable(torch.from_numpy(data_Y[i:i+1]).long())

        loss += nn.CrossEntropyLoss(size_average=False)(outputs, labels).data[0]

        score, predicted = torch.max(outputs, 1)
        correct += (labels.data == predicted.data).sum()
        
        total += labels.size(0)

    acc = correct / total
    loss /= total

    return loss, acc
