import time
import torch
from torch.autograd import Variable

from train_dataset import *
from neural_network import *

# get the train set for later train
train_set = TrafficSignTrain()

def train_data(checkpoint_path, model_name, learning_rate = (1e-4), num_epochs = 5, 
               batch_size = 1, resume = False, num_class = 43):
    # initial the network
    net = Net()
    
    # i like to calculate time after multiple iterations
    start = time.time()
    # use the soffmax loss function
    criterion = nn.CrossEntropyLoss()
    # use adam optimization 
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    
    m = len(train_set)
    name_checkpoint = model_name + ".chkpt"
    hist = []
    if (resume): 
        # load the checkpoint
        checkpoint = torch.load(os.path.join(checkpoint_path, name_checkpoint))
        # retrieve information of that checkpoint
        hist = checkpoint["hist"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        net.load_state_dict(checkpoint["net"])
    
    # iterate through each epoch
    for iters in range(num_epochs):
        # we are using Stochastic Gradient Descent, so we get a random permutation
        indices = np.random.permutation(m)
        for i in range(m):
            # get the image and classId
            input = train_set[indices[i]]['image']
            target = train_set[indices[i]]['classId']
            # convert the image and classId into Tensor of Pytorch
            input = Variable(torch.from_numpy(input).float())
            target = Variable(torch.from_numpy(target).long())
            # calculate the loss
            out = net(input)
            loss = criterion(out, target)
            # back-propagate using the Adam optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # output the status every 4900 iterations
            if (i%4900==0):
                now = time.time()
                print("iterate %d: loss = %.9f, spent = %.5f" % (i + 1, loss.data[0], now - start))
                start = now
        # output the status after each epoch
        print("Epoch %d: loss = %.9f" % (iters + 1, loss.data[0]))
        # append history of the loss for later usage (we will draw some diagram)
        hist.append(loss.data[0])
    
    # save them to the checkpoint
    checkpoint = {
        "optimizer": optimizer.state_dict(),
        "hist": hist,
        "net": net.state_dict()
    }
    torch.save(checkpoint, os.path.join(checkpoint_path, name_checkpoint))
    
    return hist