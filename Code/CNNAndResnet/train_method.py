import time
import torch
from torch.autograd import Variable

from train_dataset import *
from neural_network import *

# get the train set for later train
train_set = TrafficSignTrain()

def train_data(checkpoint_path, model_name, learning_rate = (1e-4), num_epochs = 5,
               batch_size = 64, resume = False, num_class = 43, num_output = 6):
    # initial the network
    net = Net()

    # output after iterate N times
    N = int(len(train_set) / batch_size / num_output)

    # i like to calculate time after multiple iterations
    start = time.time()
    # use the soffmax loss function
    criterion = nn.CrossEntropyLoss()
    # use adam optimization
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # change the learning rate as your will
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    name_checkpoint = model_name + ".chkpt"
    hist = []
    if (resume):
        # load the checkpoint
        checkpoint = torch.load(os.path.join(checkpoint_path, name_checkpoint))
        # retrieve information of that checkpoint
        hist = checkpoint["hist"]
        optimizer.load_state_dict(checkpoint["optimizer"])
        net.load_state_dict(checkpoint["net"])

    # initialize the train loader
    dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # iterate through each epoch
    for iters in range(num_epochs):
        i = 0
        for i_batch, sample_batched in enumerate(dataloader):
            input = sample_batched['image']
            target = sample_batched['classId']
            # convert the image and classId into Variable
            input = Variable(input.float())
            target = Variable(target)
            # calculate the loss
            out = net(input)
            loss = criterion(out, target)
            # back-propagate using the Adam optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # output the status every N iterations
            if (i%N==0):
                now = time.time()
                print("iterate %d: loss = %.9f, spent = %.5f" % (i + 1, loss.data[0], now - start))
                start = now
            i = i+1
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
