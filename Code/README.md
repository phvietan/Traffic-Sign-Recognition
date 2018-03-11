# Checkpoint folder

The `checkpoint` folder contains the record of my pre-trained network. You can use it to test with your own image. Follow further instruction in `NoneHiddenLayer` folder

# CNN, CNNAndResnet, NoneHiddenLayer, OneHiddenLayer folders

4 folders contain very similar code. Their only difference is the `neural_network.py`.

### Content of each file
* 1) Neural_network.py: a class that represent the neural network architecture.
* 2) Train_dataset.py and Test_dataset.py: 2 classes for the Dataloader of Pytorch.
* 3) Train_method.py: contains every method we need for training.
* 4) Evaluate.py: contains every method we need for testing.
* 5) Train.py: run this file to start training.
* 6) Test.py: run this file to start testing.

### Start training and testing:

Train:
```bash
$ cd <your_desired_folder>
$ python train.py
```
I do not use learning rate decay, so you need to change the learning rate when the loss starts to diverge:
```bash
$ nano train.py
```
Test:
```bash
$ python test.py
```
# Architecture of each model

### Neural network without any hidden layer

<p align="center">
<img src="https://github.com/phvietan/Traffic-Sign-Recognition/blob/master/img/NormalNN.png" height="400" width="500">
</p>

### Neural network with one hidden layer

<p align="center">
<img src="https://github.com/phvietan/Traffic-Sign-Recognition/blob/master/img/NNWithOneHiddenLayer.png" height="400" width="500">
</p>

### Convolution neural network

<p align="center">
<img src="https://github.com/phvietan/Traffic-Sign-Recognition/blob/master/img/CNN.png" height="650" width="850">
</p>

### Convolution neural network based on ResNet architecture

<p align="center">
<img src="https://github.com/phvietan/Traffic-Sign-Recognition/blob/master/img/CNNWithResNet1.png" height="400" width="700">
<img src="https://github.com/phvietan/Traffic-Sign-Recognition/blob/master/img/CNNWithResNet2.png" height="500" width="700">
</p>
