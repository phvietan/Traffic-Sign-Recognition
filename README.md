# Traffic Sign Recognition on the GTSRB dataset

In this work, I would like to explore different architecture of the neural network in recognizing traffic signs on the GTSRB dataset. Because of the exponential growth of the deep learning in machine learning, I would like to test and experiment on different models to see and to compare their performance.

Official web page of GTSRB dataset: http://benchmark.ini.rub.de/?section=gtsrb&subsection=news

The GTSRB consists of 43 types of traffic sign which is included inside the `Classes.txt` file.
There are 39209 pictures for the train set and xxx pictures for the test set. The total number of pictures is therefore yyy. 

More information about the dataset can be looked up at the website.

I have tried the following appraches:
    
    None hidden layer neural network: 82.438638% accuracy
    1 hidden layer neural network: 84.821853% accuracy
    8 layers of CNN/maxpool with 3 fully connected layers neural network: 87.616785% accuracy
    12 layers of CNN/maxpool with 8 fully connected layer using residual neural network: 88.519398% accuracy

Prerequisites

    Python OpenCV.
    PyTorch.

This was the project of myself after I learned Deep Learning course on Coursera by Andrew Ng.

# How to run the project.

Firstly, just like any github project, you have to clone it down by doing:
    
    git clone https://github.com/phvietan/Traffic-Sign-Recognition.git
    
After that, change directory into Dataset, and follow the instruction of `README.md` inside.

And then, you will find 4 folders which name is: `NoneHiddenlayer`, `OneHiddenlayer`, `CNN`, and `CNNAndResnet`

The 4 folders actually contain very similar code, the only difference is their architecture which is written inside the file `neural_network.py`

The folder `NoneHiddenlayer` contains `README.md` which will tell you what to do next in order to run the project.

Each of 4 folders will contains a picture represent in detail about their architecture.

There is another folder named `checkpoint` which contains my own training network, you can test it if you like.
