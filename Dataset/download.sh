#!/bin/bash

#Download the dataset
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip
wget http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip

#Unzip them
unzip GTSRB_Final_Training_Images -d trainSet
unzip GTSRB_Final_Test_Images -d testSet
unzip GTSRB_Final_Test_GT.zip
