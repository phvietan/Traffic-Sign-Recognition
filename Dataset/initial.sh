#!/bin/bash

#Inside the "GTSRB_Final_Test_Images.zip" we don't use the csv file
#Instead we use the csv file that is given inside "GTSRB_Final_Test_GT.zip"

#Firstly we remove the csv file that we don't use
rm testSet/GTSRB/Final_Test/Images/GT-final_test.test.csv
echo "Done removed unused file"

#Secondly we move the csv file that we gonna use into the correct directory
mv GT-final_test.csv testSet/GTSRB/Final_Test/Images/GT-final_test.csv
echo "Done moving file to desired directory"
