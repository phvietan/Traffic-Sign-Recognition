# Initial dataset

Firstly, you need to download and unzip them: 
```bash
$ sh download.sh
```
You can delete the .zip file to clear storage:
```bash
$ sh delete.sh
```
Secondly, because we will be using data-loader of the Pytorch, we need to initialize the .csv file that contains the directory with the classId of every images:
```bash
$ sh initial.sh
```
After waiting for a short time, the dataset is now ready to be used.
