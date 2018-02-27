import cv2
import numpy as np
import os
import csv

def load_test_y():
    result = []
    with open('GTSRB/GT-final_test.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        cnt = 0
        for row in reader:
            value = row['Filename;Width;Height;Roi.X1;Roi.Y1;Roi.X2;Roi.Y2;ClassId'].split(';')
            result.append(int(value[7]))
    result = np.asarray(result)
    return result

def load_test_set_with_one(n_row = 64, m_row = 64):
    path = os.path.join(".","GTSRB","Final_Test","Images")
    images = sorted(os.listdir(path))
    result_x = []
    numFile = 0
    
    for image_name in images:
        if not image_name.endswith("ppm"):
            continue

        numFile += 1
        image = cv2.imread(os.path.join(".","GTSRB","Final_Test","Images", image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ##
        image = cv2.resize(image, (n_row, m_row))
        #phuc tap vay
        image = image.reshape((n_row*m_row*3, 1));
        image = image / 256

        result_x.append(image)
    #
    result_y = load_test_y()
    
    result_x = np.asarray(result_x).reshape((numFile, n_row * m_row * 3))
    
    MOT = np.ones((numFile, 1))
    result_x = np.hstack((result_x, MOT))
    
    return result_x, result_y