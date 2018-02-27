import cv2
import numpy as np
import os
import csv
import pandas as pd

def load_test_y():
    csv_name = os.path.join('GTSRB', 'GT-final_test.csv')
    
    landmarks_frame = pd.read_csv(csv_name)
    return np.array(landmarks_frame.loc[:,"ClassId"])

def load_test_set(n_row = 64, m_row = 64):
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
        
        image = cv2.resize(image, (n_row, m_row))
        image = image / 256

        result_x.append(image)
    #
    result_y = load_test_y()
    result_x = np.asarray(result_x).reshape((numFile, n_row, m_row, 3))
    
    return result_x, result_y

