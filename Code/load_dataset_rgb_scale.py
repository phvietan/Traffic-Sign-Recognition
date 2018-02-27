import cv2
import numpy as np
import os

def get_dataset_rgb_scale(numClass = 43, n_row = 64, m_row = 64):

    folders = ["%05d" % class_id for class_id in range(numClass)]
    result_x = []
    result_y = []
    cntClass = 0
    numFile = 0

    for folder in folders:
        folder_name = os.path.join("..", "Images", folder)

        images = sorted(os.listdir(folder_name))
        for image_name in images:

            if not image_name.endswith("ppm"):
                continue

            numFile += 1
            result_y.append(cntClass)

            image = cv2.imread(os.path.join("..", "Images", folder, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = cv2.resize(image, (n_row, m_row))
            image = image / 256

            result_x.append(image)
        #
        cntClass += 1
    #
    result_x = np.asarray(result_x).reshape((numFile, n_row, m_row, 3))
    result_y = np.asarray(result_y)

    return result_x, result_y
