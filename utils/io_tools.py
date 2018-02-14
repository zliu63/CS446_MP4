"""Input and output helpers to load in data.
(This file will not be graded.)
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.

    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.

    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,8,8,3)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,1)
            containing the loaded label.

            N is the number of examples in the data split, the exampels should
            be stored in the same order as in the txt file.
    """
    N = FileLen(data_txt_file)
    text_file = open(data_txt_file)
    data = {}
    data['image'] = np.zeros((N,8,8,3))
    data['label'] = np.zeros((N,1))
    i = 0
    for line in text_file:
        name,l = line.split(',')
        name = "/"+name+".jpg"
        img = skimage.io.imread(image_data_path + name)
        data['image'][i] = img
        data['label'][i][0] = float(l)
        i+=1
    text_file.close()
    return data

def FileLen(path_to_file):
    f = open(path_to_file)
    ret = sum(1 for g in f)
    f.close()
    return ret
