"""
Created on July 25 01:15, 2020

@author: fassial
"""
import os
import random
import numpy as np
from skimage import io
# local dep
import sys
sys.path.append("./hog")
from hog import hog

TRUE       = "true"
FALSE      = "false"
PREFIX     = os.path.join("..", "..", "data", "lecture1")
DATASET    = os.path.join(PREFIX, "dataset")
P_TRAINSET = 0.7

"""
get_files:
    
"""
def get_files(path, filelist):
    # check whether path exists
    if os.path.exists(path):
        for file in os.listdir(path):
            sub_path = os.path.join(path, file)
            if os.path.isdir(sub_path):
                # sub_dir
                get_files(sub_path, filelist)
            else:
                # file
                filelist.append(sub_path)

"""
load_data:
    load dataset from file
    @params:
        None
    @rets:
        x_train(np.array)   : list of x_train
        y_train(np.array)   : list of y_train
        x_test(np.array)    : list of x_test
        y_test(np.array)    : list of y_test
"""
def load_data():
    # init res
    x_all, y_all = [], []
    # get filelist
    filelist_t, filelist_f = [], []
    get_files(os.path.join(DATASET, TRUE), filelist_t)
    get_files(os.path.join(DATASET, FALSE), filelist_f)
    # set x_all & y_all
    for i in range(len(filelist_t)):
        if i % 100 == 0: print("preprocessing..." + str(i) + "/" + str(len(filelist_t)))
        image = io.imread(filelist_t[i])
        _, hog_image = hog(
            image = image,
            orientations = 9,
            pixels_per_cell = (8, 8),
            cells_per_block = (2, 2),
            visualize = True
        )
        x_all.append(hog_image)
        y_all.append(1)
    for i in range(len(filelist_f)):
        if i % 100 == 0: print("preprocessing..." + str(i) + "/" + str(len(filelist_f)))
        image = io.imread(filelist_f[i])
        _, hog_image = hog(
            image = image,
            orientations = 9,
            pixels_per_cell = (8, 8),
            cells_per_block = (2, 2),
            visualize = True
        )
        x_all.append(hog_image)
        y_all.append(0)
    # shuffle
    xy_all = list(zip(x_all, y_all))
    random.shuffle(xy_all)
    x_all[:], y_all[:] = zip(*xy_all)
    # set x_train & x_test
    x_train, y_train = x_all[:int(len(x_all) * P_TRAINSET)], y_all[:int(len(x_all) * P_TRAINSET)]
    x_test, y_test = x_all[int(len(x_all) * P_TRAINSET):], y_all[int(len(x_all) * P_TRAINSET):]
    # return data
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)