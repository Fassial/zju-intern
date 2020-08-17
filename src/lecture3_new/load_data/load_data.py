import numpy as np
from My_Explorer import MyExplorer
from hog.hog import hog

def load_data(ipath,lpath,is_one_frame=False):
    print("start loading data....")
    image_explorer = MyExplorer(ipath,lpath)
    #one frame or two frame
    if is_one_frame==True:
        train_x,train_y,valid_x,valid_y = image_explorer.split_single()
    else:
        train_x, train_y, valid_x, valid_y = image_explorer.split_double()
    return train_x,train_y,valid_x,valid_y

def split_data(train_x, train_y, valid_x, valid_y,total_number=60000):
    train_percent = 0.8
    train_number = (int)(total_number * train_percent)
    test_number = (int)(total_number - train_number)
    train_x_new = train_x[0:train_number]
    train_y_new = train_y[0:train_number]
    valid_x_new = valid_x[0:test_number]
    valid_y_new = valid_y[0:test_number]
    print("loading data end...")
    return train_x_new, train_y_new, valid_x_new, valid_y_new

def get_hog(x):
    X = []
    for tmpx in x:
        normalised_blocks , hog_image = normalised_blocks,hog_image = \
            hog(tmpx, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=True)
        X.append(normalised_blocks)
    X = np.array(X)
    return X

def load_data_and_get_hog(ipath,lpath,total_number = 60000,is_one_frame=False):
    train_x, train_y, valid_x, valid_y = \
                    load_data(ipath,lpath,is_one_frame)

    train_x_new, train_y_new, valid_x_new, valid_y_new = \
                split_data(train_x, train_y, valid_x, valid_y,total_number)

    print("start computing hog...")
    X = get_hog(train_x_new)
    valid_X = get_hog(valid_x_new)
    train_y_new = np.array(train_y_new,dtype = 'int64')
    valid_y_new = np.array(valid_y_new,dtype = 'int64')
    print("computing hog end...")

    return X,train_y_new,valid_X,valid_y_new




