#here
from python.svmutil import *
from My_Explorer import *
from hog.hog import hog
import numpy as np
# here
from ROC_plot.draw_plot import draw_roc_plot
from sklearn.metrics import roc_curve
# here
dataset = MyExplorer(ipath,lpath)
train_x,train_y,valid_x,valid_y = dataset.split_single()


def get_hog(x):
    X = []
    for tmpx in x:
        normalised_blocks , hog_image = normalised_blocks,hog_image = hog(tmpx, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=True)
        X.append(normalised_blocks)
    X = np.array(X)
    return X

train_X = get_hog(train_x)
valid_X = get_hog(valid_x)


def test_kernels():
    name_list = ['linear_c_0.5','linear_c_1','rbf_c_4_gamma_0.1','poly_c_4_gamma_0.1']
    param_list = ['-t 0 -c 0.5','-t 0 -c 1','-t 2 -c 4 -g 0.1','-t 1 -c 4 -g 0.1']
    n = 4 
    prob  = svm_problem(train_y, train_X)
    for i in range(n):
        m = svm_train(prob, param_list[i])
        p_labels, p_acc, p_vals = svm_predict(valid_y, valid_X, m)
        y_true = []
        for j in p_val:
            y_true.append(j[1])
        np.save(name_list[i]+'.npy',np.array(y_true))

def find_rbf_param():
    name_list = ['rbf_c_0.25_gamma_0.001','rbf_c_0.25_gamma_0.01','rbf_c_0.25_gamma_0.1','rbf_c_0.25_gamma_1',
                'rbf_c_1_gamma_0.001','rbf_c_1_gamma_0.01','rbf_c_1_gamma_0.1','rbf_c_1_gamma_1',
                'rbf_c_4_gamma_0.001','rbf_c_4_gamma_0.01','rbf_c_4_gamma_0.1','rbf_c_4_gamma_1',
                'rbf_c_16_gamma_0.001','rbf_c_16_gamma_0.01','rbf_c_16_gamma_0.1','rbf_c_16_gamma_1']
    param_list = ['-t 2 -c 0.25 -g 0.001','-t 2 -c 0.25 -g 0.01','-t 2 -c 0.25 -g 0.1','-t 2 -c 0.25 -g 1',
                 '-t 2 -c 1 -g 0.001','-t 2 -c 1 -g 0.01','-t 2 -c 1 -g 0.1','-t 2 -c 1 -g 1',
                 '-t 2 -c 4 -g 0.001','-t 2 -c 4 -g 0.01','-t 2 -c 4 -g 0.1','-t 2 -c 4 -g 1',
                 '-t 2 -c 16 -g 0.001','-t 2 -c 16 -g 0.01','-t 2 -c 16 -g 0.1','-t 2 -c 16 -g 1']
    n = 16
    prob  = svm_problem(train_y, train_X)
    for i in range(n):
        m = svm_train(prob, param_list[i])
        p_labels, p_acc, p_vals = svm_predict(valid_y, valid_X, m)
        y_true = []
        for j in p_val:
            y_true.append(j[1])
        np.save(name_list[i]+'.npy',np.array(y_true))

## main function here
if __name__ == "__main__":
    test_kernels()
    find_rbf_param()

