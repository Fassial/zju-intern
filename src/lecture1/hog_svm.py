"""
Created on July 25 01:45, 2020

@author: fassial
"""
import math
import numpy as np
from sklearn.svm import SVC
# local dep
import preprocess

class hog_svm:

    def __init__(self):
        self.svm = SVC(
            C = 1.0
        )

    def train(x_train, y_train, batch_size = 100):
        n_cycle = math.ceil(x_train.shape[0] / batch_size)
        print("training...")
        for i in range(n_cycle):
            print("training..." + str(i) + "/" + str(n_cycle))
            xi_train = x_train[i*batch_size:(i+1)*batch_size,:] if (i+1)*batch_size <= x_train.shape[0] else x_train[i*batch_size:,:]
            yi_train = y_train[i*batch_size:(i+1)*batch_size] if (i+1)*batch_size <= y_train.shape[0] else y_train[i*batch_size:]
            self.svm.fit(xi_train, yi_train)

    def score(x_test, y_test, batch_size = 100):
        score = 0
        n_cycle = math.ceil(x_test.shape[0] / batch_size)
        print("testing...")
        for i in range(n_cycle):
            print("testing..." + str(i) + "/" + str(n_cycle))
            xi_test = x_test[i*batch_size:(i+1)*batch_size,:] if (i+1)*batch_size <= x_test.shape[0] else x_test[i*batch_size:,:]
            yi_test = y_test[i*batch_size:(i+1)*batch_size] if (i+1)*batch_size <= y_test.shape[0] else y_test[i*batch_size:]
            score += self.svm.score(xi_test, yi_test) * xi_test.shape[0]
        score /= x_test.shape[0]
        return score

if __name__ == "__main__":
    hog_svm_inst = hog_svm()
    x_train, y_train, x_test, y_test = preprocess.load_data()
    hog_svm_inst.train(x_train, y_train)
    score = hog_svm_inst.score(x_test, y_test)
    print("score: " + str(score))
