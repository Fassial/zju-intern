"""
Created on August 01 02:06, 2020

@author: fassial
"""
import numpy as np

def sigmoid(x):
    """The logistic sigmoid function"""
    return 1 / (1 + np.exp(-x))
