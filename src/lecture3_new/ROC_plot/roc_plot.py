from libsvm.python.svmutil import *
'''
reference:
# Precomputed kernel data (-t 4)
# Dense data
y, x = [1,0,1], [[1, 2, -2], [2, -2, 2],[3,3,3]]
# Sparse data
#y, x = [1,-1], [{0:1, 1:2, 2:-2}, {0:2, 1:-2, 2:2}]
# isKernel=True must be set for precomputed kernel
param = '-t 0 -c 4 -b 1'

m = svm_train(y,x,param)
# For the format of precomputed kernel, please read LIBSVM README.


# Other utility functions
svm_save_model('./heart_scale.model', m)
m = svm_load_model('heart_scale.model')
p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
ACC, MSE, SCC = evaluations(y, p_label)
'''
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve
import numpy as np
from My_Explorer import MyExplorer

color_list = ['red','blue','black','chocolate','yellow','green','pink','violet','brown','darkblue','gray']
font = {'family': 'Times New Roman',  'weight': 'normal',  'size': 9, }


save_fig_path = '../figure_rel'
ipath = '../picture/allfull.jpg'
lpath = '../picture/label.npy'

def read_data():
    score_list = []
    number_list = []
    c_list = [0.25, 1, 4, 16]
    gamma_list = [0.001, 0.01, 0.1, 1]
    name_list = ['rbf_c_0.25_gamma_0.001', 'rbf_c_0.25_gamma_0.01', 'rbf_c_0.25_gamma_0.1', 'rbf_c_0.25_gamma_1',
                 'rbf_c_1_gamma_0.001', 'rbf_c_1_gamma_0.01', 'rbf_c_1_gamma_0.1', 'rbf_c_1_gamma_1',
                 'rbf_c_4_gamma_0.001', 'rbf_c_4_gamma_0.01', 'rbf_c_4_gamma_0.1', 'rbf_c_4_gamma_1',
                 'rbf_c_16_gamma_0.001', 'rbf_c_16_gamma_0.01', 'rbf_c_16_gamma_0.1', 'rbf_c_16_gamma_1']
    for i in range(8,16):
        c = np.load(os.path.join(save_fig_path,name_list[i] + ".npy"))
        score_list.append(c)
        num_tmp = [c_list[int(i/4)],gamma_list[int(i%4)]]
        number_list.append(num_tmp)
    return score_list,number_list


def draw_roc_plot(score_list,true_list,number_list,save_fig_path,fig_name):
    #name of x/y axis
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Recall')
    for i in range(0,len(score_list)):
        num = number_list[i]
        y_pred = score_list[i]
        y = true_list
        #compute fpr & tpr
        fpr, tpr, threshold = roc_curve(y,y_pred)
        plt.semilogx(fpr,tpr,color = color_list[i],label = 'rbf_c=%.2f_gamma=%.2f'%(num[0],num[1]))
        plt.legend(loc='upper left', prop=font, frameon=False)
    save_fig_path = os.path.join(save_fig_path,fig_name)
    plt.savefig(save_fig_path, format='svg')
    plt.show()


dataset = MyExplorer(ipath,lpath)
train_x,train_y,valid_x,valid_y = dataset.split_double()
test_number = int(60000*0.2)
valid_y = valid_y[0:test_number]
print('end reading')
fig_name = 'rbf_para_1.svg'
score_list,number_list = read_data()
draw_roc_plot(score_list,valid_y,number_list,save_fig_path,fig_name)

