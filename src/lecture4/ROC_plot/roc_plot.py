import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

color_list = ['red','blue','black','chocolate','yellow','green','pink','violet']
font = {'family': 'Times New Roman',  'weight': 'normal',  'size': 9, }
def draw_roc_plot(save_fig_path,fig_name):
    '''
    print(np.array(score_list).shape, np.array(true_list).shape, np.array(number_list).shape)
    #np.savetxt(os.path.join(save_fig_path,fig_name+"-scorelist"), np.array(score_list).reshape((np.array(score_list).shape[0],-1)), delimiter=',')
    #np.savetxt(os.path.join(save_fig_path,fig_name+"-truelist"), np.array(true_list).reshape((np.array(true_list).shape[0],-1)), delimiter=',')
    #np.savetxt(os.path.join(save_fig_path,fig_name+"-numlist"), np.array(number_list).reshape((np.array(number_list).shape[0],-1)), delimiter=',')
    np.save(os.path.join(save_fig_path,fig_name+"-scorelist.npy"),np.array(score_list))
    np.save(os.path.join(save_fig_path,fig_name+"-truelist.npy"),np.array(true_list))
    np.save(os.path.join(save_fig_path,fig_name+"-numlist.npy"),np.array(number_list))
    return
    '''

    score_list = pd.read_csv("../figure/two_layer.svg-scorelist", sep=',', header=None)
    score_list = np.array(score_list)
    true_list = pd.read_csv("../figure/two_layer.svg-truelist", sep=',', header=None)
    true_list = np.array(true_list)
    number_list = pd.read_csv("../figure/two_layer.svg-numlist", sep=',', header=None)
    number_list = np.array(number_list)
    #name of x/y axis
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Recall')
    for i in range(0,len(score_list)):
        num = number_list[i]
        y_pred = score_list[i]
        y = true_list[i]
        #compute fpr & tpr
        fpr, tpr, threshold = roc_curve(y,y_pred)
        plt.semilogx(fpr,tpr,color = color_list[i],label = '$number=%i$'%i)
        plt.legend(loc='upper left', prop=font, frameon=False)
    save_fig_path = os.path.join(save_fig_path,fig_name)
    plt.savefig(save_fig_path, format='svg')
    plt.show()

if __name__ == "__main__":
    save_fig_path = "../figure"
    fig_name = "one_layer.svg"
    draw_roc_plot(save_fig_path,fig_name)