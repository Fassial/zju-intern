import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import roc_curve

color_list = ['red','blue','black','chocolate','yellow','green','pink','violet']
font = {'family': 'Times New Roman',  'weight': 'normal',  'size': 9, }
def draw_roc_plot(score_list,true_list,number_list,save_fig_path,fig_name):
    print(np.array(score_list).shape, np.array(true_list).shape, np.array(number_list).shape)
    np.savetxt(os.path.join(save_fig_path,fig_name+"-scorelist"), np.array(score_list).reshape((np.array(score_list).shape[0],-1)), delimiter=',')
    np.savetxt(os.path.join(save_fig_path,fig_name+"-truelist"), np.array(true_list).reshape((np.array(true_list).shape[0],-1)), delimiter=',')
    np.savetxt(os.path.join(save_fig_path,fig_name+"-numlist"), np.array(number_list).reshape((np.array(number_list).shape[0],-1)), delimiter=',')
    return
    #name of x/y axis
    plt.xlabel('False Alarm Rate')
    plt.ylabel('Recall')
    for i in range(0,len(score_list)):
        num = number_list[i]
        y_pred = score_list[i]
        y = true_list[i]
        #compute fpr & tpr
        fpr, tpr, threshold = roc_curve(y,y_pred)
        plt.semilogx(fpr,tpr,color = color_list[i],label = '$number=%i$'%num)
        plt.legend(loc='upper right', prop=font, frameon=False)
    save_fig_path = os.path.join(save_fig_path,fig_name)
    plt.savefig(save_fig_path, format='svg')
    # plt.show()


