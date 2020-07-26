import sys
sys.path.append('..')
from hog.hog import hog
import numpy as np
import matplotlib.pyplot as plt

"""
get_hog:
    @params:
        image_list(list) : the list of image to compute hog
    @rets:
        feature_list(list)  : the list of hog of pictures
"""
def get_hog(image_list):
    feature_list = []
    for image in image_list:
        #compute hog
        normalised_blocks,hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4), visualize=True)
        feature_list.append(normalised_blocks)
    return feature_list



"""
get_positive_centre:
    @params:
        feature_list(list)  : the list of hog of pictures
    @rets:
        positive_centre(list)  : the centre of given pictures
"""
def get_positive_centre(feature_list):
    #compute the centre of positive training set
    positive_centre = np.mean(feature_list,axis = 0)
    return  positive_centre


"""
get_distance:
    @params:
        feature_list_1(list)  : hog of one picture
        feature_list_2(list)  : hog of the other picture
    @rets:
        result(float) : the Euclidean distance between two pictures
"""
def get_distance(feature_list_1,feature_list_2):
    #compute the distance between
    result = np.sqrt(np.sum(np.square(feature_list_1-feature_list_2)))
    return result


"""
test_classifier:
    @params:
        feature_list_test(list)  : list of hog of testing pictures
        feature_list_train_positive(list)  : list of hog of positive training picture
    @rets:
        result_distance(list) : the Euclidean distance between testing pictures and centre of 
                                positive training pictures
"""
def test_classifier(feature_list_test,feature_list_train_positive):
    positive_centre = get_positive_centre(feature_list_train_positive)
    result_distance = []
    for ft in feature_list_test:
        tmpRel = get_distance(ft,positive_centre)
        result_distance.append(tmpRel)
    return result_distance


"""
threshold_classifier:
    do classification according to the given threshold
    @params:
        threshold(float)  : given threshold
        distance_list(list)  : the Euclidean distance between testing pictures and centre of 
                               positive training pictures
    @rets:
        result_label(list) : list of label
"""
def threshold_classifier(threshold,distance_list):
    result_label = []
    for distance in distance_list:
        #positive
        if distance<=threshold:
            result_label.append(1)
        #negative
        else:
            result_label.append(0)
    return result_label


"""
get_tpr_and_fpr:
    get tpr and fpr
    @params:
        result_label(list)  : the result list of label 
        test_label(list)  : list of true label of testing pictures
    @rets:
        tpr(float) : true positive rate
        fpr(float) : false positive rate
"""
def get_tpr_and_fpr(result_label,test_label):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(0,len(result_label)):
        if result_label[i]==test_label[i]:
            if result_label[i]==1:
                tp = tp + 1
            else:
                tn = tn + 1
        else:
            if result_label[i]==1:
                fp = fp + 1
            else:
                fn = fn + 1
    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    return tpr,fpr


"""
get_roc_data:
    get roc data for ROC Curve
    @params:
        feature_list_test(list)  : the list of hog of testing pictures
        feature_list_train_positive(list)  : list of hog of positive training picture
    @rets:
        roc_data(list)  : list of tpr & fpr of each given threshold
"""
def get_roc_data(feature_list_test,test_label,feature_list_train_positive):
    roc_data = []
    result_distance = test_classifier(feature_list_test,feature_list_train_positive)
    #get tpr,fpr according to threshold
    for threshold in result_distance:
        result_label = threshold_classifier(threshold,result_distance)
        tpr,fpr = get_tpr_and_fpr(result_label,test_label)
        roc_data.append([fpr,1-tpr])
    return roc_data


"""
draw_roc_data:
    @params:
        roc_data(list)  : list of tpr & fpr of each given threshold
        save_fig_path(str)  : folder to save ROC Curve
    @rets:
        None
"""
def draw_roc_data(roc_data,save_fig_path):
    x_list = []
    y_list = []
    for data in roc_data:
        x_list.append(data[0])
        y_list.append(data[1])
    #title
    plt.figure('ROC Curve')
    #range of x/y axis
    x_ticks = np.arange(0,1,0.1)
    y_ticks = np.arange(0,1,0.1)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    #name of x/y axis
    plt.xlabel('FPR')
    plt.ylabel('1-TPR')
    #draw the plot
    plt.scatter(x_list, y_list, s=10, c=None, marker='o')
    plt.savefig(save_fig_path,format = 'svg')
    plt.show()














