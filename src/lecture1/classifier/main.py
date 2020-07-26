from get_data.get_data import read_image
from binary_classification.binary_classification import get_hog,get_roc_data,draw_roc_data
from sklearn.model_selection import train_test_split
import configparser


"""
set_config:
    set the path of file
    @params:
        None
    @rets:
        positive_image_path(str)    : the name of 
        negative_image_path_v1(str)     : the name of file with negative pictures in v1
        negative_image_path_v2(str)     : the name of file with negative pictures in v2
        negative_image_path_v3(str)     : the name of file with negative pictures in v3
        save_fig_path(str)  : folder to save ROC Curve
"""
def set_config():
    #read config file
    config = configparser.ConfigParser()
    config.readfp(open('config.ini'))
    positive_image_path = config.get("user_para","positive_image_path")
    negative_image_path_v1 = config.get("user_para","negative_image_path_v1")
    negative_image_path_v2 = config.get("user_para","negative_image_path_v2")
    negative_image_path_v3 = config.get("user_para","negative_image_path_v3")
    save_fig_path = config.get("user_para", "save_fig_path")
    print("config setting completed")
    return positive_image_path,negative_image_path_v1,negative_image_path_v2,negative_image_path_v3,save_fig_path


"""
split_data:
    compute hog and split data
    @params:
        positive_image_path(str)    : the name of 
        negative_image_path_v1(str)     : the name of file with negative pictures in v1
        negative_image_path_v2(str)     : the name of file with negative pictures in v2
        negative_image_path_v3(str)     : the name of file with negative pictures in v3
    @rets:
        x_train(list)   : hog feature of training pictures
        x_test(list)    : label of training pictures
        y_train(list)   : hog feature of testing pictures
        y_test(list)    : label of testing pictures
"""
def split_data(positive_image_path,negative_image_path_v1,negative_image_path_v2,negative_image_path_v3):
    #read image
    print("start reading images")
    image_data_positive,positive_label = read_image(positive_image_path,isPositive=True)
    image_data_negative_v1, negative_label_v1 = read_image(negative_image_path_v1,isPositive=False)
    image_data_negative_v2, negative_label_v2 = read_image(negative_image_path_v2, isPositive=False)
    image_data_negative_v3, negative_label_v3 = read_image(negative_image_path_v3, isPositive=False)
    print("reading images completed")
    #compute hog
    print("start computing hog")
    feature_positive = get_hog(image_data_positive)
    feature_negative_v1 = get_hog(image_data_negative_v1)
    feature_negative_v2 = get_hog(image_data_negative_v2)
    feature_negative_v3 = get_hog(image_data_negative_v3)
    train_data = feature_positive+feature_negative_v1+feature_negative_v2+feature_negative_v3
    train_target = positive_label+negative_label_v1+negative_label_v2+negative_label_v3
    print("computing hog completed")
    #split data
    print("start splitting data")
    x_train,x_test,y_train,y_test = train_test_split(train_data,train_target,test_size = 0.3)
    print("splitting data completed")
    return x_train,x_test,y_train,y_test


"""
train_and_test:
    draw the ROC_Curve
    @params:
        x_train(list)   : hog feature of training pictures
        x_test(list)    : label of training pictures
        y_train(list)   : hog feature of testing pictures
        y_test(list)    : label of testing pictures
        save_fig_path(str)  : folder to save ROC Curve
    @rets:
        None
"""
def train_and_test(x_train,x_test,y_train,y_test,save_fig_path):
    train_positive_list = []
    for i in range(0,len(x_train)):
        if y_train[i]==1:
            train_positive_list.append(x_train[i])
    print("start training")
    roc_data = get_roc_data(x_test,y_test,train_positive_list)
    draw_roc_data(roc_data,save_fig_path)
    print("ROC_Curve completed")

if __name__ == "__main__":
    #set config
    positive_image_path,negative_image_path_v1,negative_image_path_v2,\
    negative_image_path_v3,save_fig_path = set_config()
    #split data
    x_train, x_test, y_train, y_test = split_data(positive_image_path,negative_image_path_v1,
                                                  negative_image_path_v2,negative_image_path_v3)
    #draw plot
    train_and_test(x_train, x_test, y_train, y_test, save_fig_path)








