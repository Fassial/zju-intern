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
        ipath(str)  : the name of folder with images
        lpath(str)  : the name of folder with labels
        save_fig_path(str)  : folder to save ROC Curve
"""
def set_config():
    #read config file
    config = configparser.ConfigParser()
    config.readfp(open('config.ini'))
    ipath = config.get("user_para","ipath")
    lpath = config.get("user_para","lpath")
    save_fig_path = config.get("user_para", "save_fig_path")
    print("config setting completed")
    return ipath,lpath,save_fig_path


"""
split_data:
    compute hog and split data
    @params:
        ipath(str)  : the name of folder with pictures
        lpath(str)  : the name of folder with labels
    @rets:
        x_train(list)   : hog feature of training pictures
        x_test(list)    : label of training pictures
        y_train(list)   : hog feature of testing pictures
        y_test(list)    : label of testing pictures
"""
def split_data(ipath,lpath):
    #read image
    print("start reading images")
    image_data,image_label = read_image(ipath,lpath)
    print("reading images completed")
    #compute hog
    print("start computing hog")
    train_data = get_hog(image_data)
    train_target = image_label
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
    ipath,lpath,save_fig_path = set_config()
    #split data
    x_train, x_test, y_train, y_test = split_data(ipath,lpath)
    #draw plot
    train_and_test(x_train, x_test, y_train, y_test, save_fig_path)








