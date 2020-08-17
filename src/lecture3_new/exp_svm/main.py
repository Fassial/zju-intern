import configparser
import sys
sys.path.append("..")
from exp_svm import test_svm

def set_config():
    #read config file
    config = configparser.ConfigParser()
    config.readfp(open('config.ini'))
    ipath = config.get("user_para","ipath")
    lpath = config.get("user_para","lpath")
    save_fig_path = config.get("user_para", "save_fig_path")
    print("config setting completed")
    return ipath,lpath,save_fig_path

if __name__ == "__main__":
    #set config
    ipath, lpath, save_fig_path = set_config()
    test_svm = test_svm()
    test_svm.test_kernel(ipath,lpath,save_fig_path)
