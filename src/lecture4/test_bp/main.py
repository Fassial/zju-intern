import configparser
import argparse

from test_bp import testBp

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
    parser = argparse.ArgumentParser(description="test_bp")
    #add argument number
    parser.add_argument('-n', '--number', type=int, default=0)
    num = parser.parse_args().number
    test_bp = testBp()
    if num==1:
        fig_name = "one_layer.svg"
        test_bp.test_one_layer(
            ipath = ipath,
            lpath = lpath,
            save_fig_path = save_fig_path,
            fig_name = fig_name
        )

    elif num==2:
        fig_name = "two_layer.svg"
        test_bp.test_two_layer(
            ipath = ipath,
            lpath = lpath,
            save_fig_path = save_fig_path,
            fig_name = fig_name
        )

    else:
        print("invalud output!")