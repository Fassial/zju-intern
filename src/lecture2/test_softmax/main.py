import os
import configparser
import argparse
# local dep
import train_softmax

exp_name_list = ["set_number","batch_size","lr","weight_decay","line_search","one_frame",""]

def set_config():
    #read config file
    config = configparser.ConfigParser()
    config.readfp(open('config.ini'))
    ipath = config.get("user_para","ipath")
    lpath = config.get("user_para","lpath")
    save_fig_path = config.get("user_para", "save_fig_path")
    if not os.path.exists(save_fig_path): os.mkdir(save_fig_path)
    print("config setting completed")
    return ipath,lpath,save_fig_path

def choose_exp(exp,ipath,lpath,save_fig_path,exp_name=None):
    if exp_name == exp_name_list[0]:
        print("exp0:diff_number")
        fig_name = "diff_number.svg"
        number_list = [1000 ,2000,4000,8000,16000,32000,48000,60000]
        exp.train_diff_number(ipath,lpath,number_list,is_one_frame=True,
                          save_fig_path = save_fig_path,fig_name = fig_name)

    elif exp_name == exp_name_list[1]:
        print("exp1:diff_batch_size")
        fig_name = "diff_batch_size.svg"
        number_list = [100, 200, 300, 400, 500, 600]
        exp.train_diff_batch_size(ipath, lpath, number_list, is_one_frame=True,
                              save_fig_path = save_fig_path,fig_name = fig_name)

    elif exp_name == exp_name_list[2]:
        print("exp2:diff_lr")
        fig_name = "diff_lr.svg"
        number_list = [0.1,0.2,0.3,0.4,0.5]
        exp.train_diff_lr(ipath, lpath, number_list, is_one_frame=True,
                        save_fig_path=save_fig_path, fig_name=fig_name)

    elif exp_name == exp_name_list[3]:
        print("exp3:weight_decay")
        fig_name = "weight_decay.svg"
        exp.if_weight_decay(ipath, lpath, is_one_frame=True,
                            save_fig_path=save_fig_path, fig_name=fig_name)

    elif exp_name == exp_name_list[4]:
        print("exp4:line_search")
        fig_name = "line_search.svg"
        exp.if_line_search(ipath, lpath, is_one_frame=True,
                            save_fig_path=save_fig_path, fig_name=fig_name)

    elif exp_name == exp_name_list[5]:
        print("exp5:one_frame")
        fig_name = "one_frame.svg"
        exp.if_one_frame(ipath, lpath,save_fig_path=save_fig_path, fig_name=fig_name)

    elif exp_name == exp_name_list[6]:
        print("exp6:early_stop")
        fig_name = "early_stop.svg"
        exp.if_early_stop(ipath, lpath,is_one_frame=True,
                          save_fig_path=save_fig_path, fig_name=fig_name)

    else:
        print("invalid input")





if __name__ == "__main__":
    #set config
    ipath, lpath, save_fig_path = set_config()
    parser = argparse.ArgumentParser(description="exp_train")
    #add argument number
    parser.add_argument('-n', '--number', type=int, default=0)
    n_classes = 2
    exp = train_softmax.exp_train(n_classes =n_classes)
    num = parser.parse_args().number
    exp_name = exp_name_list[num]
    choose_exp(exp,ipath,lpath,save_fig_path,exp_name=exp_name)
