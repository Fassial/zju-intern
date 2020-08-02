import numpy as np
# local dep
import sys
sys.path.append("..")
from test_softmax import test_softmax_classifier
from load_data.load_data import load_data
from ROC_plot.ROC_plot.draw_plot import draw_roc_plot

class exp_train():
    def __init__(self, n_classes, penalty="l2", fit_intercept=False):
        self.n_features = 0
        self.n_classes = n_classes
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    def train_diff_number(self,ipath,lpath,number_list,is_one_frame=True,
                          save_fig_path = None,fig_name = None):
        rel_score = []
        true_list = []
        for number in number_list:
            x_train,y_train,x_test,y_test = \
                load_data.load_data_and_get_hog(ipath,lpath,total_number=number,is_one_frame=is_one_frame)
            self.n_features = x_train.shape[1]
            test_sft = test_softmax_classifier(
                n_features=self.n_features,
                n_classes=self.n_classes,
            )
            X, y = (x_train, x_test, x_test), (y_train, y_test, y_test)

            test_sft.train(X,y)
            y_test_pred = test_sft.get_score(x_test)
            rel_score.append(y_test_pred)
            true_list.append(y_test)
        draw_roc_plot(rel_score,true_list,number_list,save_fig_path,fig_name)

    def train_diff_batch_size(self, ipath, lpath, number_list, is_one_frame=True,
                          save_fig_path=None, fig_name=None):
        rel_score = []
        true_list = []
        for number in number_list:
            x_train, y_train, x_test, y_test = \
                load_data.load_data_and_get_hog(ipath, lpath, is_one_frame=is_one_frame)
            self.n_features = x_train.shape[1]
            test_sft = test_softmax_classifier(
                n_features=self.n_features,
                n_classes=self.n_classes,
            )
            X, y = (x_train, x_test, x_test), (y_train, y_test, y_test)
            print("start training...")
            test_sft.train(X, y,batch_size=number)
            y_test_pred = test_sft.get_score(x_test)
            rel_score.append(y_test_pred)
            true_list.append(y_test)
        draw_roc_plot(rel_score, true_list, number_list, save_fig_path, fig_name)

    def train_diff_lr(self, ipath, lpath, number_list, is_one_frame=True,
                          save_fig_path=None, fig_name=None):
        rel_score = []
        true_list = []
        for number in number_list:
            x_train, y_train, x_test, y_test = \
                load_data.load_data_and_get_hog(ipath, lpath, is_one_frame=is_one_frame)
            self.n_features = x_train.shape[1]
            test_sft = test_softmax_classifier(
                n_features=self.n_features,
                n_classes=self.n_classes,
            )
            X, y = (x_train, x_test, x_test), (y_train, y_test, y_test)

            test_sft.train(X, y,lr=number)
            y_test_pred = test_sft.get_score(x_test)
            rel_score.append(y_test_pred)
            true_list.append(y_test)
        draw_roc_plot(rel_score, true_list, number_list, save_fig_path, fig_name)

    def if_weight_decay(self, ipath, lpath, is_one_frame=True,
                          save_fig_path=None, fig_name=None):
        number_list = [0,1]
        rel_score = []
        true_list = []
        for number in number_list:
            if number==0 : gamma = 0
            else: gamma = 0.1
            x_train, y_train, x_test, y_test = \
                load_data.load_data_and_get_hog(ipath, lpath, is_one_frame=is_one_frame)
            self.n_features = x_train.shape[1]
            test_sft = test_softmax_classifier(
                n_features=self.n_features,
                n_classes=self.n_classes,
                gamma = gamma
            )
            X, y = (x_train, x_test, x_test), (y_train, y_test, y_test)

            test_sft.train(X, y)
            y_test_pred = test_sft.get_score(x_test)
            rel_score.append(y_test_pred)
            true_list.append(y_test)
        draw_roc_plot(rel_score, true_list, number_list, save_fig_path, fig_name)

    def if_line_search(self, ipath, lpath, is_one_frame=True,
                          save_fig_path=None, fig_name=None):
        number_list = [0,1]
        rel_score = []
        true_list = []
        for number in number_list:
            x_train, y_train, x_test, y_test = \
                load_data.load_data_and_get_hog(ipath, lpath, is_one_frame=is_one_frame)
            self.n_features = x_train.shape[1]
            test_sft = test_softmax_classifier(
                n_features=self.n_features,
                n_classes=self.n_classes,
            )
            X, y = (x_train, x_test, x_test), (y_train, y_test, y_test)

            if number==0:test_sft.train(X, y)
            else: test_sft.line_search_train(X,y)
            y_test_pred = test_sft.get_score(x_test)
            rel_score.append(y_test_pred)
            true_list.append(y_test)
        draw_roc_plot(rel_score, true_list, number_list, save_fig_path, fig_name)

    def if_one_frame(self, ipath, lpath,save_fig_path=None, fig_name=None):
        number_list = [0,1]
        rel_score = []
        true_list = []
        for number in number_list:
            if number==0:
                x_train, y_train, x_test, y_test = \
                    load_data.load_data_and_get_hog(ipath, lpath, is_one_frame=False)
            else:
                x_train, y_train, x_test, y_test = \
                    load_data.load_data_and_get_hog(ipath, lpath, is_one_frame=True)
            self.n_features = x_train.shape[1]
            test_sft = test_softmax_classifier(
                n_features=self.n_features,
                n_classes=self.n_classes,
            )
            X, y = (x_train, x_test, x_test), (y_train, y_test, y_test)

            test_sft.train(X, y)
            y_test_pred = test_sft.get_score(x_test)
            rel_score.append(y_test_pred)
            true_list.append(y_test)
        draw_roc_plot(rel_score, true_list, number_list, save_fig_path, fig_name)

    def if_early_stop(self, ipath, lpath,is_one_frame=True,save_fig_path=None, fig_name=None):
        number_list = [0,1]
        rel_score = []
        true_list = []
        for number in number_list:
            x_train, y_train, x_test, y_test = \
                load_data.load_data_and_get_hog(ipath, lpath, is_one_frame=is_one_frame)
            self.n_features = x_train.shape[1]
            test_sft = test_softmax_classifier(
                n_features=self.n_features,
                n_classes=self.n_classes,
            )
            X, y = (x_train, x_test, x_test), (y_train, y_test, y_test)

            if number==0:test_sft.train(X,y,patience_value=float('inf'))
            else: test_sft.train(X,y)
            y_test_pred = test_sft.get_score(x_test)
            rel_score.append(y_test_pred)
            true_list.append(y_test)
        draw_roc_plot(rel_score, true_list, number_list, save_fig_path, fig_name)
















