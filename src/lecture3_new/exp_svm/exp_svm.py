#here
from sklearn import svm
import os
import sys
sys.path.append("..")
from load_data.load_data import *
# here


class test_svm(object):
    def get_score(self,score_list):
        score_rel = []
        for s in score_list:
            score_rel.append(s[1])
        return np.array(score_rel)

    def get_acc(self,y_pred,y):
        total = len(y_pred)
        correct = 0
        for i in range(0,total):
            if y_pred[i]==y[i]:
                correct += 1
        return correct/total

    def test_kernel(self,ipath,lpath,save_fig_path):

        train_X, train_y, valid_X, valid_y = load_data_and_get_hog(
            ipath=ipath,
            lpath=lpath,
        )

        kernel_list = ['linear', 'linear', 'rbf', 'poly']
        c_list = [0.5, 1.0, 4, 4]
        gamma_list = ['auto', 'auto', 0.1, 0.1]
        name_list = ['linear_c_0.5', 'linear_c_1', 'rbf_c_4_gamma_0.1', 'poly_c_4_gamma_0.1']
        n = 4
        for i in range(n):
            clf = svm.SVC(C=c_list[i], kernel=kernel_list[i], gamma=gamma_list[i], probability=True)
            clf.fit(train_X, train_y)
            score = clf.predict_proba(valid_X)
            y_pred = clf.predict(valid_X)
            print(self.get_acc(y_pred, valid_y))
            score = self.get_score(score)
            np.save(os.path.join(save_fig_path, name_list[i] + ".npy"), score)


    def find_rbf_param(self,ipath,lpath,save_fig_path):

        train_X,train_y,valid_X,valid_y = load_data_and_get_hog(
            ipath = ipath,
            lpath = lpath,
        )

        name_list = ['rbf_c_0.25_gamma_0.001','rbf_c_0.25_gamma_0.01','rbf_c_0.25_gamma_0.1','rbf_c_0.25_gamma_1',
                    'rbf_c_1_gamma_0.001','rbf_c_1_gamma_0.01','rbf_c_1_gamma_0.1','rbf_c_1_gamma_1',
                    'rbf_c_4_gamma_0.001','rbf_c_4_gamma_0.01','rbf_c_4_gamma_0.1','rbf_c_4_gamma_1',
                    'rbf_c_16_gamma_0.001','rbf_c_16_gamma_0.01','rbf_c_16_gamma_0.1','rbf_c_16_gamma_1']
        c_list = [0.25, 1, 4, 16]
        gamma_list = [0.001, 0.01, 0.1, 1]
        for i in range(0,len(c_list)):
            for j in range(0,len(gamma_list)):
                clf = svm.SVC(C=c_list[i], kernel='rbf', gamma=gamma_list[j],probability=True)
                clf.fit(train_X, train_y)
                score = clf.predict_proba(valid_X)
                y_pred = clf.predict(valid_X)
                print(self.get_acc(y_pred,valid_y))
                score = self.get_score(score)
                np.save(os.path.join(save_fig_path,name_list[i*4+j] + ".npy"),score)
