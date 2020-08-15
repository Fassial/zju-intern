import sys
#local dep
import sys
import numpy as np
sys.path.append("..")
from load_data import load_data
from ROC_plot.draw_plot import draw_roc_plot
from bp_network.bp_without_theano import MLP

class testBp(object):
    def get_score(self,prob_list):
        score = []
        for p in prob_list:
            score.append(p[1])
        return np.array(score)

    def test_one_layer(self,ipath,lpath,save_fig_path, fig_name):
        x_train, y_train, x_test, y_test = \
            load_data.load_data_and_get_hog(ipath, lpath, is_one_frame=False)
        X, y = (x_train, x_test, x_test), (y_train, y_test, y_test)

        self.n_in = x_train.shape[1]
        self.n_out = 2

        rng = np.random.RandomState(1234)

        num_list = [[2],[10],[20],[50],[100],[500]]

        rel_score = []
        true_list = []
        draw_number = []

        for n in num_list:
            mlp_test = MLP(
                rng=rng,
                input=X,
                n_in=self.n_in,
                n_hidden=n,
                n_out=self.n_out
            )

            mlp_test.train(
                X=X,
                y=y
            )

            prob_list = mlp_test.predict(x_test)
            prob_score = self.get_score(prob_list)
            rel_score.append(prob_score)
            true_list.append(y_test)
            draw_number.append(n[0])

        draw_roc_plot(rel_score, true_list, draw_number, save_fig_path, fig_name)

    def test_two_layer(self,ipath,lpath,save_fig_path, fig_name):
        x_train, y_train, x_test, y_test = \
            load_data.load_data_and_get_hog(ipath, lpath, is_one_frame=False)
        X, y = (x_train, x_test, x_test), (y_train, y_test, y_test)

        self.n_in = x_train.shape[1]
        self.n_out = 2

        rng = np.random.RandomState(1234)

        num_list = [[20,8],[10,4],[50,10]]

        rel_score = []
        true_list = []
        draw_number = []

        for n in num_list:
            mlp_test = MLP(
                rng=rng,
                input=X,
                n_in=self.n_in,
                n_hidden=n,
                n_out=self.n_out
            )

            mlp_test.train(
                X=X,
                y=y
            )

            prob_list = mlp_test.predict(x_test)
            prob_score = self.get_score(prob_list)
            rel_score.append(prob_score)
            true_list.append(y_test)
            draw_number.append(n)

        draw_roc_plot(rel_score, true_list, draw_number, save_fig_path, fig_name)






