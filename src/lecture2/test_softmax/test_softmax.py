import os
import copy
import sys
import timeit
import numpy as np
import six.moves.cPickle as pickle
# local dep
sys.path.append("..")
from logistic_regression.softmax_regression import SoftmaxRegression
from logistic_regression import utils

class test_softmax_classifier(SoftmaxRegression):
    def __init__(self, n_features, n_classes, penalty="l2", gamma=0, fit_intercept=False):
        super().__init__(n_features, n_classes, penalty, gamma, fit_intercept)

    #get score to draw roc_curve
    def get_score(self,X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        probs = self.forward(X)
        index_max = np.argmax(probs, axis=1)
        scores = []
        for i in range(0,len(index_max)):
            if index_max[i]==1:
                scores.append(probs[i][index_max[i]])
            else:
                scores.append(1-probs[i][index_max[i]])
        return np.array(scores).reshape(-1, 1)

    def line_search_lr(self,X,y,down_learning_rate=1e-4):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        ori_beta = copy.deepcopy(self.beta)
        # get probs
        probs_pred = self.forward(X)
        # get loss
        original_loss = self._NLL(
            y=y,
            probs_pred=probs_pred
        )
        p = self._NLL_grad(X,y,probs_pred)
        slope = np.dot(p,p.T).sum(axis=0)
        cur_learning_rate = 0.5
        self.beta = ori_beta - cur_learning_rate*p
        cur_loss = 0
        c = 0.5
        k = 0.8
        # update beta
        for i in range(0,self.n_classes):
            cur_learning_rate = 0.5
            while 1:
                probs_pred = self.forward(X)
                cur_loss = self._NLL(y,probs_pred)
                t = c*slope[i]
                if cur_loss <= original_loss+t:
                    break
                else:
                    cur_learning_rate *= k
                    if cur_learning_rate < down_learning_rate:
                        break
                    self.beta[i] = ori_beta[i]-cur_learning_rate * p[i]
            original_loss = cur_loss
        return cur_loss


    def line_search_train(self, X, y, n_epochs=1e3, batch_size=200,patience_value = 5000):
        """
        Fit the regression coefficients via gradient descent on the negative
        log likelihood.
        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The binary targets for each of the `N` examples in `X`.
        lr : float
            The gradient descent learning rate. Default is 1e-7.
        max_iter : float
            The maximum number of iterations to run the gradient descent
            solver. Default is 1e7.
        """
        # get trainset & validset & testset
        x_train, x_valid, x_test = X[0], X[1], X[2]
        y_train, y_valid, y_test = y[0], y[1], y[2]
        # get n_batches
        n_train_batches = x_train.shape[0] // batch_size
        n_valid_batches = x_valid.shape[0] // batch_size
        n_test_batches = x_test.shape[0] // batch_size

        # train
        print('... training the model')
        # early-stopping parameters
        patience = patience_value  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                # get loss of train-minibatch
                minibatch_avg_cost = self.line_search_lr(
                    X=x_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size],
                    y=y_train[minibatch_index * batch_size:(minibatch_index + 1) * batch_size]
                )
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [self._valid_model(
                        X=x_valid[i * batch_size:(i + 1) * batch_size],
                        y=y_valid[i * batch_size:(i + 1) * batch_size]
                    ) for i in range(n_valid_batches)]
                    # compute mean valid-loss
                    this_validation_loss = np.mean(validation_losses)
                    # print valid info
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%' %
                        (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set
                        test_losses = [self._test_model(
                            X=x_test[i * batch_size:(i + 1) * batch_size],
                            y=y_test[i * batch_size:(i + 1) * batch_size]
                        ) for i in range(n_test_batches)]
                        # compute mean test-loss
                        test_score = np.mean(test_losses)
                        # print test info
                        print(
                            (
                                '     epoch %i, minibatch %i/%i, test error of'
                                ' best model %f %%'
                            ) %
                            (
                                epoch,
                                minibatch_index + 1,
                                n_train_batches,
                                test_score * 100.
                            )
                        )
                        # save the best model
                        with open('best_model.pkl', 'wb') as f:
                            pickle.dump(self, f)

                if patience <= iter:
                    done_looping = True
                    break

        # print train info
        end_time = timeit.default_timer()
        print(
            (
                'Optimization complete with best validation score of %f %%,'
                'with test performance %f %%'
            )
            % (best_validation_loss * 100., test_score * 100.)
        )
        print('The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time)))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.1fs' % ((end_time - start_time))), file=sys.stderr)

DATASET = 'mnist.pkl.gz'

def test():
    # get datasets
    datasets = utils.load_data(dataset = DATASET)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = datasets
    (y_train, y_valid, y_test) = (y_train.reshape((-1,1)), y_valid.reshape((-1,1)), y_test.reshape((-1,1)))
    # test = x_train[0].copy(); utils.visualize_image(test)
    X, y = (x_train, x_valid, x_test), (y_train, y_valid, y_test)
    # get n_features & n_classes
    n_features = x_train.shape[1]
    n_classes = 10
    # inst SoftmaxRegression
    sr_inst = test_softmax_classifier(
        n_features = n_features,
        n_classes = n_classes
    )
    # train sr
    sr_inst.line_search_train(
        X = X,
        y = y
    )
    # test sr
    y_test_pred = sr_inst.predict(
        X = x_test
    )
    accu = np.sum(y_test_pred == y_test) / y_test.shape[0]
    print("test accu:", accu)

if __name__ == "__main__":
    test()
