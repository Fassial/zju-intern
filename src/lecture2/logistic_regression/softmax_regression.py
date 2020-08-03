"""
Created on August 01 19:57, 2020

@author: fassial
"""
import os
import sys
import timeit
import numpy as np
import six.moves.cPickle as pickle
# local dep
from . import utils
# import utils

class SoftmaxRegression:

    def __init__(self, n_features, n_classes, penalty="l2", gamma=0.01, fit_intercept=False):
        r"""
        A simple softmax regression model fit via gradient descent on the
        penalized negative log likelihood.
        Parameters
        ----------
        penalty : {'l1', 'l2'}
            The type of regularization penalty to apply on the coefficients
            `beta`. Default is 'l2'.
        gamma : float
            The regularization weight. Larger values correspond to larger
            regularization penalties, and a value of 0 indicates no penalty.
            Default is 0.
        fit_intercept : bool
            Whether to fit an intercept term in addition to the coefficients in
            b. If True, the estimates for `beta` will have `M + 1` dimensions,
            where the first dimension corresponds to the intercept. Default is
            True.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        # init sr params
        self.n_features = n_features
        self.n_classes = n_classes
        self.gamma = gamma
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        # set beta according to fit_intercept
        self.beta = np.random.rand(n_classes, n_features+1) \
            if self.fit_intercept else np.random.rand(n_classes, n_features)

    def _NLL(self, y, probs_pred):
        r"""
        Penalized negative log likelihood of the targets under the current
        model.
        """
        n_samples = probs_pred.shape[0]
        beta, gamma = self.beta, self.gamma
        # get norm_beta
        order = 2 if self.penalty == "l2" else 1
        norm_beta = np.linalg.norm(beta, ord=order)
        # get y_one_hot
        y_one_hot = utils.one_hot(
            label = y,
            n_samples = n_samples,
            n_classes = self.n_classes
        )

        nll = -np.sum(y_one_hot * np.log(probs_pred))
        penalty = (gamma / 2) * norm_beta ** 2 if order == 2 else gamma * norm_beta
        # loss = (penalty + nll) / n_samples
        loss = penalty + (nll / n_samples)
        return loss

    def _NLL_grad(self, X, y, probs_pred):
        """Gradient of the penalized negative log likelihood wrt beta"""
        n_samples = X.shape[0]
        l1norm = lambda x: np.linalg.norm(x, 1)  # noqa: E731
        p, beta, gamma = self.penalty, self.beta, self.gamma
        d_penalty = gamma * beta if p == "l2" else gamma * np.sign(beta)
        # get y_one_hot
        y_one_hot = utils.one_hot(
            label = y,
            n_samples = n_samples,
            n_classes = self.n_classes
        )
        # grad = -(np.dot((y_one_hot - probs_pred).T, X) + d_penalty) / n_samples
        grad = -(np.dot((y_one_hot - probs_pred).T, X) / n_samples) + d_penalty
        grad[:, 0] -= d_penalty[:, 0]
        return grad

    def forward(self, X):
        scores = X.dot(self.beta.T)
        probs = utils.softmax(scores)
        return probs

    def predict(self, X):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        probs = self.forward(X)
        return np.argmax(probs, axis=1).reshape((-1,1))

    def _errors(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y != y_pred)

    def _train_model(self, X, y, lr=0.1):
        # convert X to a design matrix if we're fitting an intercept
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # get probs
        probs_pred = self.forward(X)
        # get loss
        loss = self._NLL(
            y = y,
            probs_pred = probs_pred
        )
        # update beta
        self.beta -= lr * self._NLL_grad(X, y, probs_pred)
        # print("accu:", sum(np.argmax(probs_pred, axis=1).reshape((-1,1)) == y)/y.shape[0])
        return loss

    def _valid_model(self, X, y):
        errors = self._errors(X, y)
        return errors

    def _test_model(self, X, y):
        errors = self._errors(X, y)
        return errors

    def train(self, X, y, lr=0.1, n_epochs=1e3, batch_size = 600, patience = 5000):
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
        n_test_batches  = x_test.shape[0]  // batch_size

        # train
        print('... training the model')
        # early-stopping parameters
        # patience look as this many examples regardless
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

        valid_loss = []
        done_looping = False
        epoch = 0
        # compute zero-one loss on validation set
        validation_losses = [self._valid_model(
            X = x_valid[i*batch_size:(i+1)*batch_size],
            y = y_valid[i*batch_size:(i+1)*batch_size]
        ) for i in range(n_valid_batches)]
        # compute mean valid-loss
        this_validation_loss = np.mean(validation_losses)
        valid_loss.append(this_validation_loss)
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):
                # get loss of train-minibatch
                minibatch_avg_cost = self._train_model(
                    X = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
                    y = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
                )
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [self._valid_model(
                        X = x_valid[i*batch_size:(i+1)*batch_size],
                        y = y_valid[i*batch_size:(i+1)*batch_size]
                    ) for i in range(n_valid_batches)]
                    # compute mean valid-loss
                    this_validation_loss = np.mean(validation_losses)
                    valid_loss.append(this_validation_loss)
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
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set
                        test_losses = [self._test_model(
                            X = x_test[i*batch_size:(i+1)*batch_size],
                            y = y_test[i*batch_size:(i+1)*batch_size]
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
        return valid_loss

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
    sr_inst = SoftmaxRegression(
        n_features = n_features,
        n_classes = n_classes
    )
    # train sr
    valid_loss = sr_inst.train(
        X = X,
        y = y
    ); print(valid_loss)
    # test sr
    y_test_pred = sr_inst.predict(
        X = x_test
    )
    accu = np.sum(y_test_pred == y_test) / y_test.shape[0]
    print("test accu:", accu)

if __name__ == "__main__":
    test()

