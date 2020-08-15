import numpy as np
import timeit
import os
import sys
import six.moves.cPickle as pickle
import utils

#using tanh
class HiddenLayer(object):
    def __init__(self, rng, n_in, n_out, input=None,W=None, b=None,
                 activation='tanh'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=float
            )
            if activation == 'sigmoid':
                W_values *= 4
            self.W = W_values
        else:
            self.W = W




        if b is None:
            b_values = np.zeros((n_out,), dtype=float)
            self.b = b_values
        else:
            self.b = b

        #lin_output = np.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]
        self.output = None

    def forward(self,X):
        lin_output = np.dot(X,self.W) + self.b
        probs_pred = np.tanh(lin_output)
        self.output = probs_pred
        return probs_pred

    def get_output_delta(self,next_W,next_delta):
        derivative = 1 - self.output**2
        self.delta = np.dot(next_delta,next_W.transpose()) * derivative

    def update_w_and_b(self, x, learning_rate, L2_lamda):
        delta_w = - 1.0 * np.dot(x.transpose(), self.delta) / x.shape[0]
        delta_b = - 1.0 * np.mean(self.delta, axis=0)
        self.W -= learning_rate * (delta_w + L2_lamda * self.W)
        self.b -= learning_rate * delta_b



#using softmax
class OutputLayer(object):
    def __init__(self, rng,n_in, n_out, input = None,W=None, b=None,
                 activation='softmax'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        self.delta = 0
        self.n_in = n_in
        self.n_out = n_out
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=float
            )
            if activation == 'softmax':
                W_values *= 4
            self.W = W_values
        else:
            self.W = W

        if b is None:
            b_values = np.zeros((n_out,), dtype=float)
            self.b = b_values
        else:
            self.b = b

        #lin_output = np.dot(input, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]

        self.output = None

    def forward(self,X):
        lin_output = np.dot(X, self.W) + self.b
        probs = utils.softmax(lin_output)
        self.output = probs
        return probs

    def get_nll(self, y, probs_pred,L2_lamda):
        r"""
        Penalized negative log likelihood of the targets under the current
        model.
        """
        n_samples = probs_pred.shape[0]

        norm_beta = np.linalg.norm(self.W, ord=2)
        # get y_one_hot
        y_one_hot = utils.one_hot(
            label = y,
            n_samples = n_samples,
            n_classes = self.n_out
        )

        nll = -np.sum(y_one_hot * np.log(probs_pred))
        penalty = (L2_lamda / 2) * norm_beta ** 2
        # loss = (penalty + nll) / n_samples
        loss = penalty + (nll / n_samples)
        return loss

    def get_output_delta(self,y):
        probs_pred = self.output
        n_samples = probs_pred.shape[0]
        y_one_hot = utils.one_hot(
            label=y,
            n_samples=n_samples,
            n_classes= self.n_out
        )

        self.delta = y_one_hot - probs_pred

    def update_w_and_b(self ,x ,learning_rate, L2_lamda):
        delta_w = - 1.0 * np.dot(x.transpose(),self.delta)/x.shape[0]
        delta_b = - 1.0 * np.mean(self.delta,axis=0)
        self.W -= learning_rate * (delta_w + L2_lamda * self.W)
        self.b -= learning_rate * delta_b


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: list
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        self.input = input
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = []
        self.hidden_layer_number = len(n_hidden) #number of hidden layers
        n_in_tmp = n_in
        for num in n_hidden:
            hidden_layer = HiddenLayer(
                rng=rng,
                n_in=n_in_tmp,
                n_out=num,
                activation='tanh'
            )
            n_in_tmp = num
            self.hiddenLayer.append(hidden_layer)

        self.outputLayer = OutputLayer(
            rng=rng,
            n_in=n_in_tmp,
            n_out=n_out,
            activation='sigmoid'
        )


    def predict(self,input):
        input_tmp = input
        for hl in self.hiddenLayer:
            hl.forward(input_tmp)
            input_tmp = hl.output

        self.outputLayer.forward(input_tmp)
        return self.outputLayer.output

    def predict_class(self, X):
        probs = self.predict(X)
        return np.argmax(probs, axis=1).reshape((-1,1))

    def _errors(self, X, y):
        y_pred = self.predict_class(X)
        return np.mean(y != y_pred)

    def backpropagation(self,x,y,learning_rate,L2_lamda):
        #update w and b in output layer
        self.outputLayer.get_output_delta(y)
        x_input = self.hiddenLayer[-1].output
        self.outputLayer.update_w_and_b(x_input,learning_rate,L2_lamda)
        next_W = self.outputLayer.W
        next_delta = self.outputLayer.delta
        total = self.hidden_layer_number
        while total > 0:
            self.hiddenLayer[total-1].get_output_delta(next_W,next_delta)
            if total==1 :
                x_input = x
            else:
                x_input = self.hiddenLayer[total-2].output
            self.hiddenLayer[total-1].update_w_and_b(x_input,learning_rate,L2_lamda)

            next_W  = self.hiddenLayer[total-1].W
            next_delta = self.hiddenLayer[total-1].delta
            total = total-1



    def _train_model(self, X, y, lr=0.1,L2_lamda=0):
        probs_pred = self.predict(X)
        loss = self.outputLayer.get_nll(y,probs_pred,L2_lamda)
        self.backpropagation(X,y,lr,L2_lamda)
        return loss

    def _valid_model(self, X, y):
        errors = self._errors(X, y)
        return errors

    def _test_model(self, X, y):
        errors = self._errors(X, y)
        return errors

    def train(self, X, y, lr=0.1, n_epochs=1e3, batch_size = 200,patience_value = 5000):
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
                minibatch_avg_cost = self._train_model(
                    X = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
                    y = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size],
                    lr = lr
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


DATASET = 'mnist.pkl.gz'

def test():
    # get datasets
    datasets = utils.load_data(dataset = DATASET)
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = datasets
    (y_train, y_valid, y_test) = (y_train.reshape((-1,1)), y_valid.reshape((-1,1)), y_test.reshape((-1,1)))
    # test = x_train[0].copy(); utils.visualize_image(test)
    X, y = (x_train, x_valid, x_test), (y_train, y_valid, y_test)
    # get n_features & n_classes
    n_in = x_train.shape[1]
    n_out = 10

    rng = np.random.RandomState(1234)

    mlp_test = MLP(
        rng = rng,
        input = X,
        n_in = n_in,
        n_hidden = [500],
        n_out = n_out
    )
    # train mlp
    mlp_test.train(
        X = X,
        y = y
    )
    # test sr
    y_test_pred = mlp_test.predict_class(
        X = x_test
    )

    accu = np.sum(y_test_pred == y_test) / y_test.shape[0]
    print("test accu:", accu)

if __name__ == "__main__":
    test()




        


