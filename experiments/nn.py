"""
This script is a collection of objects and methods used for MLPs: inspired from Theano MLP tutorial http://deeplearning.net/tutorial
"""
__docformat__ = 'restructedtext en'


import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class NetworkLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical layer of a MLP: units are fully-connected and have
        tanh activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

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

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
                           
        # parameters of the model
        self.params = [self.W, self.b]



def MSE(y_pred, y):
    """Return the mean of squared error, i.e. the difference between
    the predicted output and target

    :type y_pred: theano.tensor.TensorType
    :param y: corresponds to a vector that gives the prediction fo each example

    :type y: theano.tensor.TensorType
    :param y: corresponds to a vector that gives the target for each example
    
    """
    # using dimshuffle to convert the batch output from (batch_size,) to
    # column vector (batch_size,1) so it aligns with the network output
    return T.mean((y_pred - y.dimshuffle(0,'x')) ** 2)
