"""
This script is a collection of objects and methods used for MDNs: inspired from Theano MLP tutorial http://deeplearning.net/tutorial
"""
__docformat__ = 'restructedtext en'


import cPickle
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T
import sys


class MDNoutputLayer(object):
    def __init__(self, rng, input, n_in, n_out, mu_activation, n_components):

        self.input = input

        W_mu_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_out, n_components)),
                              dtype=theano.config.floatX)
        b_values = np.zeros((n_out,n_components), dtype=theano.config.floatX)
        
        self.W_mu = theano.shared(value=W_mu_values, name='W_mu',
                                   borrow=True)
        
        # self.b_mu = theano.shared(value=b_values, name='b_mu', borrow=True)

        if mu_activation:
            self.mu = mu_activation(T.tensordot(input, self.W_mu,
                                              axes = [[1],[0]])) #+\
                                              # self.b_mu.dimshuffle('x',0,1))
        else:
            self.mu = T.tensordot(input, self.W_mu,
                                   axes = [[1],[0]]) #+\
                                   #self.b_mu.dimshuffle('x',0,1)

        W_values = np.asarray(rng.uniform(
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
            size=(n_in, n_components)),
                              dtype=theano.config.floatX)
        
        self.W_sigma = theano.shared(value=W_values, name='W_sigma',
                                     borrow=True)
        self.W_mixing = theano.shared(value=W_values.copy(), name='W_sigma',
                                      borrow=True)
        # b_values = np.zeros((n_components,), dtype=theano.config.floatX)
        
        self.b_sigma = theano.shared(value=b_values.copy(), name='b_sigma',
                                     borrow=True)
        self.b_mixing = theano.shared(value=b_values.copy(), name='b_mixing',
                                      borrow=True)
        
        self.sigma = T.nnet.softplus(T.dot(input, self.W_sigma)) #+\
                                      #self.b_sigma.dimshuffle('x',0))
        self.mixing = T.nnet.softmax(T.dot(input, self.W_mixing)) #+\
                                      #self.b_mixing.dimshuffle('x',0))
                                   
        # parameters of the model
        # self.params = [self.W_mu, self.b_mu, self.W_sigma, self.b_sigma,
        #                self.W_mixing, self.b_mixing]

        self.params = [self.W_mu, self.W_sigma, self.W_mixing]


        
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
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        
        self.output = (lin_output if activation is None
                       else activation(lin_output))
                           
        # parameters of the model
        self.params = [self.W, self.b]

        
    def set_symbolic_input(self, input):
        """We use this function to bind a symbolic variable with the input
        of the network layer. Added to specify that in training time."""
        self.input = input
        
        
        
class MDN(object):
    """Mixture Density Network
    """

    def __init__(self, input, rng, n_in, n_hiddens, hid_activations,
                 n_out, out_activation, n_components):
        """Initialize the parameters for the multilayer perceptron

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden_list: list of int
        :param n_hidden_list: a list of number of units in each hidden layer

        :type activations_list: list of lambdas
        :param n_hidden_list: a list of activations used in each hidden layer
        
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        from theano.tensor.shared_randomstreams import RandomStreams
        self.srng = RandomStreams(seed=1234)
        
        self.input = input

        # We are dealing with multiple hidden layers MLP
        layer0 = NetworkLayer(rng=rng, input=input,
                              n_in=n_in, n_out=n_hiddens[0],
                              activation=hid_activations[0])

        h_layers = [('hiddenLayer0',layer0)]
        
        for i in range(1,len(n_hiddens)):
            h_layers.append(('hiddenLayer%d'%i,
                        NetworkLayer(rng=rng, input=h_layers[i-1][1].output,
                                     n_in=n_hiddens[i-1], n_out=n_hiddens[i],
                                     activation=hid_activations[i])))
                        
        self.__dict__.update(dict(h_layers))
        
        # The output layer gets as input the hidden units
        # of the hidden layer
        self.outputLayer =  MDNoutputLayer(rng=rng,
                                           input=h_layers[-1][1].output,
                                           n_in=n_hiddens[-1],
                                           n_out=n_out,
                                           mu_activation=out_activation,
                                           n_components=n_components)
        
        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.outputLayer.W_mu ** 2).sum() + \
                      (self.outputLayer.W_sigma ** 2).sum() +\
                      (self.outputLayer.W_mixing ** 2).sum()
        
        for i in range(len(n_hiddens)):
            self.L2_sqr += (self.__dict__['hiddenLayer%d'%i].W ** 2).sum()

            
        # the parameters of the model are the parameters of the all layers it
        # is made out of
        params = self.outputLayer.params
        for layer in h_layers:
            params.extend(layer[1].params)
        self.params = params

            
    def set_symbolic_input(self, input):
        """We use this function to bind a symbolic variable with the input
        of the network layer. Added to specify that in training time."""
        self.input = input


#    def train(self, x, y, training_loss, learning_rate,
    def train(self, y, training_loss, learning_rate,
              n_epochs, train_x, train_y, valid_x, valid_y, batch_size):
        """Train the MLP using SGD"""

        index = T.iscalar()  # index to a [mini]batch
        lr = T.scalar() # learning rate symbolic

        #index.tag.test_value = 1
        gparams = []
        for param in self.params:
            gparam = T.grad(training_loss, param)
            gparams.append(gparam)

        updates = []

        
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * \
                            T.cast(lr,dtype=theano.config.floatX)))
        
        
        train_model = theano.function(inputs=[index, lr],
            outputs=[training_loss],
            updates=updates,
            givens={
                self.input: train_x[index * batch_size:(index+1) * batch_size],
                y: train_y[index * batch_size:(index + 1) * batch_size]})

        
        validate_model = theano.function(inputs=[index],
            outputs=NLL(mu = self.outputLayer.mu,
                        sigma = self.outputLayer.sigma,
                        mixing = self.outputLayer.mixing,
                        y = y),
            givens={
                self.input: valid_x[index * batch_size:(index+1) * batch_size],
                y: valid_y[index * batch_size:(index + 1) * batch_size]})

        # compute number of minibatches for training and validation
        n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size
        n_valid_batches = valid_x.get_value(borrow=True).shape[0] / batch_size


    
        validate_MSE = theano.function(inputs=[index],
            outputs=MSE(self.samples(), y = y),
            givens={
                self.input: valid_x[index * batch_size:(index+1) * batch_size],
                y: valid_y[index * batch_size:(index + 1) * batch_size]})

        print 'training...'

        start_time = time.clock()
        epoch = 0

        total_training_costs = []
        total_validation_costs = []
        total_validation_MSE = []
        
        lr_time = 0
        lr_step = learning_rate / ((train_x.get_value().shape[0]*1.0/batch_size)*(n_epochs-30))
        lr_val = learning_rate
        
        while (epoch < n_epochs):
            epoch = epoch + 1
            epoch_training_costs = []
            #import pdb; pdb.set_trace()
            for minibatch_index in xrange(n_train_batches):
                
                # linear annealing after 40 epochs...
                if epoch > 40:
                    # lr_val = learning_rate / (1.0+lr_time)
                    # lr_time = lr_time + 1
                    lr_val = lr_val - lr_step
                else:
                    lr_val = learning_rate
                

                loss_value = \
                                train_model(minibatch_index, lr_val)
                epoch_training_costs.append(loss_value)

                if np.isnan(loss_value):
                    print 'got NaN in NLL'
                    sys.exit(1)
                
            this_training_cost = np.mean(epoch_training_costs)
            this_validation_cost = np.mean([validate_model(i) for i
                                            in xrange(n_valid_batches)])
            this_validation_MSE = np.mean([validate_MSE(i) for i
                                            in xrange(n_valid_batches)])

            total_training_costs.append(this_training_cost)
            total_validation_costs.append(this_validation_cost)
            total_validation_MSE.append(this_validation_MSE)
            
            print 'epoch %i, training NLL %f, validation NLL %f, MSE %f' %\
            (epoch, this_training_cost,this_validation_cost,
             this_validation_MSE)

        end_time = time.clock()

        print "Training took %.2f minutes..."%((end_time-start_time)/60.)

        #return losses and parameters..
        return total_training_costs, total_validation_costs,total_validation_MSE


    def samples(self):
        component = self.srng.multinomial(pvals=self.outputLayer.mixing)
        component_mean =  T.sum(self.outputLayer.mu * \
                                component.dimshuffle(0,'x',1),
                                axis=2)
        component_std = T.sum(self.outputLayer.sigma * \
                              component, axis=1, keepdims=True)
        
        samples = self.srng.normal(avg = component_mean, std=component_std)
        return samples
        
    def save_model(self,filename='MLP.save',
                   output_folder='output_folder'):
        """
        This function pickles the paramaters in a file for later usage
        """
        storage_file = open(os.path.join(output_folder,filename), 'wb')
        cPickle.dump(self, storage_file , protocol=cPickle.HIGHEST_PROTOCOL)
        storage_file.close()

        
    @staticmethod
    def load_model(filename='MLP.save',
                   output_folder='output_folder'):
        """
        This function loads pickled paramaters from a file
        """
        storage_file = open(os.path.join(output_folder,filename), 'rb')
        model = cPickle.load(storage_file)
        storage_file.close()
        return model

def Rectifier(x):
    """Implementation of the rectifier activation function"""
    return T.switch(x>0, x, 0)


def NLL(mu, sigma, mixing, y):
    """Computes the mean of negative log likelihood for P(y|x)
    
    y = T.matrix('y') # (minibatch_size, output_size)
    mu = T.tensor3('mu') # (minibatch_size, output_size, n_components)
    sigma = T.matrix('sigma') # (minibatch_size, n_components)
    mixing = T.matrix('mixing') # (minibatch_size, n_components)

    """
    
    # multivariate Gaussian
    exponent = -0.5 * T.inv(sigma) * T.sum((y.dimshuffle(0,1,'x') - mu)**2, axis=1)
    normalizer = (2 * np.pi * sigma)
    exponent = exponent + T.log(mixing) - (y.shape[1]*.5)*T.log(normalizer) 
    max_exponent = T.max(exponent ,axis=1, keepdims=True)
    mod_exponent = exponent - max_exponent
    gauss_mix = T.sum(T.exp(mod_exponent),axis=1)
    log_gauss = max_exponent + T.log(gauss_mix) 
    res = -T.mean(log_gauss)
    return res

    

def MSE(samples, y):
    
    return T.mean((samples - y) ** 2)