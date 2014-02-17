from dataset.timit import TIMIT


import utils
import time
import os.path
import os
import sys
import cPickle
from scikits.talkbox import segment_axis
import numpy as np

import theano
import theano.tensor as T
from theano import config
from nn import NetworkLayer,MSE



class NextSamplePredictor:
    """Multi-Layer Perceptron for predicting the next acoustic sample -
    inspired from Theano MLP tutorial http://deeplearning.net/tutorial/mlp.html
    """

    def __init__(self, input, rng, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a NetworkLayer with a tanh activation function connected to the
        # output layer
        self.hiddenLayer = NetworkLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.outputLayer =  NetworkLayer(rng=rng,
                                         input=self.hiddenLayer.output,
                                         n_in=n_hidden,
                                         n_out=n_out,
                                         activation=T.tanh)
                                         #activation=None)
        
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.hiddenLayer.W).sum() \
                + abs(self.outputLayer.W).sum()

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
                    + (self.outputLayer.W ** 2).sum()

        # the prediction is simply the output of the output layer
        self.y_pred = self.outputLayer.output
        
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.outputLayer.params

    
    def save_model(self,filename='sample_pred.save',
                   output_folder='output_folder'):
        """
        This function pickles the paramaters in a file for later usage
        """
        storage_file = open(os.path.join(output_folder,filename), 'wb')
        cPickle.dump(self, storage_file , protocol=cPickle.HIGHEST_PROTOCOL)
        storage_file.close()

        
    def load_model(filename='sample_pred.save',
                   output_folder='output_folder'):
        """
        This function loads pickled paramaters from a file
        """
        storage_file = open(os.path.join(output_folder,filename), 'rb')
        model = cPickle.load(storage_file)
        storage_file.close()
        return model

        
def build_data_sets(frame_len):
    """builds data sets for training/validating/testing the models"""
    
    print 'loading data...'

    save_stdout = sys.stdout
    sys.stdout = open('timit.log', 'w')

    # creating wrapper object for TIMIT dataset
    dataset = TIMIT()
    dataset.load("train")
    
    sys.stdout = save_stdout

    overlap = frame_len - 1

    wav_seqs = dataset.train_raw_wav[0:10]
    norm_seqs = utils.normalize(wav_seqs)
    
    # Segment into frames
    samples = map(lambda seq: segment_axis(seq, frame_len, overlap),
                  norm_seqs)

    # stack all data in one matrix, each row is a frame
    data = np.vstack(samples)
    # shuffle the frames so we can assume data is IID
    np.random.seed(123)
    data = np.random.permutation(data)

    # take 10% for test, 10% for valid, and 80% for training
    chunk = data.shape[0] / 10
    # now split data to x and y for train, valid, and test
    train_x = data[:8*chunk,:-1]
    train_y = data[:8*chunk,-1]
    valid_x = data[8*chunk:9*chunk,:-1]
    valid_y = data[8*chunk:9*chunk,-1]
    test_x = data[9*chunk:,:-1]
    test_y = data[9*chunk:,-1]


    print 'Done'
    print 'There are %d training samples'%train_x.shape[0]
    print 'There are %d validation samples'%valid_x.shape[0]
    
    return utils.shared_dataset((train_x,train_y)),\
           utils.shared_dataset((valid_x,valid_y)),\
           utils.shared_dataset((test_x,test_y))

        
        
def train_test_model(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
                     n_epochs=10, batch_size=100, frame_len=15*16,
                     n_hidden=200,
                     output_folder='output_folder'):
    """
    Trains the next acoustic sample predictor using SGD
    
    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)
    
    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)
    
    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type batch_size
    :param batch_size: number of training examples in each minibatch

    """
    
    
    rng = np.random.RandomState(1234)
    index = T.iscalar()  # index to a [mini]batch
    x = T.fmatrix('x')   # input data
    y = T.fvector('y')   # the target values
    
    datasets = build_data_sets(frame_len)
        
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print 'building the model...'

     # construct model class
    model = NextSamplePredictor(rng=rng, input=x,
                            n_in=frame_len-1,
                            n_hidden=n_hidden,
                            n_out=1)
    
    
    cost = MSE(y_pred=model.y_pred, y=y) \
           + L1_reg * model.L1 \
           + L2_reg * model.L2_sqr


    validate_model = theano.function(inputs=[index],
            outputs=MSE(y_pred=model.y_pred, y=y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    gparams = []
    for param in model.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []

    for param, gparam in zip(model.params, gparams):
        updates.append((param, param - learning_rate * gparam))


    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    
                    
    ###############
    # TRAIN MODEL #
    ###############
    print 'training...'

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    start_time = time.clock()
    epoch = 0

    total_training_losses = []
    total_validation_losses = []
    
    while (epoch < n_epochs):
        epoch = epoch + 1
        training_losses = []
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            training_losses.append(minibatch_avg_cost)
                
        
        this_training_loss = np.mean(training_losses)
        validation_losses = [validate_model(i) for i
                             in xrange(n_valid_batches)]
        this_validation_loss = np.mean(validation_losses)
            
        total_training_losses.append(this_training_loss)
        total_validation_losses.append(this_validation_loss)
            
        print 'epoch %i, training MSE %f, validation MSE %f' %\
        (epoch, this_training_loss,this_validation_loss)

    end_time = time.clock()

    print "Training took %.2f minutes..."%((end_time-start_time)/60.)

    model.save_model(output_folder=output_folder)

    #save losses and parameters..
    
    np.save(os.path.join(output_folder,'training_MSE'), np.asarray(total_training_losses))
    np.save(os.path.join(output_folder,'validation_MSE'), np.asarray(total_validation_losses))
    
if __name__ == "__main__":
    
    SAMPLE_PER_MS = 16
    FRAME_LEN_MS = 15
    
    frame_len = FRAME_LEN_MS * SAMPLE_PER_MS

    train_test_model(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
                     n_epochs=100, batch_size=1000, frame_len=frame_len,
                     n_hidden=500, output_folder='test_output')
    