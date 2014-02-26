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
from experiments.nn import MLP, MSE, Rectifier
from datasets_builder import build_data_sets
from theano import config


def prepare_data(in_samples, out_samples, shift, win_width):

    print 'loading data...'

    save_stdout = sys.stdout
    sys.stdout = open('timit.log', 'w')

    # creating wrapper object for TIMIT dataset
    dataset = TIMIT()
    dataset.load("train")
    dataset.load("valid")
    
    sys.stdout = save_stdout
    
    n_spkr = 10
    n_utts = 10
    shuffle = True
    
    # each training example has 'in_sample' inputs and 'out_samples' output
    # and examples are shifted by 'shift'
    train_x, train_y1, train_y2 = \
        build_data_sets(dataset, 'train', n_spkr, n_utts,
                        in_samples, out_samples, shift,
                        win_width,shuffle)

    n_spkr = 1
    n_utts = 10
    shuffle = False
    valid_x, valid_y1, valid_y2 = \
        build_data_sets(dataset, 'valid', n_spkr, n_utts,
                        in_samples, out_samples, shift,
                        win_width,shuffle)
    
    return (train_x, train_y1, train_y2), (valid_x, valid_y1, valid_y2)

        
def train_model(datasets, learning_rate=0.01, L2_reg=0.0001,
                n_epochs=10, batch_size=100,
                in_samples = 240, out_samples = 50, shift=10, win_width=2,
                output_folder='output_folder'):
        
    train_x, train_y1, train_y2 = datasets[0]
    valid_x, valid_y1, valid_y2 = datasets[1]


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print 'building the model...'

    rng = np.random.RandomState(1234)

    x = T.fmatrix('x')   # input data
    y1 = T.fmatrix('y1')   # the target frames
    # y2 = T.fvector('y2')  # the target shift

    #debugging
    #import pdb; pdb.set_trace()
    
    # config.compute_test_value = 'warn'
    
    # x.tag.test_value = train_x.get_value()[:200]
    # y1.tag.test_value = train_y1.get_value()[:200]    

    
    rectifier = lambda x: Rectifier(x)
    #sigmoid = theano.tensor.nnet.sigmoid
    
     # construct model class
    frame_pred = MLP(rng=rng, input=x,
                     n_in=in_samples+win_width*39,
                     n_hiddens=[300,300],
                     hid_activations=[T.tanh, T.tanh],
                     # n_hiddens=[100],
                     # hid_activations=[],
                     n_out=out_samples,
                     out_activation=T.tanh)

    # shift_pred = MLP(rng=rng, input=x,
    #                  n_in=in_samples,
    #                  n_hiddens=[500],
    #                  hid_activations=[T.nnet.sigmoid],
    #                  n_out=1,
    #                  out_activation=T.nnet.sigmoid)
    
    
    frame_cost = MSE(y_pred=frame_pred.y_pred, y=y1) \
                 + L2_reg * frame_pred.L2_sqr


    total_training_costs, total_validation_costs = \
        frame_pred.train(y=y1, training_loss=frame_cost,
                         learning_rate=learning_rate, n_epochs=n_epochs,\
                         train_x=train_x, train_y=train_y1, valid_x=valid_x,\
                         valid_y=valid_y1, batch_size=batch_size)
    
    
    frame_pred.save_model(output_folder=output_folder)

    #save losses and parameters..
    
    np.save(os.path.join(output_folder,'training_MSE'),
            np.asarray(total_training_costs))
    np.save(os.path.join(output_folder,'validation_MSE'),
            np.asarray(total_validation_costs))

    
    
if __name__ == "__main__":
    
    # SAMPLE_PER_MS = 16
    # FRAME_LEN_MS = 15
    
    # frame_len = FRAME_LEN_MS * SAMPLE_PER_MS
    #build_data_sets(frame_len)
    # train_test_model(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
    #                  n_epochs=100, batch_size=1000, frame_len=frame_len,
    #                  n_hidden=500, output_folder='test_output')

    # win_width = 2
    # i = 0
    # for in_s in np.arange(2,6)*120:
    #     for out_s in np.arange(1,3)*40:
    #         for sh in np.arange(1,3)*20:
    #             i = i + 1
    #             # if i == 1:
    #             #     datasets = prepare_data(in_s, out_s, sh, win_width)
    #             #     np.save('out_%d/datasets'%i, datasets)
                    
    #             #os.mkdir('out_%d'%i)
    #             # train_model(datasets, learning_rate=0.01, L2_reg=0.0001,
    #             #             n_epochs=50, batch_size=100,
    #             #             in_samples = in_s, out_samples = out_s,
    #             #             shift = sh, output_folder='out_%d'%i)
    #             # best: 1 5 9 13
    #             val = np.load('out_%d/validation_MSE.npy'%i)[-1]
    #             tra = np.load('out_%d/training_MSE.npy'%i)[-1]
    #             print '%d: in_size: %d, out_size: %d, shift: %d, valid: %f, train: %f'%(i,in_s, out_s, sh, val, tra)

    # picking 13 :
    in_s = 600
    out_s = 40
    sh = 20
    win_width = 2
    
    #os.mkdir('better_results')
    datasets = prepare_data(in_s, out_s, sh, win_width)
    train_model(datasets, learning_rate=0.01, L2_reg=0.0001,
                n_epochs=100, batch_size=100,
                in_samples = in_s, out_samples = out_s,
                shift = sh, output_folder='better_results')
