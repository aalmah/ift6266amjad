from dataset.timit import TIMIT
import os.path
import os
import sys
import numpy as np
import theano
import theano.tensor as T
from experiments.mdn import NLL, MDN
from experiments.exp2.datasets_builder \
     import build_data_sets, build_one_user_data, build_aa_dataset

import scipy.io.wavfile as wave
from experiments import utils

class SpeechSynthesizer:
    """Multi-Layer Perceptron based speech synthesizer.
    """

    def __init__(self, in_samples=240, out_samples = 1, shift=1,
                 win_width=2, output_folder='output_folder'):
        
        self.frame_pred = None
        self.shift_pred = None
        self.in_samples = in_samples
        self.out_samples = out_samples
        self.shift = shift
        self.win_width = win_width

        self.input_size = in_samples# + win_width*39
        self.output_size = out_samples
        self.output_folder = output_folder

        # creating wrapper object for TIMIT dataset
        save_stdout = sys.stdout
        sys.stdout = open('timit.log', 'w')
        self.timit = TIMIT()
        self.timit.load("train")
        self.timit.load("valid")
        sys.stdout = save_stdout

        
    def set_models(self, frame_pred=None, shift_pred=None):
        """Loads models previously trained"""
        
        self.frame_pred = frame_pred
        self.shift_pred = shift_pred
        

    def prepare_data(self):
        """This function prepares training/validation datasets used for training
        the models """
        
        print 'loading data...'

        # training set configurations..
        # TODO: move them as parameters to the function, but for now we'll fix
        # values for current experiments


        # usr_id = 0
        # shuffle = True

        # # each training example has 'in_sample' inputs and 'out_samples' output
        # # and examples are shifted by 'shift'
        # train_x, train_y1, train_y2, valid_x, valid_y1, valid_y2 = \
        #         build_one_user_data(self.timit, self.in_samples,
        #                             self.out_samples, self.shift,
        #                             self.win_width, shuffle, usr_id)

        # n_spkr = 10
        # n_utts = 10
        # shuffle = True

        # # each training example has 'in_sample' inputs and 'out_samples' output
        # # and examples are shifted by 'shift'
        # train_x, train_y1, train_y2 = \
        #     build_data_sets(self.timit, 'train', n_spkr, n_utts,
        #                     self.in_samples, self.out_samples,
        #                     self.shift, self.win_width, shuffle)

        # n_spkr = 1
        # n_utts = 1
        # shuffle = False
        # valid_x, valid_y1, valid_y2 = \
        #     build_data_sets(self.timit, 'valid', n_spkr, n_utts,
        #                     self.in_samples, self.out_samples,
        #                     self.shift, self.win_width, shuffle)

        train_x, train_y, valid_x, valid_y = build_aa_dataset(self.in_samples,
                                                              self.out_samples,
                                                              self.shift)
        
        
        #return (train_x, train_y1, train_y2), (valid_x, valid_y1, valid_y2)
        return (train_x, train_y, None), (valid_x, valid_y, None)



    def train_models(self, learning_rate=0.001, L2_reg=0.0001, n_epochs=100,
                     batch_size=128, n_hiddens=[300,300],
                     hid_activations=[T.tanh, T.tanh],
                     n_components=5):
        
        """This function trains MLP models used by this speech synthesizer"""

        datasets = self.prepare_data()
        
        train_x, train_y1, train_y2 = datasets[0]
        valid_x, valid_y1, valid_y2 = datasets[1]

        print 'building the model...'

        rng = np.random.RandomState(1234)

        x = T.fmatrix('x')   # input data (minibatch_size, input_size)
        y1 = T.fmatrix('y1')   # the target frames (minibatch_size, output_size)
        # y2 = T.vector('y2')  # the target shifts
        
        # construct model class
        self.frame_pred = MDN(rng=rng, input=x,
                              n_in=self.input_size,
                              n_hiddens=n_hiddens,
                              hid_activations=hid_activations,
                              n_out=self.output_size,
                              #out_activation=T.tanh,
                              out_activation=None,
                              n_components=n_components)
        
        # commented out for now as it's not used
        
        # self.shift_pred = MLP(rng=rng, input=x,
        #                       n_in=in_samples+win_width*39,
        #                       n_hiddens=[300,300],
        #                       hid_activations=[T.tanh, T.tanh],
        #                       n_out=1,
        #                       out_activation=T.nnet.sigmoid)

        frame_cost = NLL(mu = self.frame_pred.outputLayer.mu,
                         sigma = self.frame_pred.outputLayer.sigma,
                         mixing = self.frame_pred.outputLayer.mixing,
                         y = y1) #+ L2_reg * self.frame_pred.L2_sqr
        

        total_training_costs, total_validation_costs, total_validation_MSE = \
            self.frame_pred.train(y=y1, training_loss=frame_cost,
                                  learning_rate=learning_rate,
                                  n_epochs=n_epochs,\
                                  train_x=train_x, train_y=train_y1, \
                                  valid_x=valid_x, valid_y=valid_y1, \
                                  batch_size=batch_size)
    
    
        self.frame_pred.save_model(output_folder=self.output_folder)

        #save losses and parameters..
    
        np.save(os.path.join(self.output_folder,'training_NLL'),
                np.asarray(total_training_costs))
        np.save(os.path.join(self.output_folder,'validation_NLL'),
                np.asarray(total_validation_costs))
        np.save(os.path.join(self.output_folder,'validation_MSE'),
                np.asarray(total_validation_MSE))


        
    def generate_speech1(self):
        """This function generates speech signal (or a supposed one!) for a
        sequence of phonemes, this function does some cheating though, as
        uses the alignment information from the test sequence."""
        
        # n_spkr = 1
        # n_utts = 1
        # shuffle = False
                
        # valid_x, valid_y1, valid_y2 = \
        #     build_data_sets(self.timit, 'valid', n_spkr, n_utts,
        #                     self.in_samples, self.out_samples,
        #                     self.shift, self.win_width, shuffle)

        # train_x, train_y, valid_x_t, valid_y1_t = \
        #         build_aa_dataset(self.in_samples,
        #                          self.out_samples,
        #                          self.shift,n_valid=1)

        
        # # # they're returned as shared variables, and I only need here the values
        # valid_x = valid_x_t.get_value()
        # valid_y1 = valid_y1_t.get_value()

        # symbolic previus output, which is used as input for the next frame
        # predictor
        prev_y = T.fmatrix('x')

        # Theano function the computes next frame given previous frames
        get_next_frame = theano.function(inputs=[prev_y],
                            outputs=self.frame_pred.samples(),
                            givens={self.frame_pred.input: prev_y})

        # initialize first frame to normal noise
        prev_frame = np.random.normal(loc=0.0, scale=1.0e-3,
                                      size=(self.in_samples,))

        output_signal = np.zeros(1500*self.shift, dtype='float32')
        
        current_frame = np.zeros((self.input_size), dtype='float32')
        
        for i in range(1400):
            
            #current_win = valid_x[i][self.in_samples:]
            current_frame[:self.in_samples] = prev_frame
            #current_frame[self.in_samples:] = current_win

            predicted_frame = \
                    get_next_frame(current_frame.reshape(1,self.input_size))

            output_signal[i*self.shift:i*self.shift+self.shift] =\
                             predicted_frame[0][:self.shift]

            prev_frame = np.roll(prev_frame, -self.shift)
            prev_frame[-self.shift:] = predicted_frame[0][:self.shift]


        
        output_signal = output_signal * utils.MAX_SIGNAL_AMP
        
        np.save(os.path.join(self.output_folder,'generated_speech'),
                output_signal)
        wave.write(os.path.join(self.output_folder,'generated_speech.wav'),
                   16000, output_signal.astype('int16'))
        
        print 'Done with generation test!'

        
    
if __name__ == "__main__":
    folder = 'results'
    
    if "train" in sys.argv:
        synthesizer = SpeechSynthesizer(output_folder=folder)
        synthesizer.train_models()
            
    elif "test" in sys.argv:
        frame_pred = MDN.load_model(output_folder=folder)
        synthesizer = SpeechSynthesizer(output_folder=folder)
        synthesizer.set_models(frame_pred=frame_pred)
        synthesizer.generate_speech1()
    else:
        print 'specify an argument!'