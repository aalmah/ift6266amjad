import numpy as np
import sys
import dataset.timit
import theano
import theano.tensor as T

dtype = 'float32'

MAX_SIGNAL_AMP = 18102

def data_avg_std(dataset):
    elems = np.hstack((np.hstack(dataset.train_raw_wav),
                       np.hstack(dataset.valid_raw_wav),
                       np.hstack(dataset.test_raw_wav)))
                      
    felems = elems.astype(dtype)
                      
    std = np.std(felems)
    avg = np.mean(felems)
    print 'data std:',std
    print 'data avg:',avg

    return (avg,std)

def normalize(wav_seqs):
    
    max_e = MAX_SIGNAL_AMP
    norm_seqs = np.asarray([seq.astype('float32')/max_e for seq in wav_seqs])
    #make sure all values in [-1,1]
    for seq in norm_seqs:
        assert np.all(seq < 1.001) and np.all( seq > -1.001)
        
    return norm_seqs
    

def shared_dataset(data, borrow=True):
    """ Function that loads the dataset into shared variables
    """
    shared_data = theano.shared(np.asarray(data,dtype=dtype),borrow=borrow)
   
    return shared_data

def shared_dataset_xy(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    """
    data_x , data_y = data_xy
    shared_data_x = theano.shared(np.asarray(data_x,dtype=dtype),borrow=borrow)
    shared_data_y = theano.shared(np.asarray(data_y,dtype=dtype),borrow=borrow)
   
    return shared_data_x, shared_data_y

    
if __name__ == "__main__":
    save_stdout = sys.stdout
    sys.stdout = open('timit.log', 'w')
    
    dataset = dataset.timit.TIMIT()
    dataset.load("train")
    dataset.load("valid")
    dataset.load("test")
    sys.stdout = save_stdout

    wav_seqs = dataset.train_raw_wav[0:10]
    normalize(wav_seqs)
    