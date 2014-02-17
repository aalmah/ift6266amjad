import numpy as np
import sys
import dataset.timit
import theano
import theano.tensor as T

dtype = 'float32'

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
    elems = np.hstack(wav_seqs)
    felems = elems.astype(dtype)
    
    std = np.std(felems)
    avg = np.mean(felems)
    norm_seqs = np.asarray([(seq.astype('float32')-avg)/std \
                            for seq in wav_seqs])
    assert norm_seqs.shape == wav_seqs.shape

    return norm_seqs
    

def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,dtype=dtype),borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,dtype=dtype),borrow=borrow)
   
    return shared_x, shared_y

    
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
    