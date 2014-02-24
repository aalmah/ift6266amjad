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
from experiments.nn import NetworkLayer,MSE


        
def analyze_phones():
    """analyzing phones in a speech signal"""
    
    print 'loading data...'

    save_stdout = sys.stdout
    sys.stdout = open('timit.log', 'w')

    # creating wrapper object for TIMIT dataset
    dataset = TIMIT()
    dataset.load("train")
    
    sys.stdout = save_stdout


    # for i in range(1,dataset.train_phn.shape[0]):
    #     prev = dataset.train_phn[i-1]
    #     curr = dataset.train_phn[i]
    #     if curr[0] != prev[1] and curr[0] != 0:
    #         print 'gap between %d and %d'%(i-1,i)
    #         if curr[0] > prev[1]:
    #             print 'this gap is within the sequence'
    #         else:
    #             print 'gap beginning of sequence at sample %d next phone is %d ending at %d'\
    #             %(curr[0],curr[2],curr[1])
                
    for i in range(1,dataset.train_phn.shape[0]):
        curr = dataset.train_phn[i]
        if curr[0] == 0:
            if curr[2] != 4:
                print 'problem'
    print 'start phone is %s'%dataset.phonemes[4]
    print len(dataset.phonemes)
    print dataset.phonemes
        
            

    # spkr_start = 3
    # n_spkr = 3
    # spkr_indx = spkr_start*10

    # for i in range(n_spkr):
    #     print 'analyzing data for speaker %s'\
    #     % dataset.spkrid[dataset.train_spkr[spkr_indx+i*10]]
    
    # wav_seqs = dataset.train_raw_wav[spkr_indx:spkr_indx+10*n_spkr]
    # seqs_to_phns = dataset.train_seq_to_phn[spkr_indx:spkr_indx+10*n_spkr]
    # phones = np.asarray(dataset.phonemes)
    # counter = {}
    # for idx, seq in enumerate(seqs_to_phns[:10*n_spkr-1]):
    #     phn_subseq = dataset.train_phn[seq[0]:seq[1]]
    #     for phn_info in phn_subseq:
    #         phn = phones[phn_info[2]]
    #         if phn in counter:
    #             counter[phn] = counter[phn] + phn_info[1] - phn_info[0]
    #         else:
    #             counter[phn] = phn_info[1] - phn_info[0]

    #     print 'phones in seq %d:'%idx
    #     for phn in phones[phn_subseq[:,2]]:
    #         print phn,
    #     print

    # print 'test utturance...'
    # for idx, seq in enumerate(seqs_to_phns[10*n_spkr-1:]):
    #     phn_subseq = dataset.train_phn[seq[0]:seq[1]]
    #     for phn_info in phn_subseq:
    #         phn = phones[phn_info[2]]
    #         if phn in counter:
    #             counter[phn] = counter[phn] + phn_info[1] - phn_info[0]
    #         else:
    #             print 'new phone:',phn
    #             counter[phn] = phn_info[1] - phn_info[0]
                
    #     print 'phones in seq %d:'%idx
    #     for phn in phones[phn_subseq[:,2]]:
    #         print phn,
    #     print


    # phn_count = sorted([(phn,c) for phn,c in counter.iteritems()],
    #                    key=lambda tup: tup[1])
    # print 'phone : samples'
    # for phn,c in phn_count:
    #     print "%s : %d"%(phn,c)



    
if __name__ == "__main__":
    
    SAMPLE_PER_MS = 16
    FRAME_LEN_MS = 15
    
    frame_len = FRAME_LEN_MS * SAMPLE_PER_MS

    analyze_phones()
    # train_test_model(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001,
    #                  n_epochs=100, batch_size=1000, frame_len=frame_len,
    #                  n_hidden=500, output_folder='test_output')