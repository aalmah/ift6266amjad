import dataset.timit
import utils

import sys
from scikits.talkbox import segment_axis
import numpy as np

SAMPLE_PER_MS = 16

class NextSamplePredictor:

    def __init__(self):

        save_stdout = sys.stdout
        sys.stdout = open('timit.log', 'w')
        
        self.dataset = dataset.timit.TIMIT()
        self.dataset.load("train")
        self.dataset.load("valid")
        self.dataset.load("test")
        sys.stdout = save_stdout

        

    def build_data_sets(self,frame_ms=15):
        """build data sets for training/validating/testing the models"""

        print 'loading data...'
        frame_len = frame_ms * SAMPLE_PER_MS
        overlap = frame_len - 1

        wav_seqs = self.dataset.train_raw_wav[0:10]
        norm_seqs = utils.normalize(wav_seqs)
        
        # Segment into frames
        samples = map(lambda seq: segment_axis(seq, frame_len, overlap),
                      norm_seqs)
         # stack all data in one matrix, each row is a frame
        data = np.vstack(samples)
        # shuffle the frames so we can assume data is IID
        np.random.seed(123)
        data = np.random.permutation(data)
        
        # print len(samples)
        # count = 0
        # for i in range(10):
        #     print samples[i].shape
        #     count += samples[i].shape[0]
        # print count
        # print data.shape

        # take 10% for test, 10% for valid, and 80% for training
        chunk = data.shape[0] / 10
        # now split data to x and y for train, valid, and test
        train_x = data[:8*chunk,:-1]
        train_y = data[:8*chunk,-1]
        valid_x = data[8*chunk:9*chunk,:-1]
        valid_y = data[8*chunk:9*chunk,-1]
        test_x = data[9*chunk:,:-1]
        test_y = data[9*chunk:,-1]

        
        # print train_x.shape,train_y.shape,valid_x.shape,valid_y.shape
        # print test_x.shape,test_y.shape
        print 'Done'
        
        return  utils.shared_dataset((train_x,train_y)),\
                utils.shared_dataset((valid_x,valid_y)),\
                utils.shared_dataset((test_x,test_y))
    
if __name__ == "__main__":
    
    predictor = NextSamplePredictor()
    predictor.build_data_sets(frame_ms=15)
    