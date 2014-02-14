import dataset.timit
import sys

SAMPLE_PER_MS = 16

class NextSamplePredictor:

    def __init__(self, frame_ms):

        save_stdout = sys.stdout
        sys.stdout = open('timit.log', 'w')
        
        self.dataset = dataset.timit.TIMIT()
        self.dataset.load("train")
        self.dataset.load("valid")
        self.dataset.load("test")
        sys.stdout = save_stdout

        self.frame_ms = frame_ms
        

    def build_training_set(self):
        """build training set for training the models"""

        frame_len = self.frame_ms * SAMPLE_PER_MS
        
        seq = dataset.get_raw_seq(subset='train',seq_id=0,
                                  frame_length=frame_len,
                                  overlap=frame_len-1)
        

        [wav_seq, phn_seq, end_phn, wrd_seq, end_wrd, spkr_info] = seq

        # for item in seq:
        #     print item.shape
        print 'in one sequence we have..'
        print '# frames:',wav_seq.shape[0]
        print 'one sample looks like:', wav_seq[0][0], type(wav_seq[0][0])
        print 'first n phones:',phn_seq[0:10]
        print 'end_phn flag:',end_phn[0:10]
        print 'speaker info:'
        print spkr_info
    
if __name__ == "__main__":

    save_stdout = sys.stdout
    sys.stdout = open('timit.log', 'w')
    
    dataset = dataset.timit.TIMIT()
    dataset.load("train")

    sys.stdout = save_stdout


    

