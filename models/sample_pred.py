import dataset.timit
import sys

SAMPLE_PER_MS = 16
FRAME_MS = 15

def prepare_input(dataset):
    frame_len = FRAME_MS * SAMPLE_PER_MS
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

    prepare_input(dataset)
    

