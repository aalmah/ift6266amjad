from dataset.timit import TIMIT
from experiments import utils
import sys
from scikits.talkbox import segment_axis
import numpy as np

        
def build_data_sets(dataset, subset, n_spkr, n_utts,
                    in_samples, out_samples, shift,
                    win_width, shuffle):
    """builds data sets for training/validating/testing the models"""

    print "building %s dataset..."%subset
    
    wav_seqs = dataset.__dict__[subset+"_raw_wav"][0:n_utts*n_spkr]
    norm_seqs = utils.normalize(wav_seqs)

    seqs_to_phns = dataset.__dict__[subset+"_seq_to_phn"][0:n_utts*n_spkr]
    frame_len = in_samples + out_samples
    overlap = frame_len - shift
    
    samples = []
    seqs_phn_info = []
    seqs_phn_shift = []

        
    # CAUTION!: I am using here reduced phone set
    # we can also try using the full set but we must store phn+1
    # because 0 no more refers to 'h#' (no speech)

    for ind in range(len(norm_seqs)):
        #import pdb; pdb.set_trace()
        wav_seq = norm_seqs[ind]
        phn_seq = seqs_to_phns[ind]
        phn_start_end = dataset.__dict__[subset+"_phn"][phn_seq[0]:phn_seq[1]]

        # create a matrix with consecutive windows
        # phones are padded by h#, because each window will be shifted once
        # the first phone samples has passed

        phones = np.append(phn_start_end[:,2].astype('int16'),
                           np.zeros((1,),dtype='int16'))
        # phones = np.append(phn_start_end[:,2],
        #                    np.zeros((1,)))

        phn_windows = segment_axis(phones, win_width, win_width-1)

        # array that has endings of each phone
        phn_ends = phn_start_end[:,1]
        # extend the last phone till the end, this is not wrong as long as the
        # last phone is no speech phone (h#)
        phn_ends[-1] = wav_seq.shape[0]-1

        # create a mapping from each sample to phn_window
        phn_win_shift = np.zeros_like(wav_seq,dtype='int16')
        phn_win_shift[phn_ends] = 1
        phn_win = phn_win_shift.cumsum(dtype='int16')
        # minor correction!
        phn_win[-1] = phn_win[-2]

        # Segment samples into frames
        samples.append(segment_axis(wav_seq, frame_len, overlap))

        # for phones we care only about one value to mark the start of a new window.
        # the start of a phone window in a frame is when all samples of previous
        # phone hav passed, so we use 'min' function to choose the current phone
        # of the frame
        phn_frames = segment_axis(phn_win, frame_len, overlap).min(axis=1)
        # replace the window index with the window itself
        win_frames = phn_windows[phn_frames]
        seqs_phn_info.append(win_frames)

        #import pdb; pdb.set_trace()
        # create a window shift for each frame
        shift_frames_aux = np.roll(phn_frames,1)
        shift_frames_aux[0] = 0
        shift_frames = phn_frames - shift_frames_aux
        # to mark the ending of the sequence - countering the first correction!
        shift_frames[-1] = 1
        seqs_phn_shift.append(shift_frames)
        #import pdb; pdb.set_trace()
    
        
    #import pdb; pdb.set_trace()
    # stack all data in one matrix, each row is a frame
    samples_data = np.vstack(samples[:])
    phn_data = np.vstack(seqs_phn_info[:])
    shift_data = np.hstack(seqs_phn_shift[:])

    
    #convert phone data to one-hot
    from pylearn2.format.target_format import OneHotFormatter
    fmt = OneHotFormatter(max_labels=39, dtype='float32')
    
    phn_data = fmt.format(phn_data)
    phn_data = phn_data.reshape(phn_data.shape[0],
                                phn_data.shape[1]*phn_data.shape[2])
    
    full_data = np.hstack([samples_data[:,:in_samples], phn_data, #input
                           samples_data[:,in_samples:], #out1
                           shift_data.reshape(shift_data.shape[0],1)]) #out2
    
    # TODO: do this more efficiently
    # shuffle the frames so we can assume data is IID
    if shuffle:
        np.random.seed(123)
        full_data = np.random.permutation(full_data)

    
    data_x = full_data[:,:in_samples+win_width*39]
    data_y1 = full_data[:,in_samples+win_width*39:-1]
    data_y2 = full_data[:,-1]
    
        
    print 'Done'
    print 'There are %d examples in %s set'%(data_x.shape[0],subset)

    return utils.shared_dataset(data_x), \
           utils.shared_dataset(data_y1),\
           utils.shared_dataset(data_y2)
    
    
if __name__ == "__main__":
    
    
    print 'loading data...'

    save_stdout = sys.stdout
    sys.stdout = open('timit.log', 'w')

    # creating wrapper object for TIMIT dataset
    dataset = TIMIT()
    dataset.load("train")
    dataset.load("valid")
    
    sys.stdout = save_stdout

    
    in_samples = 240
    out_samples = 50
    shift = 10
    win_width = 2
    n_spkr = 1
    n_utts = 10
    shuffle = True
    # each training example has 'in_sample' inputs and 'out_samples' output
    # and examples are shifted by 'shift'
    build_data_sets(dataset, 'train', n_spkr, n_utts,
                    in_samples, out_samples, shift,
                    win_width,shuffle)

    