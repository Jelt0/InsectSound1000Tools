
"""
Preprocessing Script Generating Audio-Samples out of a large 
recording file.

The calculation of the activity level is roughly based on the process
described in section II B in:
Z. Le-Qing, "Insect Sound Recognition Based on MFCC and PNN," 2011 
International Conference on Multimedia and Signal Processing, Guilin, 
China, 2011, pp. 42-46, doi: 10.1109/CMSP.2011.100

Input is a TDMS-files.
Output are wave-files 

Please cite:

Branding et al. (2023), Scientific Data, InsectSound1000 An Insect
Sound Dataset for Deep Learning based Acoustic Insect Recognition

                                                    Jelto Branding, 2023-10-13
"""

# puplic imports:
import librosa
import librosa.display
from scipy.signal import butter, sosfilt
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
from scipy import signal
import os
from nptdms import TdmsFile
from numba import njit
from scipy.io import wavfile

# local imports:
from fetchfiles import fetchfiles


def butter_highpass_filter(data, fs, cut_off, order):
    """ Highpass filter """
    # cal normalised cut off frequencies:
    normal_cutoff = cut_off / (0.5*fs)
    # Get the filter coefficients
    sos = butter(order, normal_cutoff, btype='high', analog=False,
                 output='sos')
    # Apply filter
    filtered = sosfilt(sos , data)
    return filtered


def butter_lowpass_filter(data, fs, cut_off, order):
    """ Lowpass filter """
    # cal normalised cut off frequencies:
    normal_cutoff = cut_off / (0.5*fs)
    # Get the filter coefficients
    sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    # Apply filter
    filtered = sosfilt(sos, data)
    return filtered


def est_energies(data, stepsize, framesize):
    """
    Slide a window over the data and calc the energy contained in each.
    """
    # Parameters:
    K = len(data)       # Number of sample points in recording
    L = stepsize        # stepsize (in sample points)
    N = framesize       # frame size (in sample points)

    # Generate idx-matrix for vectorized windowing:
    a = np.arange(0, N)
    b = np.arange(0, (K-N+1), L)
    idx = b[:, np.newaxis] + a

    # Calc energies by means of vector operations and
    # get energy of the frame by summing up every row:
    frame_energies = np.sum((data[idx] ** 2), axis=1)

    # Normalise by dividing through framesize:
    frame_energies = frame_energies / framesize

    return frame_energies


@njit
def get_activity(data, rate, stepsize, framesize, frame_energies, thres):
    """
    Mark sections of activity in recording.
    """

    # Parameters:
    K = len(data)  # Number of sample points in recording
    L = stepsize  # stepsize (in sample points)
    N = framesize  # frame size (in sample points)
    
    # empty array for activity:
    active = np.zeros((K, 1), dtype=np.bool_)

    # Mark sections of activity in record:
    for k, energy in enumerate(frame_energies):
        if energy > thres:
            i_start = k * L
            i_end = k * L + N
            active[i_start: i_end] = 1

    return active


def get_segments(data, data_ori, rate, active, min_len_activity, len_sample):
    """
    Extract valid samples from recording.
    """

    def add_sample_to_dict(data, data_ori, min_len, len_sample, ind_1, ind_2,
                           s, this_dict):
        '''now process this segment:'''

        # skip this sample if this activity is already captured by the
        # previous sample:
        if s > 1:
            # get end of last sample:
            s_2_previous = int(this_dict[s-1][-1, 0])
            if ind_2 < s_2_previous:
                print('This sample was skipped, because all its data is'
                      ' already contained in a previous sample.')
                return this_dict, s

        # Check if segment is longer than min_len and equal, shorter or
        # longer than the target sample length len_sample:
        # lenght of this sample:
        this_len = ind_2 - ind_1

        # if longer than min:
        if this_len > min_len:

            # equal
            if this_len == len_sample:
                s_1 = ind_1
                s_2 = ind_2

            else:
                # shorter than full len_sample:
                # Get center as middle between ind_2 and ind_1
                center = int((ind_2 + ind_1) / 2)
                # get start index:
                s_1 = int(center - (len_sample / 2))
                
                # if sample is on the very beginning, start at zero:
                if s_1 < 0:
                    s_1 = 0
                    # set end as after one sample length:
                    s_2 = int(len_sample)
                
                # else get the normal end:
                else:
                    s_2 = int(center + (len_sample / 2))
                    # but if at the very end, stop at end:
                    if s_2 > len(data):
                        s_2 = len(data)
                        # set beginning as one len_sample before
                        # file end:
                        s_1 = int(len(data) - len_sample)

                # Make sure samples don't overlap:
                # if sample start ist before last sample end, ...:
                if s > 1 and s_1 < s_2_previous:
                    s_1 = s_2_previous
                    s_2 = s_1 + len_sample
                    
                # if sample is at the very end, dicard it:
                if s_2 > len(data_ori)-2:
                    print('This sample was skipped, '
                          'because it is at the very end of a recoding.')
                    return this_dict, s

            # Add sample to output dict:
            indexes = np.arange(s_1, s_2)
            indexes = np.expand_dims(indexes, axis=1)
            sample_data = np.array(data_ori[s_1:s_2, :])
            this_dict[s] = np.append(indexes, sample_data, axis=1).astype('float32')
            # Count up samples:
            s += 1

        else:
            print('This sample was to short. It was discarded.')

        return this_dict, s


    # Prep:
    len_sample = int(len_sample * rate)      # get this in samples
    min_len = int(min_len_activity * rate)   # get this in samples
    dict = {}                                # Create dict to store results in
    s = 1                                    # Counting stuff

    # Now go through activity array:
    i = 0
    while i < len(active):
        # if active, start new activity:
        if active[i]:
            ind_1 = i
            # first, set end of activity equal to full length sample
            ind_2 = i + len_sample
            # if this is already the end of the recording, stop here:
            if ind_2 > len(active):
                ind_2 = len(active)-1
                
            # now starting form ind_2 look backwarts for the actual end:
            if not active[ind_2]:
                for j in range(len_sample):
                    if active[ind_2-j]:
                        ind_2 = ind_2-j
                        break

            # add sample to sample dict:
            dict, s = add_sample_to_dict(data, data_ori,
                                         min_len, len_sample, ind_1,
                                         ind_2, s, dict)

            # start looking for the next activity at the end of this one
            i = i + len_sample + 1

        else:
            i += 1

    return dict


def save_samples(samples_dict, rate, fullfilepath, ch):
    """
    Save samples to wave files.
    """
    # get path and filename:
    path, filename = os.path.split(fullfilepath)
    
    if len(samples_dict) > 0:
        for s, sample in samples_dict.items():

            # save all the files directly to the data set folder:
            out_path = target_dic + '/' + filename[:-5] \
                       + '_s%s_ch%s.wav' % (s, ch)
            # write wave file:
            wavfile.write(out_path, rate, data=sample[:, 1:])
        
        print('%s sample-files where saved.' % len(samples_dict))


def plot_all_the_things(data, rate, frame_energies, thres, samples_dict,
                        filename_wav_in, ch, show='off', safe='on'):
    """
    Name says it all. Can be used for debugging and finding good settings.
    """

    # Prep:
    time_array = (np.arange(0, len(data), 1)) / rate
    # streche frame_energies to length of data:
    t_data = np.arange(0, len(data), 1)
    t_frame_energies = np.linspace(0, len(data), len(frame_energies), dtype=int)
    frame_energies = np.interp(t_data, t_frame_energies, frame_energies)

    # if plot to long than 1 min, create multiple plots:
    len_s = int(len(data)/rate)
    if len_s > 61:
        for i in range(20, len_s, 50):
            # Cut out 60s snippeds of data, overlapping 10s:
            snipped_start = ((i-20)*rate)
            snipped_stop = ((i+50)*rate)
            # snipped end of the last one is the record end:
            if (i+50) > len_s:
                snipped_stop = (len_s * rate)

            time_snipped = time_array[snipped_start:snipped_stop]
            data_snipped = data[snipped_start:snipped_stop]
            frame_energies_snipped = frame_energies[snipped_start:snipped_stop]

            # Count throw filenames:
            folder, filename = os.path.split(filename_wav_in)
            plotname = filename[:-4] + '_ch%s_%ss_to_%ss' \
                       % (ch, int(snipped_start/rate), int(snipped_stop/rate))

            # call plot_function for this snipped:
            plot_results(time_snipped, data_snipped, frame_energies_snipped,
                         rate, thres, samples_dict, filename_wav_in, ch, show,
                         safe, plotname)

    else:
        plot_results(time_array, data, frame_energies, rate, thres,
                     samples_dict, filename_wav_in, ch, show, safe)


def plot_results(time_array, data, frame_energies, rate, thres,
                 samples_dict, filename_wav_in, ch, show, safe, plotname=None):
    # set figure size based on recording length:
    w, h = int((len(data)/rate)*0.5), 9
    # but keep a minimum size:
    if w < 7.5: w = 7.5
    # Create plot:
    fig, (axs1, axs2, axs3) = plt.subplots(3, 1, figsize=(w, h))

    # 1 -- Plot input signal after HP and mark sample beginning and end:
    axs1.plot(time_array, data, label='Signal', color='b')
    # Plot samples by marking beginning and end with red lines and
    # filling in light red in between:
    for samples in samples_dict.values():
        # plot only if the sample is within this time window:
        if (samples[0, 0]/rate) >= time_array[0] and \
                (samples[-1, 0]/rate) <= time_array[-1]:
            # draw a vertical line at sample start and end:
            axs1.axvline((samples[0, 0]/rate), color='r')
            axs1.axvline((samples[-1, 0]/rate), color='r')
            # fill light red in between
            axs1.axvspan((samples[0, 0]/rate), (samples[-1, 0]/rate), alpha=0.5,
                         color='red')
    axs1.margins(x=0)
    axs1.set_ylim([-0.1, 0.1])
    axs1.set_title('Filtered Signal')
    axs1.set_ylabel('Amplitude')

    # Plot signal energy vs its threshold:
    axs2.plot(time_array, frame_energies, label='Energy', color='g')
    axs2.hlines(thres, min(time_array), max(time_array), color='r')
    axs2.fill_between(time_array, thres, frame_energies,
                      where=frame_energies >= thres, facecolor='red')
    axs2.margins(x=0)
    axs2.set_ylim([0, thres*3])
    axs2.set_title('Energy and Energy Threshold')
    axs2.set_ylabel('Magnitude')

    # Adding the Spectrogram:
    # Create spectrogram :
    f, t, Sxx = signal.spectrogram(data, rate,
                                   window=('hann'),
                                   nperseg=4096,
                                   nfft=None,
                                   scaling='density',
                                   mode='psd')

    # Get f_1 and f_2 form f_start and f_end:
    f_max = rate / 2
    f_1 = 0
    f_2 = int(Sxx.shape[0] * (1500 / f_max))

    # Cut Sxx form f1 to f2 and convert to dB scale:
    Sxx_cut = 20 * np.log10(Sxx[f_1:f_2])

    # Plot image:
    img = axs3.imshow(Sxx_cut, origin='lower', aspect='auto')

    axs3.set_title('Linear-frequency power spectrogram')
    axs3.set_ylabel('Frequency [Hz/10]')
    axs3.label_outer()
    axs3.axes.get_xaxis().set_visible(False)

    fig.legend()

    if show == 'on':
        fig.show()

    elif show != 'off':
        raise ValueError('"show" must be either "on" or "off".'
                         ' Defaults to "on".')

    if safe == 'on':
        # set target dict path:
        target_dic = filename_wav_in[:-5] + '_samples'
        # create target dict if not existing:
        if not os.path.exists(target_dic):
            os.makedirs(target_dic)

        if plotname == None:
            folder, file = os.path.split(filename_wav_in)
            im_path = target_dic + '/' + file[:-5] + \
                      'ch%s_segmentation_result.png' % ch

        else:
            im_path = target_dic + '/' + plotname + '_segmentation_result.png'

        fig.savefig((im_path), dpi=300, bbox_inches='tight')
        plt.close('all')

    elif safe != 'off':
        raise ValueError('"safe" must be either "on" or "off".'
                         ' Defaults to "off".')


def stopwatch(function_name, last_timestamp):
    print('%s: %ss' % (function_name, round(time.time()-last_timestamp, 1)))
    new_timestamp = time.time()

    return new_timestamp


def pic_loudest_ch(data_multi_ch):
    # square signal to get rid of negativ values:
    data_multi_ch_power = data_multi_ch ** 2
    # get sum of every squared channel as a rough estimate of channel energy:
    ch_sums = data_multi_ch_power.sum(axis=0)
    # get index of loudest channel:
    ch = np.argmax(ch_sums)
    # separate loudest channel:
    loudest_ch = data_multi_ch[:, ch]

    return loudest_ch, ch


def read_tdms(filename):
    # create empty array to store data:
    data = []
    # read tdm-file:
    tdms_file = TdmsFile.read(filename)
    for group in tdms_file.groups():
        group_name = group.name
        for channel in group.channels():
            channel_name = channel.name
            # Access dictionary of properties:
            properties = channel.properties
            # Get sample rate form properties:
            sr = int(1 / properties['SampleDistance'])
            # Access numpy array of data for channel and append:
            data.append(channel[:])

    # python list to numpy array:
    data = np.array(data)
    # Make sure we use 32 float dtype:
    data = data.astype(np.float32)
    # Flip rows and columns:
    data = np.transpose(data)

    return sr, data


def resample_multi_ch(data_in, sr_in, sr_out):
    """resamples multi-channel audio data"""

    for ch in range(data_in.shape[1]):
        channel_data = librosa.resample(data_in[:, ch],
                                     orig_sr=sr_in,
                                     target_sr=sr_out,
                                     res_type='kaiser_best',
                                     fix=True,
                                     scale=False)
        if ch == 0:
            data_out = channel_data
            data_out = np.expand_dims(data_out, axis=1)
        else:
            data_out = np.column_stack((data_out, channel_data))

    return data_out


def main(filename_wav_in, stepsize_s, framesize_s, median_th, min_len_activity,
         len_sample):

    # Load wav file:
    orig_rate, data_multi_ch_orig = read_tdms(filename_wav_in)
    
    timestamp = stopwatch('read_tdms', start_time)

    # Downsample to 16 k Hz:
    rate = 16000
    data_multi_ch = resample_multi_ch(data_multi_ch_orig, orig_rate, rate)
    
    timestamp = stopwatch('resample_multi_ch', timestamp)

    # if its a 5 or 6 channel recording form my measurment mic array,
    # discard the last 2 channels and loop over the first 4:
    if data_multi_ch.shape[1] == 6:
        # delete channels containing outside recoding:
        data_multi_ch = np.delete(data_multi_ch, [4, 5], 1)
        # get index of loudest channel in array recording:
        data_ch, ch = pic_loudest_ch(data_multi_ch)
        # FOR DEBUGGING ONLY: cut to first 5 min :
        #data_ch = data_ch[:(rate*60*5)]

        timestamp = stopwatch('pic_loudest_ch', timestamp)

        # Apply a fast and good enough IIR lowpass filter to loudest ch:
        data_ch = butter_lowpass_filter(data_ch, rate, 1500, 4)
        # Apply a still fast and good enough IIR highpass filter to loudest ch:
        data_ch = butter_highpass_filter(data_ch, rate, 180, 30)
        
        timestamp = stopwatch('band_pass_filtering', timestamp)
      
        # Segmentation:
        # Estimate the energy contained in different parts of the
        # recording by moving a frame over the record and estimating the
        # energy contained by the signal in the frame:
        stepsize = int(stepsize_s * rate)
        framesize = int(framesize_s * rate)
        frame_energies = est_energies(data_ch, stepsize, framesize)

        timestamp = stopwatch('frame_energies', timestamp)

        # Calc energy_th:
        energy_th = np.median(frame_energies) * median_th

        # Find activity within the recording :
        active = get_activity(data_ch, rate, stepsize, framesize,
                                  frame_energies, energy_th)

        timestamp = stopwatch('get_activity', timestamp)

        # Extract valid samples:
        samples_dict = get_segments(data_ch, data_multi_ch, rate, active,
                                 min_len_activity, len_sample)

        timestamp = stopwatch('get_segments', timestamp)

        # Save samples:
        save_samples(samples_dict, rate, filename_wav_in, ch)

        timestamp = stopwatch('save_samples', timestamp)

        # Plot all the things:
        #plot_all_the_things(data, rate, frame_energies, energy_th,
        #                    samples_dict, filename_wav_in, ch, show='off',
        #                    safe='on')

        #timestamp = stopwatch('plot_all_the_things', timestamp)

        if len(samples_dict) == 0:
            print('No valid samples where found in %s' % filename_wav_in)


###############################################################################

# Start the clock first:
start_time = time.time()

# Energy threshold for detecting activity as times means value
energy_th = 1.60

# Set minimum activity length and sample length [s]:
min_len_activity = 1
len_sample = 2.5

# Set stepsize and framesize of energy estimation:
times_the_paper_value = 20
stepsize_s = 0.0032 * times_the_paper_value
framesize_s = 0.01024 * times_the_paper_value

# make sure min_len_activity is > framesize_s:
if min_len_activity <= framesize_s:
    min_len_activity = framesize_s * 1.5

# set path to input directory, conatining mutliple recording folders
# named by date [yyyymmdd]:
input_dic = 'y:/YourRecordingFolder/'

# set a path to a target directory to store the samples in:
target_dic = 'y:/DataSets/InsectSound1000v2'

# go through all the folders in the directorys:
for recording_date in [20231004,
                       20231005
                       #...
                       ]:

    path = input_dic + str(recording_date)

    # Save terminal output to file:
    sys.stdout = open(path + '/SampleExtractor_log_2500ms.txt', 'w')

    # print path just like a headline:
    print('Now processing ' + path)

    # get input files:
    files = fetchfiles(path, '.tdms', '.tdms_index')

    for file in files:
        # Do the thing !!!:
        try:
            main(file, stepsize_s, framesize_s, energy_th,
                 min_len_activity, len_sample)
        except:
            print('Could not process ' + file)

    # Close log file:
    sys.stdout.close()
