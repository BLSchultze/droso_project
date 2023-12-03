""" Module with functions to analyze audio data of behavioural experiments with optogenetics
This module contains the following functions:
    - mean_center:          center the data to the mean, normalize if requested
    - normalize:            normalize the data
    - rolling_rms:          calculate the root-mean-square (RMS) of a signal
    - analyze_light_stim:   retrieve on- and offsets as well as stimulation intensities
    - signal_power:         calculate the signal power in given sections based on the RMS
    - frequency_anlysis:    analyze a signal in the frequency domain using frequency spectra or Fourier transforms of given sections

Author:         Bjarne Schultze
Last modified:  01.12.2023
"""

import numpy as np
import pandas as pd
from itertools import accumulate, groupby
from scipy import signal
from scipy.fft import rfft, rfftfreq



def mean_center(signal, norm=False):
    """ Mean-center the given signal (i.e. subtract mean to get zero mean data)
    Optionally, normalize to absolute maximum value
    
    Args:
        signal (n*m array): with n data points in m channels/columns

    Returns:
        n*m array: mean-centered data (normalized if requested)
    """
    if norm:
        m_signal = signal - np.mean(signal, axis=0)
        return (m_signal / np.max(np.abs(m_signal), axis=0))
    else:
        return signal - np.mean(signal, axis=0)


def normalize(signal):
    """ Normalize the given signal to its absolute maximum value
    
    Args:
        signal (n*m array): with n data points in m channels/columns

    Returns:
        n*m array: normalized signal
    
    """
    return (signal / np.max(np.abs(signal), axis=0))



def rolling_rms(signal, window_len):
    """ Calculate the root mean square (RMS) of a signal at all time points
    The input signal is mean-centered and normalized to the absolute maximum before calculation of the RMS.

    Args:
        signal (n*m array): with n data points in m channels/columns
        window_len (int): the length (index) of the window for averaging the squared signal

    Returns:
        n*m np.array: RMS signal
    """

    # Normalize and mean-center the input signal 
    signal = mean_center(signal, norm=True)
    # Pad signal with window_len zeros
    pad_vec = np.zeros((window_len,len(signal[0,:])))
    padded_signal = pd.DataFrame(np.concatenate((pad_vec,signal,pad_vec)))
    # Calculate the root-mean-square of each window of length window_len
    rms_signal = ((padded_signal**2).rolling(window_len).mean())**0.5
    
    # Return the RMS of the signal with the padding removed and the window centered (1/2 window_len data points get lost)
    return np.array(rms_signal[round(1.5*window_len):-window_len])



def analyze_light_stim(light_chans):
    """ Analyze the light stimulus to get on- and offsets as well as stimulation intensities

    Args:
        light_chans (n*2 array): channels with the light stimulus, n data points in two channels (0th channel: green light, 1st channel red light)
    
    Returns:
        n*1 array: light stimulus values at all n sample points
        string: stating the stimulation mode, either 'activation' or 'inactivation'
        list (m items): indices of all m stimulus onsets
        list (m items): indices of all m stimulus offsets
        list (m items): stimulation voltage for all m stimuli (same order as onsets and offsets)
    """
    # Find the channel with the light stimulus (0 or 1 depending on whether red or green light)
    light_stim_index = np.argmax(np.max(light_chans, axis=0))
    # Save the stimulation mode for later use
    if light_stim_index == 1:
        stim_mode = 'activation'
    elif light_stim_index == 0:
        stim_mode = 'inactivation'
    # Extract the light stimulus values
    light = light_chans[:, light_stim_index]

    # Identify time points where stimulation was applied
    stim_ind_log = light > 0 + 0.05
    # Get on- and offsets for all stimulus pulses
    stim_onoff = [0] + list(accumulate(sum(1 for _ in g) for _,g in groupby(stim_ind_log)))
    # Divide into onsets and offsets
    stim_on = [ i for i in stim_onoff[:-1] if stim_ind_log[i] ]
    stim_off = [ i for i in stim_onoff[1:-1] if not stim_ind_log[i] ]

    # Get all stimulation values (in order of appearance) and the set of unique voltages
    stim_volt = np.array([ np.mean(light[i:j]).round(2) if max(light[i:j]) >= 0.5 else np.mean(light[i:j]).round(3) for i,j in zip(stim_on, stim_off) ])
    
    return light, stim_mode, stim_on, stim_off, stim_volt



def signal_power(signal, time_vec, stim_on, stim_off):
    """ Calculate the power of the signal during stimulation as the area under the RMS curve
    The function uses the composite trapezoidal rule implemented in numpy as numpy.trapz() to obtain the area under the signal curve.

    Args:
        signal (n*m array): root mean square (RMS) of the original signal (envelop estimation) with n sample points in m channels/columns
        time_vec (n*1 array): time values corresponding to the signal
        stim_on (list): indices of all stimulus onsets
        stim_off (list): indices of all stimulus offsets

    Returns:
        m*1 np.array: power of the signal (area under the curve) during stimulation
        m*1 np.array: power before the stimulation (in a time window of the same length as the stimulation)
        m*1 np.array: power after the stimulation (in a time window of the same length as the stimulation)
    """
    # Define lists to collect the calculated power
    power = []
    power_pre = []
    power_post = []
    # Calculate the stimulus length
    stim_len = stim_off[0] - stim_on[0]

    # Iterate over all stimuli
    for start, stop in zip(stim_on, stim_off):
        # Calculate the power of the signal as the area under the RMS curve during stimulation 
        power.append(np.trapz(signal[start:stop,:], x=time_vec[start:stop], axis=0))
        # Power before and after stimulation
        power_pre.append(np.trapz(signal[start-stim_len:start,:], x=time_vec[start-stim_len:start], axis=0))
        power_post.append(np.trapz(signal[stop:stop+stim_len,:], x=time_vec[stop:stop+stim_len], axis=0))

    # Transform to numpy array
    return np.array(power), np.array(power_pre), np.array(power_post)



def freqency_analysis(data, fs, stim_on, stim_off, stim_volt, continuous=True, *, nperseg=700, overlap=0, reduce=(0,1000)):
    ''' Analyze the frequency components of a signal

    Args:
        data (n*m array): signal with n data points in m channels/columns
        fs (float): sampling rate
        stim_on (list): indices of all stimulus onsets
        stim_off (list): indices of all stimulus offsets
        stim_volts (list): stimulation intensities [volt] corresponding to the stimuli in 'stim_on' and 'stim_off'
        continuous (bool): states which type of analysis should be performed. True: signal from before to after stimulation is analyzed via freqency spectra
                           with 'nperseg' long windows using an overlap of 'overlap. default: True
        nperseg (int): number of data points that should be used when calculating the frequency spectrum using consecuative short time Fourier transforms. 
                       A higher number reduces the resolution along the time axis but increases resolution along the frequency axis. default: 700
        overlap (int): number of data point that two consecuative windows of 'nperseg' length should overlap when calculating the frequency spectra. default: 0
        reduce (tuple of 2 floats): range to which the freqencies should be reduced, (lower border, upper border). default: (0,1000)

    Returns:
        if continuous:
            spec_cont_avg (np.array, shape: stimuli*freqs*channels*time_points): the continuous spectrogram averaged for stimuli of the same intensity
            freqs (np.array): frequency axis for the spectrogram (2nd dimension in stim_cont_avg)
            time_points (np.array): time axis for the spectrogram (4th dimension in stim_cont_avg)
        else:
            fft_stim_avg (np.array, shape: stimuli*freqs*channels): signal during stimulation in the frequency domain
            fft_pre_avg (np.array, shape: stimuli*freqs*channels): signal before stimulation (same length as stimulation) in the frequency domain
            fft_post_avg (np.array, shape: stimuli*freqs*channels): signal after stimulation (same length as stimulation) in the frequency domain
            fft_freqs (np.array): sample frequencies of the Fourier transform (2nd dimension in the signal matrices)
    '''

    # Mean-center and normalize the input data
    data = mean_center(data, norm=True)

    # Store the length of stimulation (equal length assumed)
    stim_len = stim_off[0] - stim_on[0]
    # Get a array of the unique stimulus values
    stim_volt_u = np.unique(stim_volt)

    # If the data has only one dimension, add a second dimension to enable matrix-like indexing
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=-1)

    # If continuous analysis is requested, frequency spectra are calculated form before to after the stimulation
    if continuous:
        # Initialize list to collect results
        spec_cont = []

        # Iterate over all stimuli
        for start, stop in zip(stim_on, stim_off):
            # Compute the spectra for all three channels (shape: freqs*channels*time_points)
            freqs_tmp, time_points_tmp, spectrogrm_tmp = signal.spectrogram(mean_center(data[start-stim_len:stop+stim_len]), fs, axis=0, nperseg=nperseg, 
                                                                            noverlap=overlap, scaling='spectrum')

            # Discard frequencies according to the 'reduce' argument, default: keep frequencies between 0 and 1000 Hz
            freq_sel = np.logical_and(freqs_tmp >= reduce[0], freqs_tmp <= reduce[1])
            # Store the reduced spectrogram
            spec_cont.append(spectrogrm_tmp[freq_sel, :, :])

        # Average the spectra for all stimuli of the same intensity
        spec_cont = np.array(spec_cont)         # shape: stimuli*freqs*channels*time_points
        spec_cont_avg = np.array( [ np.mean(spec_cont[stim_volt == uval,:,:,:], axis=0) for uval in stim_volt_u ] )

        # Store frequencies and time points (the same for all spectrograms, if equal stimuli length)
        freqs = freqs_tmp[freq_sel]
        time_points = time_points_tmp

        # Return the results
        return spec_cont_avg, freqs, time_points

    # Other than in the continuous analysis, here the signal before, during, and after stimulation is transformed to the frequency domain
    else:
        # Initialize lists to collect the results
        fft_stim = []
        fft_pre = []
        fft_post = []

        # Get the frequencies for the Fourier transforms
        fft_freqs = rfftfreq(stim_len, 1.0/fs)
        freq_sel = np.logical_and(fft_freqs <= reduce[1], fft_freqs >= reduce[0])
        fft_freqs = fft_freqs[freq_sel]

        # Iterate over all stimuli
        for start, stop in zip(stim_on, stim_off):
            # Calculate the Discrete Fourier Transform using the FFT for rational input (rfft)
            fft_stim_tmp = np.abs(rfft(mean_center(data[start:stop,:]), axis=0)) * 2.0/stim_len
            fft_pre_tmp = np.abs(rfft(mean_center(data[start-stim_len:start,:]), axis=0)) * 2.0/stim_len
            fft_post_tmp = np.abs(rfft(mean_center(data[stop:stop+stim_len,:]), axis=0)) * 2.0/stim_len
            # Store the results  
            fft_stim.append(fft_stim_tmp[freq_sel,:])
            fft_pre.append(fft_pre_tmp[freq_sel,:])
            fft_post.append(fft_post_tmp[freq_sel,:])
        
        # Convert result lists to numpy arrays
        fft_stim = np.array(fft_stim)       # shape: stimuli*freqs*channels
        fft_pre = np.array(fft_pre)
        fft_post = np.array(fft_post)
        # Average the fourier transforms for all stimuli of the same intensity
        fft_stim_avg = np.array( [ np.mean(fft_stim[stim_volt == uval,:,:], axis=0) for uval in stim_volt_u ] )
        fft_pre_avg = np.array( [ np.mean(fft_pre[stim_volt == uval,:,:], axis=0) for uval in stim_volt_u ] )
        fft_post_avg = np.array( [ np.mean(fft_post[stim_volt == uval,:,:], axis=0) for uval in stim_volt_u ] )

        # Return the results
        return fft_stim_avg, fft_pre_avg, fft_post_avg, fft_freqs
