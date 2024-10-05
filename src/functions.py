import emd.sift as sift
import emd.spectra as spectra
import numpy as np
import pingouin as pg
import sails
from scipy.signal import convolve2d, butter, filtfilt
from scipy.stats import zscore, binned_statistic
from skimage.feature import peak_local_max
from icecream import ic


def get_states(states, state_number, sample_rate):
    """
        Extract states from a binary state vector.

        Parameters:
        states (numpy.ndarray): A state vector where 5 represents REM sleep and other values indicate non-REM.
        sample_rate (int or float): The sampling rate of the data.
        state_number: A number that corresponds to a certain state in the state vector

        Returns:
        numpy.ndarray: An array of consecutive REM sleep state intervals in seconds, represented as (start, end) pairs.

        Notes:
        - This function processes a binary state vector and identifies consecutive state intervals.
        - It calculates the start and end times of each state interval based on the provided sample rate.
        - The resulting intervals are returned as a numpy array of (start, end) pairs in seconds.
    """
    states = np.squeeze(states)
    state_indices = np.where(states == state_number)[0]
    state_changes = np.diff(state_indices)
    split_indices = np.where(state_changes != 1)[0] + 1
    split_indices = np.concatenate(([0], split_indices, [len(state_indices)]))
    consecutive_states = np.empty((len(split_indices) - 1, 2))
    for i, (start, end) in enumerate(zip(split_indices, split_indices[1:])):
        start = state_indices[start] * int(sample_rate)
        end = state_indices[end - 1] * int(sample_rate)
        consecutive_states[i] = np.array([start, end])
    consecutive_states = np.array(consecutive_states)
    null_states_mask = np.squeeze(np.diff(consecutive_states) > 0)
    consecutive_states = consecutive_states[null_states_mask]
    return consecutive_states.astype(int)

def get_rem_states(states, sample_rate):
    return get_states(states=states, state_number=5,sample_rate=sample_rate)

def morlet_wt(x, sample_rate, frequencies=np.arange(1, 200, 1), n=5, mode='complex'):
    """
        Compute the Morlet Wavelet Transform of a signal.

        Parameters: 
        
        x (numpy.ndarray): The input signal for which the Morlet Wavelet Transform is computed.
        sample_rate (int or float): The sampling rate of the input signal. frequencies (numpy.ndarray, optional): An
        array of frequencies at which to compute the wavelet transform. Default is a range from 1 to 200 Hz with a
        step of 1 Hz. n (int, optional): The number of cycles of the Morlet wavelet. Default is 5. mode (str,
        optional): The return mode for the wavelet transform. Options are 'complex' (default),'amplitude', and 'power'.

        Returns:
        numpy.ndarray: The Morlet Wavelet Transform of the input signal.

        Notes: - This function computes the Morlet Wavelet Transform of a given signal. - The wavelet transform is
        computed at specified frequencies. - The number of cycles for the Morlet wavelet can be adjusted using the
        'n' parameter. - The result can be returned in either complex or magnitude form either as amplitude or power,
        as specified by the 'mode' parameter.
    """
    wavelet_transform = sails.wavelet.morlet(x, freqs=frequencies, sample_rate=sample_rate, ncycles=n,
                                             ret_mode=mode, normalise=None)
    return wavelet_transform

def bandpass_filter(x,sample_rate, theta_range=(5,12), order=4):
    """
        Applies a bandpass filter to the input signal.

        Parameters:
        x (numpy.ndarray): The input signal to be filtered.
        sample_rate (float): The sampling rate of the input signal.
        theta_range (tuple, optional): The frequency range (in Hz) for the bandpass filter.
        Defaults to (5, 12) representing the theta frequency range.
        order (int, optional): The order of the Butterworth filter. Defaults to 4.

        Returns:
        - array-like: The filtered signal.

        The function calculates the Nyquist frequency based on the provided sample rate and
        defines the low and high cutoff frequencies for the bandpass filter within the specified
        theta_range. It then designs a Butterworth bandpass filter with the given order and applies
        it to the input signal using the filtfilt method to avoid phase shift.

        Example:
        >>> filtered_data = bandpass_filter(input_data, 200, theta_range=(5, 12), order=4)
    """
    nyquist = 0.5 * sample_rate
    low = theta_range[0] / nyquist
    high = theta_range[1] / nyquist

    # Design bandpass filter using Butterworth filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter using filtfilt to avoid phase shift
    filtered_signal = filtfilt(b, a, x)

    return filtered_signal


def tg_split(mask_freq, theta_range=(5, 12)):
    """
        Split a frequency vector into sub-theta, theta, and supra-theta components.

        Parameters:
        mask_freq (numpy.ndarray): A frequency vector or array of frequency values.
        theta_range (tuple, optional): A tuple defining the theta frequency range (lower, upper).
            Default is (5, 12).

        Returns:
        tuple: A tuple containing boolean masks for sub-theta, theta, and supra-theta frequency components.

        Notes: - This function splits a frequency mask into three components based on a specified theta frequency
        range. - The theta frequency range is defined by the 'theta_range' parameter. - The resulting masks 'sub',
        'theta', and 'supra' represent sub-theta, theta, and supra-theta frequency components.
    """
    lower = np.min(theta_range)
    upper = np.max(theta_range)
    mask_index = np.logical_and(mask_freq >= lower, mask_freq < upper)
    sub_mask_index = mask_freq < lower
    supra_mask_index = mask_freq > upper
    sub = sub_mask_index
    theta = mask_index
    supra = supra_mask_index

    return sub, theta, supra


def zero_cross(x):
    """
        Find the indices of zero-crossings in a 1D signal.

        Parameters:
        x (numpy.ndarray): The input 1D signal.

        Returns:
        numpy.ndarray: An array of indices where zero-crossings occur in the input signal.

        Notes:
        - This function identifies the indices where zero-crossings occur in a given 1D signal.
        - It detects both rising and falling zero-crossings.
    """
    decay = np.logical_and((x > 0)[1:], ~(x > 0)[:-1]).nonzero()[0]
    rise = np.logical_and((x <= 0)[1:], ~(x <= 0)[:-1]).nonzero()[0]
    zero_xs = np.sort(np.append(rise, decay))
    return zero_xs


def extrema(x):
    """
        Find extrema (peaks, troughs) and zero crossings in a 1D signal.

        Parameters:
        x (numpy.ndarray): The input 1D signal.

        Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Indices of zero-crossings in the input signal.
            - numpy.ndarray: Indices of troughs in the input signal.
            - numpy.ndarray: Indices of peaks in the input signal.

        Notes:
        - This function identifies and returns the indices of zero-crossings, troughs, and peaks in a given 1D signal.
        - Zero-crossings are points where the signal crosses the zero axis.
        - Troughs are local minima, and peaks are local maxima in the signal.
    """
    zero_xs = zero_cross(x)
    peaks = np.empty((0,)).astype(int)
    troughs = np.empty((0,)).astype(int)
    for t1, t2 in zip(zero_xs, zero_xs[1:]):
        extrema0 = np.argmax(np.abs(x[t1:t2])).astype(int) + t1
        if bool(x[extrema0] > 0):
            peaks = np.append(peaks, extrema0)
        else:
            troughs = np.append(troughs, extrema0)
    return zero_xs, troughs, peaks


def get_cycles(x, mode='peak'):
    zero_x, trough, peak = extrema(x)
    shape_bool_mask = np.empty((1,5)).astype(bool)
    if mode == 'trough':
        zero_x = zero_x[(zero_x > peak[0]) & (zero_x < peak[-1])]
        trough = trough[(trough > zero_x[0]) & (trough < zero_x[-1])]
        shape_bool_mask = [False,False,True,True]
    elif mode == 'peak':
        zero_x = zero_x[(zero_x > trough[0]) & (zero_x < trough[-1])]
        peak = peak[(peak > zero_x[0]) & (peak < zero_x[-1])]
        shape_bool_mask = [True,True,False,False]
    elif mode == 'zero-peak':
        # Get rid of all troughs before the first peak and all peaks after the last trough
        trough = trough[trough > peak[0]]
        peak = peak[peak < trough[-1]]
        
        sequence = [zero_x[0],peak[0],zero_x[1],trough[0],zero_x[2]]
        sequence = np.all(np.diff (sequence) >0)
        
        while not sequence:
            zero_x=zero_x[1:]
            sequence = np.all(np.diff([zero_x[0],peak[0],zero_x[1],trough[0],zero_x[2]]) > 0)
            

        shape_bool_mask = [True,False,False,True]
    elif mode =='zero-trough':
        # Get rid of all peaks before the first trough and all troughs after the last peak
        trough = trough[trough < peak[-1]]
        peak = peak[peak > trough[0]]

        sequence = [zero_x[0],trough[0],zero_x[1],peak[0],zero_x[2]]
        sequence = np.all(np.diff (sequence) >0)
        
        while not sequence:
            zero_x=zero_x[1:]
            sequence = np.all(np.diff([zero_x[0],peak[0],zero_x[1],trough[0],zero_x[2]]) > 0)

        shape_bool_mask = [False,True,True,False]

    indices = np.sort(np.hstack([zero_x,trough,peak]))

    cycles_shape= np.array([indices[::4][:-1].shape[0],
                            indices[1::4].shape[0],
                            indices[2::4].shape[0],
                            indices[3::4].shape[0],
                            indices[::4][1:].shape[0]])
    
    
    cycles = np.zeros((np.max(cycles_shape),5)).astype(int)

    for i, (shape,cycle) in enumerate(zip(cycles_shape,cycles.T)):
        if i == 0:
            cycle[:shape]=indices[::4][:-1]     
        elif i == range(cycles.shape[1])[-1]:
            cycle[:shape]=indices[::4][1:]
        else:
            cycle[:shape]=indices[i::4]


    sequence_check=np.all(np.diff(cycles,axis=1) > 0, axis=1)
    shape_check = np.all((np.diff(x[cycles],axis=1) > 0) == shape_bool_mask,axis=1)

    cycles_mask = np.logical_and(sequence_check,shape_check)
    
    cycles = cycles[cycles_mask]
    return cycles


def bin_tf_to_fpp(x, power, bin_count):
    """
       Bin time-frequency power data into Frequency Phase Power (FPP) plots using specified time intervals of cycles.

       Parameters:
       x (numpy.ndarray): A 1D or 2D array specifying time intervals of cycles for binning.
           - If 1D, it represents a single time interval [start, end].
           - If 2D, it represents multiple time intervals, where each row is [start, end].
       power (numpy.ndarray): The time-frequency power spectrum data to be binned.
       bin_count (int): The number of bins to divide the time intervals into.

       Returns:
       fpp(numpy.ndarray): Returns FPP plots

       Notes:
       - This function takes time-frequency power data and divides it into FPP plots based on specified
         time intervals.
       - The 'x' parameter defines the time intervals, which can be a single interval or multiple intervals.
       - The 'power' parameter is the time-frequency power data to be binned.
       - The 'bin_count' parameter determines the number of bins within each time interval.
    """

    if x.ndim == 1:  # Handle the case when x is of size (2)
        bin_ranges = np.arange(x[0], x[1], 1)
        fpp = binned_statistic(bin_ranges, power[:, x[0]:x[1]], 'mean', bins=bin_count)[0]
        fpp = np.expand_dims(fpp, axis=0)  # Add an extra dimension to match the desired output shape
    elif x.ndim == 2:  # Handle the case when x is of size (n, 2)
        fpp = []
        for i in range(x.shape[0]):
            bin_ranges = np.arange(x[i, 0], x[i, 1], 1)
            fpp_row = binned_statistic(bin_ranges, power[:, x[i, 0]:x[i, 1]], 'mean', bins=bin_count)[0]
            fpp.append(fpp_row)
        fpp = np.array(fpp)
    else:
        raise ValueError("Invalid size for x")

    return fpp


def calculate_cog(frequencies, angles, amplitudes, ratio):
    """
       Calculate the Center of Gravity (CoG) of an FPP plots of cycles.

       Parameters:
       frequencies (numpy.ndarray): An array of frequencies corresponding to FPP frequencies.
       angles (numpy.ndarray): An array of phase angles in degrees.
       amplitudes (numpy.ndarray): An array of magnitude values (can be power).
           - If 2D, it represents magnitude values for multiple frequency bins.
           - If 3D, it represents magnitude values for multiple frequency bins across multiple trials or subjects.
       ratio (float): A ratio threshold for selecting magnitude values in the phase direction .

       Returns:
       numpy.ndarray: The Center of Gravity (CoG) for the FPP plot as frequency and phase.
           - For 2D amplitudes: A 2D array containing CoG values for frequency and phase.
           - For 3D amplitudes: A 2D array containing CoG values for frequency and phase cycle.

       Notes:
       - This function calculates the Center of Gravity (CoG) of the FPP plots.
       - It can handle 2D or 3D amplitude arrays, representing either single or multiple cycles.
    """
    angles = np.deg2rad(angles)
    cog = np.empty((0, 2))

    if amplitudes.ndim == 2:
        numerator = np.sum(frequencies * np.sum(amplitudes, axis=1))
        denominator = np.sum(amplitudes)
        cog_f = numerator / denominator
        floor = np.floor(cog_f).astype(int) - frequencies[0]
        ceil = np.ceil(cog_f).astype(int) - frequencies[0]
        new_fpp = np.where(amplitudes >= np.max(amplitudes[[floor, ceil], :]) * ratio, amplitudes, 0)
        cog_ph = np.rad2deg(pg.circ_mean(angles, w=np.sum(new_fpp, axis=0)))
        cog = np.array([cog_f, cog_ph])

    elif amplitudes.ndim == 3:
        indices_to_subset = np.empty((amplitudes.shape[0], 2)).astype(int)
        cog = np.empty((amplitudes.shape[0], 2))
        numerator = np.sum(frequencies * np.sum(amplitudes, axis=2), axis=1)
        denominator = np.sum(amplitudes, axis=(1, 2))
        cog_f = (numerator / denominator)

        vectorized_floor = np.vectorize(np.floor)
        vectorized_ceil = np.vectorize(np.ceil)
        indices_to_subset[:, 0] = vectorized_floor(cog_f) - frequencies[0]
        indices_to_subset[:, 1] = vectorized_ceil(cog_f) - frequencies[0]

        max_amps = np.max(amplitudes[np.arange(amplitudes.shape[0])[:, np.newaxis], indices_to_subset, :], axis=(1, 2))
        print(max_amps.shape)

        for i, max_amp in enumerate(max_amps):
            new_fpp = np.where(amplitudes[i] >= max_amp * ratio, amplitudes[i], 0)
            cog[i, 1] = np.rad2deg(pg.circ_mean(angles, w=np.sum(new_fpp, axis=0)))
        cog[:, 0] = cog_f

    return cog


def boxcar_smooth(x, boxcar_window):
    """
        Apply a boxcar smoothing filter to a 1D or 2D signal.

        Parameters:
        x (numpy.ndarray): The input signal to be smoothed.
        boxcar_window (int or tuple): The size of the boxcar smoothing window.
            - If int, it specifies the window size in both dimensions for a 2D signal.
            - If tuple, it specifies the window size as (time_window, frequency_window) for a 2D signal.

        Returns:
        numpy.ndarray: The smoothed signal after applying the boxcar smoothing filter.

        Notes:
        - This function applies a boxcar smoothing filter to a 1D or 2D signal.
        - The size of the smoothing window can be specified as an integer (square window) or a tuple (rectangular window).
        - For 2D signals, the boxcar smoothing is applied in both the time and frequency dimensions.
    """
    if x.ndim == 1:
        if boxcar_window % 2 == 0:
            boxcar_window += 1
        window = np.ones((1, boxcar_window)) / boxcar_window
        x_spectrum = np.convolve(x, window, mode='same')
    else:
        bool_window = np.where(~boxcar_window % 2 == 0, boxcar_window, boxcar_window + 1)
        window_t = np.ones((1, bool_window[0])) / bool_window[0]
        window_f = np.ones((1, bool_window[1])) / bool_window[1]
        x_spectrum_t = convolve2d(x, window_t, mode='same')
        x_spectrum = convolve2d(x_spectrum_t, window_f.T, mode='same')

    return x_spectrum

def fpp_peaks(frequencies,angles,fpp_cycles):
    """
        Identify the peak locations on the Frequency Phase Plots of individual cycles and map them to corresponding
        frequencies and angles.

        Parameters:
        frequencies (array-like): An array of frequency values.
        angles (array-like): An array of angle values.
        fpp_cycles (list of 2D arrays): A list where each element is a 2D array representing cycles of some data.

        Returns: list of 2D arrays: A list of 2D arrays where each array contains pairs of frequency and angle values
        corresponding to the peak points in the input cycles.

        Each 2D array in the returned list has shape (n_peaks, 2), where n_peaks is the number of peaks found in the
        corresponding 2D array in fpp_cycles. The first column of each 2D array contains the frequencies of the peaks,
        and the second column contains the angles of the peaks.
    """
    peaks = []
    peak_points = []
    for i, fpp in enumerate(fpp_cycles):
        peak = peak_local_max(fpp, min_distance=1, threshold_abs=0)
        peaks.append(peak)
        fpp_value = fpp[peak[:, 0], peak[:, 1]]
        fpp_value = np.round(fpp_value, decimals=4)
        peak_frequencies = frequencies[peak[:, 0]]
        peak_angles = angles[peak[:, 1]]
        peak_points.append(np.array([peak_frequencies, peak_angles, fpp_value]).T)
    return peak_points

def peak_cog(frequencies, angles, amplitudes, ratio):
    """
       Calculate the Center of Gravity (CoG) and snap to the nearest peak in the FPP array.

       Parameters:
       frequencies (numpy.ndarray): An array of frequencies.
       angles (numpy.ndarray): An array of phase angles in degrees.
       amplitudes (numpy.ndarray): An array of magnitude values (can be power).
           - If 2D, it represents magnitude values for multiple phase bins.
           - If 3D, it represents magnitude values for multiple phase bins across multiple trials or subjects.
       ratio (float): A ratio threshold for selecting magnitude values in the phase direction.

       Returns:
       numpy.ndarray: Snapped peaks of each FPP cycle .
           - For 2D amplitudes: A 2D array containing CoG peaks for frequency and phase.
           - For 3D amplitudes: A 2D array containing CoG peaks for frequency and phase for each cycle.

       Notes:
       - This function calculates the Center of Gravity (CoG) of the passed FPP cycles and their respective magnitude
       peaks.
       - The CoG is then shifted to the nearest peak of Euclidean distance treating frequency as linear and phase as
       circular.
    """

    def nearest_peaks(frequency, angle, amplitude, ratio):
        peak_indices = peak_local_max(amplitude, min_distance=1, threshold_abs=0)
        cog_f = calculate_cog(frequency, angle, amplitude, ratio)

        if peak_indices.shape[0] == 0:
            cog_peak = cog_f
        else:
            cog_fx = np.array([cog_f[0], cog_f[0] * np.cos(np.deg2rad(cog_f[1] - angle[0])),
                               cog_f[0] * np.sin(np.deg2rad(cog_f[1] - angle[0]))])
            peak_loc = peak_loc = np.empty((peak_indices.shape[0], 4))
            peak_loc[:, [0, 1]] = np.array([frequency[peak_indices.T[0]], angle[peak_indices.T[1]]]).T
            peak_loc[:, 2] = peak_loc[:, 0] * np.cos(np.deg2rad(peak_loc[:, 1] - angle[0]))
            peak_loc[:, 3] = peak_loc[:, 0] * np.sin(np.deg2rad(peak_loc[:, 1] - angle[0]))
            peak_loc = peak_loc[:, [0, 2, 3]]
            distances = np.abs(peak_loc - cog_fx)

            cog_pos = peak_indices[np.argmin(np.linalg.norm(distances, axis=1))]

            cog_peak = np.array([frequency[cog_pos[0]], angle[cog_pos[1]]])

        return cog_peak

    if amplitudes.ndim == 2:
        cog = nearest_peaks(frequencies, angles, amplitudes, ratio)
    elif amplitudes.ndim == 3:
        cog = np.empty((amplitudes.shape[0], 2))
        for i, fpp in enumerate(amplitudes):
            cog[i] = nearest_peaks(frequencies, angles, fpp, ratio)
    return cog
