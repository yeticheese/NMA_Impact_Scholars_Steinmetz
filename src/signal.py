from dataclasses import dataclass, field

import emd.sift as sift
import emd.spectra as spectra
import numpy as np
import pandas as pd
import scipy
from icecream import ic
from typing import Tuple
from src.functions import tg_split, get_cycles, get_states, morlet_wt, bin_tf_to_fpp, peak_cog, fpp_peaks


class SignalProcessor:
    """
    A class for processing signals and extracting features.
    This class has two modes of operation, a back-end and front-end.

    The functions that process the intermediate features for further process populate the back-end values.
    
    The front-end values populate their slots with back-end values if they are empty, and can be overwritten
    by values the user has calculated through different operations.

    The functions that call on these values will use the front-end values first which can be populated with back-end
    processed values or manually user-fed values.

    Attributes:
    signal (np.ndarray): The input signal.
    sample_rate (float): The sample rate of the signal.
    freq_range (tuple): The frequency range of interest.

    Front-End Attributes:
    imf (np.ndarray): Intrinsic Mode Functions.
    mask_freq (np.ndarray): Mask frequency.
    IP: Instantaneous phase.
    IF: Instantaneous frequency.
    IA: Instantaneous amplitude.
    theta (np.ndarray): Theta signal.
    cycles: Cycles of the signal.
    phasic: Phasic periods.
    tonic: Tonic periods.

    Back-End Attributes:
    _imf (np.ndarray): Intrinsic Mode Functions.
    _mask_freq (np.ndarray): Mask frequency.
    _IP: Instantaneous phase.
    _IF: Instantaneous frequency.
    _IA: Instantaneous amplitude.
    _theta (np.ndarray): Theta signal.
    _cycles: Cycles of the signal.
    _spike_df: DataFrame containing spike information.
    _phasic: Phasic periods.
    _tonic: Tonic periods.
    """

    def __init__(self, signal: np.ndarray, sample_rate: float, freq_range: tuple):
        """
        Initializes the SignalProcessor class.

        Parameters:
        signal (np.ndarray): The input signal.
        sample_rate (float): The sample rate of the signal.
        freq_range (tuple): The frequency range of interest.
        """
        self.signal = signal
        self.sample_rate = sample_rate
        self.freq_range = freq_range

        # Front-End Attributes
        self.imf: np.ndarray = None
        self.mask_freq: np.ndarray = None
        self.IP = None
        self.IF = None
        self.IA = None
        self.theta: np.ndarray = None
        self.cycles = None
        self.phasic = None
        self.tonic = None

        # Back-End attributes
        self._imf: np.ndarray = None
        self._mask_freq: np.ndarray = None
        self._IP = None
        self._IF = None
        self._IA = None
        self._theta: np.ndarray = None
        self._cycles = None
        self._spike_df = None
        self._phasic = None
        self._tonic = None

    def get_duration(self) -> np.ndarray:
        """
        Calculates the duration of the signal.

        Returns:
        np.ndarray: The duration of each sample in milliseconds.
        """

        duration = np.linspace(0, len(self.signal) / self.sample_rate, len(self.signal)) * 1000
        return duration

    # TODO: Create a dynamic EMD sifting method for the signal processor
    # def sift(self,func=None,**kwargs):
    #     if func == None:
    #         self.imf,self.mask_freq=sift.iterated_mask_sift(self.signal,
    #                                                         mask_0=mask,
    #                                                         sample_rate=self.sample_rate,
    #                                                         ret_mask_freq=True)
    #     else:

    def iter_sift(self, **kwargs) -> tuple:
        """
        Performs an iterated mask sift on the signal.

        Returns:
        tuple: Tuple containing intrinsic mode functions and mask frequency.
        """

        self._imf, self._mask_freq = sift.iterated_mask_sift(self.signal,
                                                             sample_rate=self.sample_rate,
                                                             ret_mask_freq=True)
        return self._imf, self._mask_freq

    def frequency_transform(self) -> tuple:
        """
        Transform the calculated imfs into a tuple of Instantaneous Phase, Instantaneous Frequency, Instantaneous
        Amplitude

        Returns:
        tuple: Typle containing Instantaneous Phase, Instantaneous Frequency, Instantaneous
        Amplitude
        """

        if (getattr(self, '_imf') is None) or (getattr(self, '_mask_freq') is None):
            self._imf, self._mask_freq = self.iter_sift()
        self._IP, self._IF, self._IA = spectra.frequency_transform(self._imf, self.sample_rate, 'nht')

        return self._IP, self._IP, self._IA

    def split_signals(self) -> np.ndarray:
        """
        Split the imfs into sub-theta, theta and gamma signals

        Returns:
        np.ndarray: Returns an array of 3 signals
        """

        if (getattr(self, 'imf') is None) or (getattr(self, 'mask_freq') is None):
            if (getattr(self, '_imf') is None) or (getattr(self, '_mask_freq') is None):
                self._imf, self._mask_freq = self.iter_sift()
                imf = self._imf
                mask_freq = self._mask_freq
            else:
                imf = self._imf
                mask_freq = self._mask_freq
        else:
            imf = self.imf
            mask_freq = self.mask_freq

        sub, theta, gamma = tg_split(mask_freq, self.freq_range)
        split_signals = np.empty((3, self.signal.shape[0]))
        split_signals[0] = np.sum(imf.T[sub], axis=0)
        split_signals[1] = np.sum(imf.T[theta], axis=0)
        split_signals[2] = np.sum(imf.T[gamma], axis=0)

        return split_signals

    def get_theta(self) -> np.ndarray:
        """
        Get theta signal.

        Returns:
        np.ndarray: Theta signal.
        """

        self._theta = self.split_signals()[1]
        return self._theta

    def get_cycles(self, mode='peak'):
        """
        Get cycles.

        Parameters:
        mode (str): Cycle output mode. Default is "peak"

        Returns:
        np.ndarray: Cycles array, default is peak-centred.
        """

        cycles = get_cycles(self.get_theta(), mode)
        self._cycles = cycles
        return cycles

    # TODO: Fix duration length data type adjustability
    def apply_duration_threshold(self, duration_length: float or tuple = None):
        """
        Applies a duration threshold to filter cycles.

        Args:
        duration_length (float or tuple, optional): The duration threshold in milliseconds.
        """
        if duration_length is None:
            duration_length = 1000 / np.array(self.freq_range)
        if getattr(self, 'cycles') is None:
            if getattr(self, '_cycles') is None:
                print('Back-end cycles attribute is missing')
            else:
                cycles = self._cycles
        else:
            cycles = self.cycles

        duration_check = np.diff(cycles[:, [0, -1]], axis=1) * (1000 / self.sample_rate)
        duration_check_mask = np.squeeze(
            np.logical_and(duration_check <= duration_length[0], duration_check > duration_length[1]))

        self._cycles = cycles[duration_check_mask]

    def apply_amplitude_threshold(self, mode='sleep'):
        """
        Applies an amplitude threshold to filter cycles based on their peak amplitudes. Two modes, sleep or wake each corresponding to the amplitude
        distribution of the signal during a brain state.

        """
        if getattr(self, 'cycles') is None:
            if getattr(self, '_cycles') is None:
                print('Back-end cycles attribute is missing')
            else:
                cycles = self._cycles
        else:
            cycles = self.cycles

        sub_theta = self.split_signals()[0]
        theta = self.split_signals()[1]
        theta_peak_amp = theta[self.cycles[:, 2]]
        
        if mode == 'sleep':
            amp_threshold = 2 * np.abs(sub_theta).std()
            amp_threshold_mask = theta_peak_amp > amp_threshold
            self.cycles = self.cycles[amp_threshold_mask]

        elif mode == 'wake':
            amp_threshold = np.copy(sub_theta)
            min_theta_amp = np.median(np.abs(theta))
            amp_threshold[amp_threshold < min_theta_amp] = min_theta_amp
            amp_threshold_mask = theta_peak_amp >= amp_threshold[self.cycles[:, 2]]
            self.cycles = self.cycles[amp_threshold_mask]

    def morlet_wt(self, band: int or str or Tuple[int, ...] = 'gamma', frequencies=(1,200), norm='zscore',mode='power'):
        """

        Parameters:
            band: an imf index integer (int) or frequency range (tuple) or oscillatory frequency band (str) to select
            the Intrinsic Mode Functions from the imf attribute. Multiple IMFs selected will be summed together.
            frequencies:Frequencies to perform morlet wavelet decomposition on.
            norm: normalization mode
            mode: return mode which can be either 'power' or 'amplitude' or 'complex' form

        Returns:
            np.ndarray: a 2D array of the morlet wavelet transform of the signal.

        """
        frequency_vector = np.arange(frequencies[0], frequencies[1]+1, 1)
        wavelet_signal = np.empty(self.signal.shape)
        if isinstance(band, int):
            wavelet_signal = self.imf.T[band]
        elif isinstance(band, tuple) and all(isinstance(item, int) for item in band):
            wavelet_signal = np.sum(self.imf[:,[band]].T,axis=0)
        elif isinstance(band, str):
            if band == 'gamma':
                wavelet_signal = self.split_signals()[2]
            elif band == 'theta':
                wavelet_signal = self.split_signals()[1]
            elif band == 'sub-theta':
                wavelet_signal = self.split_signals()[0]

        wavelet_transform = morlet_wt(x=wavelet_signal, sample_rate=self.sample_rate, frequencies=frequency_vector,
                                      mode=mode)

        if norm is None:
            return wavelet_transform
        elif norm == 'zscore' and mode != 'complex':
            return scipy.stats.zscore(wavelet_transform,axis=0)

    def get_fpp_cycles(self,**kwargs): # Temporary function
        """

        Parameters:
            **kwargs: keyword arguments passed to morlet_wt() function

        Returns:
            np.ndarray: Array containing frequency phase plots of each cycle.

        """
        wavelet_transform = self.morlet_wt(**kwargs)
        fpp_cycles = bin_tf_to_fpp(x=self.cycles[:, [0, -1]], power=wavelet_transform, bin_count=19)
        return fpp_cycles

    def get_fpp_peaks(self, **kwargs):
        """

        Parameters:
            **kwargs: keyword arguments passed to fpp_peaks() function

        Returns:
            np.ndarray: Array containing the locations of peaks within the frequency phase plot

        """
        fpp_cycles = self.get_fpp_cycles(**kwargs)
        frequency_vector = np.array([])
        for kwarg, v in kwargs.items():
            if kwarg == 'frequencies':
                frequency_vector = np.arange(v[0], v[1]+1, 1)
        angle_vector = np.linspace (-180,180,19)
        peak_points = fpp_peaks(frequencies=frequency_vector, angles= angle_vector, fpp_cycles=fpp_cycles)
        return peak_points



    def peak_center_of_gravity(self,**kwargs):
        """
        Calculates the peak center of gravity values from FPP plots of our cycles.
        """

        fpp_cycles = self.get_fpp_cycles(**kwargs)
        frequency_vector = np.array([])
        frequencies = np.arange(15, 141, 1)
        for kwarg, v in kwargs.items():
            if kwarg == 'frequencies':
                frequency_vector = np.arange(v[0], v[1]+1, 1)
            if kwarg == 'norm' and v == 'zscore':
                minimums = np.min(fpp_cycles, axis=(1,2))[:, np.newaxis, np.newaxis]*np.ones_like(fpp_cycles)
                fpp_cycles += minimums
        angle_vector = np.linspace(-180, 180, 19)
        cog_values = peak_cog(frequency_vector,angle_vector, fpp_cycles, 0.95)

        return cog_values

    @property
    def imf(self):
        """
        Getter for the imf array.
        """
        return self._imf

    @imf.setter
    def imf(self, value):
        """
        Getter for the imf array.
        """
        self._imf = value

    @property
    def mask_freq(self):
        """
        Getter for the mask_freq array.
        """
        return self._mask_freq

    @mask_freq.setter
    def mask_freq(self, value):
        """
        Getter for the imf array.
        """
        self._mask_freq = value

    @property
    def theta(self):
        """
        Getter for the theta signal.
        """
        return self._theta

    @theta.setter
    def theta(self, value):
        """
        Getter for the imf array.
        """
        self._theta = value

    @property
    def cycles(self):
        """
        Getter for the cycles array (theta).
        """
        return self._cycles

    @cycles.setter
    def cycles(self, value):
        """
        Setter for the cycles array.
        """
        self._cycles = value

    @property
    def phasic(self):
        """
        Getter for phasic periods.
        """
        return self._phasic

    @phasic.setter
    def phasic(self, value):
        """
        Setter for the phasic periods.
        """
        self._phasic = value

    @property
    def tonic(self):
        """
        Getter for tonic periods.
        """
        return self._tonic

    @tonic.setter
    def tonic(self, value):
        """
        Setter for the tonic periods.
        """
        self._tonic = value


class SegmentSignalProcessor(SignalProcessor):
    def __init__(self, signal: np.ndarray, period: np.ndarray, sample_rate: float, freq_range: tuple, ):
        super().__init__(signal, sample_rate, freq_range)
        self.period = period

    def get_duration(self) -> np.ndarray:
        time = self.period / self.sample_rate
        duration = np.linspace(time[0], time[1], len(self.signal)) * 1000
        return duration

    def get_cycles(self, mode='peak'):
        cycles = super().get_cycles(mode=mode) + self.period[0]
        self._cycles = cycles
        return cycles

    def get_fpp_cycles(self, **kwargs):
        wavelet_transform = self.morlet_wt(**kwargs)
        fpp_cycles = bin_tf_to_fpp(x=self.cycles[:, [0, -1]] - self.period[0], power=wavelet_transform,bin_count=19)
        return fpp_cycles

    # def peak_center_of_gravity(self):
    #     frequencies = np.arange(15, 141, 1)
    #     angles = angles = np.linspace(-180, 180, 19)
    #     gamma = self.split_signals()[2]
    #     power = morlet_wt(gamma,
    #                       self.sample_rate,
    #                       frequencies,
    #                       mode='power')
    #     power = scipy.stats.zscore(power, axis=0)
    #     shifted_zscore_power = power + 2 * np.abs(np.min(power))
    #     fpp_cycles = bin_tf_to_fpp(self.cycles[:, [0, -1]] - self.period[0], shifted_zscore_power, 19)
    #     cog_values = peak_cog(frequencies, angles, fpp_cycles, 0.95)
    #
    #     return cog_values


@dataclass
class SleepSignal(SignalProcessor):
    """
    A data class representing a signal with various attributes.

    Attributes:
    - signal (np.ndarray): The signal data.
    - sample_rate (float): The sampling rate of the signal.
    - freq_range (tuple): The theta frequency range of the signal.
    - cycles (int): Private attribute for the cycles index data. (To be added)

    Methods:
    - To be determined
    """
    signal: np.ndarray
    rem_states: np.ndarray
    sample_rate: float
    freq_range: tuple
    REM: list = field(default_factory=list)  # Initialize _REM as an empty list using default_factory
    cycles: np.ndarray = field(default_factory=lambda: np.empty((0, 5)).astype(
        int))  # Initialize cycles with empty array using default_factory

    def __post_init__(self):
        if not self.REM:
            ic('REM List is empty')
            actual_rem_states = []
            for i, rem_period in enumerate(self.rem_states):
                signal = self.signal[rem_period[0]:rem_period[1]]
                try:
                    REM = REM_Segment(signal, rem_period, self.sample_rate, self.freq_range)
                    actual_rem_states.append(i)
                    self.REM.append(REM)
                except Exception as e:
                    f'Error processing REM period {rem_period} '
                    continue
            self.rem_states = self.rem_states[actual_rem_states]
        else:
            ic('REM List provided')
        self.get_cycles()
        self.apply_duration_threshold()
        self.apply_amplitude_threshold(mode='sleep')

    def get_cycles(self, mode='peak'):
        cycles = np.empty((0, 5)).astype(int)
        for rem in self.REM:
            if hasattr(rem, 'cycles') is None:
                cycles = np.vstack([cycles, rem.get_cycles(mode=mode)])
            else:
                cycles = np.vstack([cycles, rem.cycles])
        self.cycles = cycles
        return cycles

    def apply_duration_threshold(self, duration_length: float or tuple = None):
        for rem in self.REM:
            rem.apply_duration_threshold(duration_length=duration_length)
        super().apply_duration_threshold(duration_length=duration_length)


    def apply_amplitude_threshold(self, mode='normal'):
        sub_theta = np.array([])
        theta = np.array([])
        theta_peak_amp = np.array([])
        sub_theta_pk_mask = np.array([])
        if mode == 'sleep':
            for rem in self.REM:
                sub_theta = np.append(sub_theta, np.sum(rem.imf.T[tg_split(rem.mask_freq)[0]], axis=0))
                theta = rem.get_theta()
                theta_peak_amp = np.append(theta_peak_amp, theta[rem.cycles[:, 2] - rem.period[0]])
            amp_threshold = 2 * sub_theta.std()
            amp_threshold_mask = theta_peak_amp > amp_threshold
            self.cycles = self.cycles[amp_threshold_mask]

            for rem in self.REM:
                theta_sig = rem.get_theta()
                cycles_mask = theta_sig[rem.cycles[:, 2] - rem.period[0]] > amp_threshold
                rem.cycles = rem.cycles[cycles_mask]

        elif mode == 'wake':
            for rem in self.REM:
                theta = np.append(theta, rem.get_theta())
                theta_peak_amp = np.append(theta_peak_amp, rem.get_theta()[rem.cycles[:, 2] - rem.period[0]])
            amp_threshold = np.median(np.abs(theta))
            for rem in self.REM:
                sub_theta_signal = rem.split_signals()[0]
                sub_theta = np.append(sub_theta, sub_theta_signal)
                sub_theta_peaks = sub_theta_signal[rem.cycles[:, 2] - rem.period[0]]
                sub_theta_peaks[sub_theta_peaks < amp_threshold] = amp_threshold
                sub_theta_pk_mask = np.append(sub_theta_pk_mask, sub_theta_peaks)

                theta_sig = rem.get_theta()
                cycles_mask = theta_sig[rem.cycles[:, 2] - rem.period[0]] >= sub_theta_peaks
                rem.cycles = rem.cycles[cycles_mask]

            amp_threshold_mask = theta_peak_amp >= sub_theta_pk_mask
            self.cycles = self.cycles[amp_threshold_mask]

    def spike_df(self):
        spike_df = pd.DataFrame()
        for rem in self.REM:
            spike_df = pd.concat([spike_df, rem._spike_df], axis=0, ignore_index=True)
        self._spike_df = spike_df
        return spike_df

    def get_fpp_cycles(self,**kwargs):
        ic(kwargs)
        for kwarg,v in kwargs.items():
            if kwarg == 'frequencies':
                frequency_vector = np.arange(v[0], v[1]+1, 1)
        fpp_cycles = np.empty((0, frequency_vector.shape[0], 19))
        ic(fpp_cycles.shape)
        for rem in self.REM:
            wavelet_transform = rem.morlet_wt(**kwargs)
            fpp_plots = bin_tf_to_fpp(x=rem.cycles[:, [0, -1]] - rem.period[0], power=wavelet_transform, bin_count=19)
            fpp_cycles = np.vstack((fpp_cycles, fpp_plots))
        return fpp_cycles

    def build_dataset(self, **kwargs):
        df = pd.DataFrame(self.cycles)
        df = df.rename(columns={0: 'first_trough', 1: 'first_zero_x', 2: 'peak', 3: 'last_zero_x', 4: 'last_trough'})
        df['sample_rate'] = self.sample_rate
        df['peak_amplitude'] = self.signal[self.cycles[:, 2]]
        df['fpp_peaks'] = self.get_fpp_peaks(**kwargs)

        return df


@dataclass
class REM_Segment(SegmentSignalProcessor):
    signal: np.ndarray
    period: np.ndarray
    sample_rate: float
    freq_range: tuple
    imf: np.ndarray = field(default_factory=lambda: np.array([]))
    mask_freq: float = field(default_factory=lambda: np.array([]))
    IP: np.ndarray = field(default_factory=lambda: np.array([]), metadata='Instaneous Power')
    IF: np.ndarray = field(default_factory=lambda: np.array([]), metadata='Instantaneous Frequency')
    IA: np.ndarray = field(default_factory=lambda: np.array([]), metadata='Instantaneous Amplitude')
    cycles: np.ndarray = field(default_factory=lambda: np.empty((0, 5)).astype(int))  # Initialize cycles with empty array using default_factory
    _spike_df: pd.DataFrame = None
    tonic: np.ndarray = None
    phasic: np.ndarray = None

    def __post_init__(self):
        REM = SegmentSignalProcessor(self.signal, self.period, self.sample_rate, self.freq_range)
        if (self.imf.size == 0) or (self.mask_freq.size == 0):
            ic('No imf data, generating imfs....')
            self.imf, self.mask_freq = REM.iter_sift()
            self.IP, self.IF, self.IA = REM.frequency_transform()
        if self.cycles.size == 0:
            ic('No cycle data, extracting cycles....')
            REM.get_cycles()
            self.cycles = REM._cycles

@dataclass
class WakeSignal(SignalProcessor):
    signal: np.ndarray
    sample_rate: float
    freq_range: tuple
    imf: np.ndarray = field(default_factory=lambda: np.array([]))
    mask_freq: float = field(default_factory=lambda: np.array([]))
    IP: np.ndarray = field(default_factory=lambda: np.array([]), metadata='Instaneous Power')
    IF: np.ndarray = field(default_factory=lambda: np.array([]), metadata='Instantaneous Frequency')
    IA: np.ndarray = field(default_factory=lambda: np.array([]), metadata='Instantaneous Amplitude')
    # Initialize cycles with empty array using default_factory
    cycles: np.ndarray = field(default_factory=lambda: np.empty((0, 5)).astype(int))

    def __post_init__(self):
        if (self.imf.size == 0) or (self.mask_freq.size == 0):
            ic('No imf data, generating imfs....')
            self.imf, self.mask_freq = self.iter_sift()
            self.IP, self.IF, self.IA = self.frequency_transform()
        if self.cycles.size == 0:
            ic('No cycle data, extracting cycles....')
            self.get_cycles()
        self.apply_duration_threshold()
        self.apply_amplitude_threshold(mode='wake')

    def build_dataset(self,**kwargs):
        df = pd.DataFrame(self.cycles)
        df = df.rename(columns={0: 'first_trough', 1: 'first_zero_x', 2: 'peak', 3: 'last_zero_x', 4: 'last_trough'})
        df['sample_rate'] = self.sample_rate
        df['peak_amplitude'] = self.signal[self.cycles[:, 2]]
        df['fpp_peaks'] = self.get_fpp_peaks(**kwargs)
        return df