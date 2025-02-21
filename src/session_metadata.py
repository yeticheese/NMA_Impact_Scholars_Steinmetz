from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import tarfile
import numpy as np
from typing import Dict, Optional
from src.file_ops import npy_loader
import pandas as pd

def calculate_peak_to_trough_duration(waveforms, sampling_rate=30000):
    """
    Calculate the peak-to-trough duration of a waveform.

    Parameters:
    waveform (np.ndarray): Template waveform array
    sampling_rate (int): Sampling rate in Hz, default 30000 for typical ephys

    Returns:
    float: Peak-to-trough duration in milliseconds
    """
    # Find peak and trough indices

    peak_idx = np.argmax(waveforms)
    trough_idx = np.argmin(waveforms)

    # Calculate time difference
    time_diff_samples = abs(peak_idx - trough_idx)

    # Convert to milliseconds
    duration_ms = (time_diff_samples / sampling_rate) * 1000

    return duration_ms

@dataclass
class BaseLoader(ABC):
    """Abstract base class for loading data from tar archives."""

    @classmethod
    @abstractmethod
    def get_file_mapping(cls) -> Dict[str, str]:
        """Subclasses must define the mapping of tar filenames to attributes."""
        pass

    @classmethod
    def from_tar(cls, tar_path: str | Path) -> 'BaseLoader':
        """Load data from a tar archive."""
        instance = cls()
        file_attr_map = cls.get_file_mapping()

        with tarfile.open(tar_path, 'r') as tar:
            for file_name, attr_name in file_attr_map.items():
                if file_name not in tar.getnames():
                    print(f"Warning: {file_name} not found in tar archive")
                    continue
                setattr(instance, attr_name, np.squeeze(npy_loader(tar, file_name)))

        return instance

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts field data to a pandas DataFrame.
        For multi-dimensional arrays, only the first dimension is used as the index,
        and the remaining dimensions are stored as array objects in the cells.
        """
        data_dict = {}
        base_length = None

        # Process each attribute
        for attr_name, value in self.__dict__.items():
            if value is not None:
                if len(value.shape) == 1:
                    # 1D arrays can be directly added
                    data_dict[attr_name] = value
                    if base_length is None:
                        base_length = len(value)
                else:
                    # For multi-dimensional arrays, store them as objects
                    # Each row will contain a slice of the array
                    data_dict[attr_name] = [value[i] for i in range(value.shape[0])]
                    if base_length is None:
                        base_length = value.shape[0]

        # Create DataFrame
        df = pd.DataFrame(data_dict)

        return df



@dataclass
class Clusters(BaseLoader):
    """Represents cluster-related data."""
    depths: Optional[np.ndarray] = field(default=None)
    original_ids: Optional[np.ndarray] = field(default=None)
    site: Optional[np.ndarray] = field(default=None)
    probe: Optional[np.ndarray] = field(default=None)
    template_waveform_chans: Optional[np.ndarray] = field(default=None)
    template_waveforms: Optional[np.ndarray] = field(default=None)
    waveform_duration: Optional[np.ndarray] = field(default=None)
    phy_annotation: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        """Ensure numeric fields are of integer type where applicable."""
        for attr in ["original_ids", "site", "probe"]:
            value = getattr(self, attr, None)
            if value is not None:
                setattr(self, attr, value.astype(int))

    @classmethod
    def from_tar(cls, tar_path: str | Path) -> 'BaseLoader':
        """Load data from a tar archive."""
        instance = cls()
        file_attr_map = cls.get_file_mapping()

        with tarfile.open(tar_path, 'r') as tar:
            for file_name, attr_name in file_attr_map.items():
                if file_name not in tar.getnames():
                    print(f"Warning: {file_name} not found in tar archive")
                    continue
                elif attr_name == 'template_waveforms':
                    setattr(instance, attr_name, np.squeeze(npy_loader(tar, file_name))[:,:,0])
                else:
                    setattr(instance, attr_name, np.squeeze(npy_loader(tar, file_name)))

        return instance

    @classmethod
    def get_file_mapping(cls) -> Dict[str, str]:
        return {
            'clusters.depths.npy': 'depths',
            'clusters.originalIDs.npy': 'original_ids',
            'clusters.peakChannel.npy': 'site',
            'clusters.probes.npy': 'probe',
            'clusters.templateWaveformChans.npy': 'template_waveform_chans',
            'clusters.templateWaveforms.npy': 'template_waveforms',
            'clusters.waveformDuration.npy': 'waveform_duration',
            'clusters._phy_annotation.npy': 'phy_annotation',
        }

    def to_dataframe(self) -> pd.DataFrame:
        df = super().to_dataframe()
        df['peak_to_trough_duration'] = df['template_waveforms'].apply(calculate_peak_to_trough_duration)

        return df

@dataclass
class Trials(BaseLoader):
    """Represents trial-related data."""
    feedback_type: Optional[np.ndarray] = field(default=None)
    feedback_times: Optional[np.ndarray] = field(default=None)
    gocue_times: Optional[np.ndarray] = field(default=None)
    included: Optional[np.ndarray] = field(default=None)
    intervals: Optional[np.ndarray] = field(default=None)
    repNum: Optional[np.ndarray] = field(default=None)
    response_choice: Optional[np.ndarray] = field(default=None)
    response_times: Optional[np.ndarray] = field(default=None)
    contrast_left: Optional[np.ndarray] = field(default=None)
    contrast_right: Optional[np.ndarray] = field(default=None)
    stimulus_times: Optional[np.ndarray] = field(default=None)

    @classmethod
    def get_file_mapping(cls) -> Dict[str, str]:
        return {
            'trials.feedbackType.npy': 'feedback_type',
            'trials.feedback_times.npy': 'feedback_times',
            'trials.goCue_times.npy': 'gocue_times',
            'trials.included.npy': 'included',
            'trials.intervals.npy': 'intervals',
            'trials.repNum.npy': 'repNum',
            'trials.response_choice.npy': 'response_choice',
            'trials.response_times.npy': 'response_times',
            'trials.visualStim_contrastLeft.npy': 'contrast_left',
            'trials.visualStim_contrastRight.npy': 'contrast_right',
            'trials.visualStim_times.npy': 'stimulus_times',
        }

    def to_dataframe(self) -> pd.DataFrame:
        df = super().to_dataframe()
        df['trial_interval'] = df.stimulus_times.apply(lambda x: np.array([x-0.5,x+2]))
        df['trial_start'] = df.stimulus_times.apply(lambda x: x-0.5)
        df['trial_end'] = df.stimulus_times.apply(lambda x: x+2)
        df['quiescence_intervals'] = df.stimulus_times.apply(lambda x: np.array([x-0.5,x]))
        df['quiescence_start'] = df.stimulus_times.apply(lambda x: x-0.5)
        df['quiescence_end'] = df.stimulus_times.apply(lambda x: x)

        return df

@dataclass
class Spikes(BaseLoader):
    """Represents spike-related data."""
    amps: Optional[np.ndarray] = field(default=None)
    times: Optional[np.ndarray] = field(default=None)
    clusters: Optional[np.ndarray] = field(default=None)
    depths: Optional[np.ndarray] = field(default=None)

    @classmethod
    def get_file_mapping(cls) -> Dict[str, str]:
        return{
            'spikes.amps.npy': 'amps',
            'spikes.times.npy': 'times',
            'spikes.clusters.npy': 'clusters',
            'spikes.depths.npy': 'depths',
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the Spikes data to a pandas DataFrame.
        For multi-dimensional arrays, only the first dimension is used as the index,
        and the remaining dimensions are stored as array objects in the cells.
        """

        # Create DataFrame
        df = super().to_dataframe()
        df['clusters'] = df['clusters'].astype(int)
        return df


@dataclass
class Channels(BaseLoader):
    """Represents channel-related data."""
    brain_location: Optional[pd.DataFrame] = None
    site_positions: Optional[np.ndarray] = field(default=None)
    site: Optional[np.ndarray] = field(default=None)
    probe: Optional[np.ndarray] = field(default=None)
    raw_row: Optional[np.ndarray] = field(default=None)

    @classmethod
    def get_file_mapping(cls) -> Dict[str, str]:
        return {
            'channels.sitePositions.npy': 'site_positions',
            'channels.probe.npy': 'probe',
            'channels.site.npy': 'site',
            'channels.rawRow.npy': 'raw_row',
            'channels.brainLocation.tsv': 'brain_location',
        }

    @classmethod
    def from_tar(cls, tar_path: str | Path) -> 'BaseLoader':
        """Load data from a tar archive."""
        instance = cls()
        file_attr_map = cls.get_file_mapping()

        with tarfile.open(tar_path, 'r') as tar:
            for file_name, attr_name in file_attr_map.items():
                if file_name not in tar.getnames():
                    print(f"Warning: {file_name} not found in tar archive")
                    continue
                if attr_name == 'brain_location':
                    setattr(instance, attr_name, pd.read_csv(tar.extractfile(file_name), sep='\t'))
                else:
                    setattr(instance, attr_name, np.squeeze(npy_loader(tar, file_name)))
        return instance

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the Channels data to a pandas DataFrame.
        For multi-dimensional arrays, only the first dimension is used as the index,
        and the remaining dimensions are stored as array objects in the cells.
        """
        data_dict = {}
        base_length = None

        # Process each attribute
        for attr_name, value in self.__dict__.items():
            if attr_name == 'brain_location':
                continue
            elif value is not None:
                if len(value.shape) == 1:
                    # 1D arrays can be directly added
                    data_dict[attr_name] = value
                    if base_length is None:
                        base_length = len(value)
                else:
                    # For multi-dimensional arrays, store them as objects
                    # Each row will contain a slice of the array
                    data_dict[attr_name] = [value[i] for i in range(value.shape[0])]
                    if base_length is None:
                        base_length = value.shape[0]

        # Create DataFrame
        df = pd.concat([pd.DataFrame(data_dict),self.brain_location], axis=1, sort=False)

        return df

@dataclass
class Wheel(BaseLoader):
    """Represents wheel data."""
    times: Optional[np.ndarray] = field(default=None)
    positions: Optional[np.ndarray] = field(default=None)

    @classmethod
    def get_file_mapping(cls) -> Dict[str, str]:
        return {
            'wheel.timestamps.npy': 'times',
            'wheel.position.npy': 'positions',
        }

    @classmethod
    def from_tar(cls, tar_path: str | Path) -> 'BaseLoader':
        """Load data from a tar archive."""
        instance = cls()
        file_attr_map = cls.get_file_mapping()

        with tarfile.open(tar_path, 'r') as tar:
            for file_name, attr_name in file_attr_map.items():
                if file_name not in tar.getnames():
                    print(f"Warning: {file_name} not found in tar archive")
                    continue
                if attr_name == 'times':
                    setattr(instance, attr_name, np.squeeze(npy_loader(tar, file_name)))
                else:
                    setattr(instance, attr_name, np.squeeze(npy_loader(tar, file_name)))
        return instance

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the Wheel data to a pandas DataFrame.
        For multi-dimensional arrays, only the first dimension is used as the index,
        and the remaining dimensions are stored as array objects in the cells.
        """
        data_dict = {}
        base_length = None

        # Process each attribute
        for attr_name, value in self.__dict__.items():
            if attr_name == 'times':
                continue
            elif value is not None:
                if len(value.shape) == 1:
                    # 1D arrays can be directly added
                    data_dict[attr_name] = value
                    if base_length is None:
                        base_length = len(value)
                else:
                    # For multi-dimensional arrays, store them as objects
                    # Each row will contain a slice of the array
                    data_dict[attr_name] = [value[i] for i in range(value.shape[0])]
                    if base_length is None:
                        base_length = value.shape[0]

        # Create DataFrame
        df = pd.DataFrame(data_dict)
        df['times'] = np.linspace(self.times[0,1],self.times[1,1],int(self.times[1,0]+1))
        # Constants
        TICKS_PER_REV = 1440
        DEGREES_PER_TICK = 360 / TICKS_PER_REV

        # Compute angle position (modulo 1440 to stay within one revolution)
        df['angle_position'] = df['positions'].apply(
            lambda x: (x % TICKS_PER_REV) * DEGREES_PER_TICK
        )

        # Compute current revolution count (integer division)
        df['current_revolution'] = df['positions'].apply(
            lambda x: x // TICKS_PER_REV
        )

        # Compute direction change (+1 forward, -1 backward, 0 if no change)
        df['angle_change'] = np.diff(df['positions'],
                                                prepend=df['positions'][0]) * DEGREES_PER_TICK

        return df

@dataclass
class Session(BaseLoader):
    """Represents session data."""
    mouse_name: str = field(default=None)
    date_exp: str = field(default=None)
    cluster_df: Optional[pd.DataFrame] = field(default=None)
    spikes_df: Optional[pd.DataFrame] = field(default=None)
    trials_df: Optional[pd.DataFrame] = field(default=None)
    wheel_df: Optional[pd.DataFrame] = field(default=None)
    channels_df: Optional[pd.DataFrame] = field(default=None)
    quiescence_wheel_df: Optional[pd.DataFrame] = field(default=None)
    quiescence_spikes_df: Optional[pd.DataFrame] = field(default=None)

    @classmethod
    def get_file_mapping(cls) -> Dict[str, str]:
        return {
            'Clusters':'cluster_df',
            'Spikes':'spikes_df',
            'Trials':'trials_df',
            'Wheels':'wheel_df',
            'Channels':'channels_df',
        }

    @classmethod
    def from_tar(cls, tar_path: str | Path) -> 'Session':
        """Load data from a tar archive and return a Session instance."""
        instance = cls()
        file_attr_map = cls.get_file_mapping()

        setattr(instance, 'mouse_name', Path(tar_path).stem.split('_')[0])
        setattr(instance, 'date_exp', Path(tar_path).stem.split('_')[1])

        for file_name, attr_name in file_attr_map.items():
            if file_name == 'Clusters':
                session_data = Clusters.from_tar(tar_path)
                setattr(instance, attr_name, session_data.to_dataframe())
                instance.cluster_df = session_data.to_dataframe().assign(
                    mouse_name=instance.mouse_name,
                    date_exp=instance.date_exp
                )
            elif file_name == 'Spikes':
                session_data = Spikes.from_tar(tar_path)
                setattr(instance, attr_name, session_data.to_dataframe())
                instance.spikes_df = session_data.to_dataframe().assign(
                    mouse_name=instance.mouse_name,
                    date_exp=instance.date_exp
                )
            elif file_name == 'Trials':
                session_data = Trials.from_tar(tar_path)
                setattr(instance, attr_name, session_data.to_dataframe())
                instance.trials_df = session_data.to_dataframe().assign(
                    mouse_name=instance.mouse_name,
                    date_exp=instance.date_exp
                )
            elif file_name == 'Wheels':
                session_data = Wheel.from_tar(tar_path)
                setattr(instance, attr_name, session_data.to_dataframe())
                instance.wheel_df = session_data.to_dataframe().assign(
                    mouse_name=instance.mouse_name,
                    date_exp=instance.date_exp
                )
            elif file_name == 'Channels':
                session_data = Channels.from_tar(tar_path)
                setattr(instance, attr_name, session_data.to_dataframe())
                instance.channels_df = session_data.to_dataframe().assign(
                    mouse_name=instance.mouse_name,
                    date_exp=instance.date_exp
                )

        closed_sites = instance.channels_df.iloc[np.where(
            np.diff(instance.channels_df.site.values) != 1)[0]].site.values + 1
        instance.cluster_df.loc[instance.cluster_df.site.isin(closed_sites), 'site'] -= 1
        instance.spikes_df = instance.spikes_df.merge(instance.cluster_df,
                                                    left_on='clusters',
                                                    right_index=True,
                                                    how='left',suffixes=("","_del")).query('phy_annotation != 1')

        spikes_sorted = instance.spikes_df.sort_values('times')
        trials_sorted = instance.trials_df.sort_values('trial_start')
        wheel_sorted = instance.wheel_df.sort_values('times')

        # Drop columns only if they exist
        columns_to_drop = ['mouse_name_del', 'date_exp_del', 'depths_del']



        # Assign the closest trial start to each spike and wheel position
        spikes_with_intervals = pd.merge_asof(spikes_sorted,
                                              trials_sorted,
                                              left_on='times',
                                              right_on='trial_start',
                                              direction='backward',
                                              suffixes=("", "_del")).dropna()


        wheel_with_intervals = pd.merge_asof(wheel_sorted,
                                             trials_sorted,
                                             left_on='times',
                                             right_on='trial_start',
                                             direction='backward',
                                             suffixes=("", "_del"))

        # Keep only spikes within their assigned interval
        spikes_with_intervals = spikes_with_intervals[spikes_with_intervals['times'] <= spikes_with_intervals['trial_end']]
        wheel_with_intervals = wheel_with_intervals[wheel_with_intervals['times'] <= wheel_with_intervals['trial_end']]


        # Repeat for quiescence periods
        quiescence_trials_sorted = instance.trials_df.sort_values('quiescence_start')

        quiescence_spikes_with_intervals = pd.merge_asof(spikes_sorted,
                                                         quiescence_trials_sorted,
                                                         left_on='times',
                                                         right_on='quiescence_start',
                                                         direction='backward',
                                                         suffixes=("", "_del")).dropna()

        quiescence_wheel_with_intervals = pd.merge_asof(wheel_sorted,
                                                        quiescence_trials_sorted,
                                                        left_on='times',
                                                        right_on='quiescence_start',
                                                        direction='backward',
                                                        suffixes=("", "_del")).dropna()

        quiescence_spikes_with_intervals = quiescence_spikes_with_intervals[
            quiescence_spikes_with_intervals['times'] <= quiescence_spikes_with_intervals['quiescence_end']]

        quiescence_wheel_with_intervals = quiescence_wheel_with_intervals[
            quiescence_wheel_with_intervals['times'] <= quiescence_wheel_with_intervals['quiescence_end']]


        instance.spikes_df = spikes_with_intervals.drop(
            columns=[col for col in columns_to_drop if col in spikes_with_intervals.columns])

        instance.wheel_df = wheel_with_intervals.drop(
            columns=[col for col in columns_to_drop if col in wheel_with_intervals.columns])

        instance.quiescence_spikes_df = quiescence_spikes_with_intervals.drop(
            columns=[col for col in columns_to_drop if col in quiescence_spikes_with_intervals.columns])

        instance.quiescence_wheel_df = quiescence_wheel_with_intervals.drop(
            columns=[col for col in columns_to_drop if col in quiescence_wheel_with_intervals.columns])


        # instance.cluster_df = instance.cluster_df.merge(instance.channels_df[['probe', 'site', 'ccf_ap', 'ccf_dv', 'ccf_lr', 'allen_ontology']],
        #                               on=['probe', 'site'], how='left')

        return instance