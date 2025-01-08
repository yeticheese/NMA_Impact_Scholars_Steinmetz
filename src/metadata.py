from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from enum import IntEnum

class MovementType(IntEnum):
    FLINCH = 0
    LEFT = 1
    RIGHT = 2

class ResponseChoice(IntEnum):
    RIGHT = -1
    NOGO = 0
    LEFT = 1

@dataclass
class Eye:
    area: np.ndarray  # [arb. units] (nFrames)
    blink: np.ndarray  # [logical] (nFrames)
    xy_pos: np.ndarray  # [arb. units] (nFrames,2)
    timestamps: np.ndarray

@dataclass
class Face:
    motion_energy: np.ndarray  # [arb. units] (nFrames)
    timestamps: np.ndarray

@dataclass
class LickPiezo:
    raw: np.ndarray  # [V] (nSamples)
    timestamps: np.ndarray
    lick_times: np.ndarray  # (nLicks)

@dataclass
class Wheel:
    position: np.ndarray  # [encoder ticks] (nSamples)
    timestamps: np.ndarray

@dataclass
class WheelMove:
    type: np.ndarray  # MovementType enum (nDetectedMoves)
    intervals: np.ndarray  # (nDetectedMoves, 2)

@dataclass
class Trial:
    feedback_type: np.ndarray  # [-1, 1] (nTrials)
    feedback_times: np.ndarray
    go_cue_times: np.ndarray
    included: np.ndarray  # [logical] (nTrials)
    rep_num: np.ndarray  # [integer] (nTrials)
    response_choice: np.ndarray  # ResponseChoice enum (nTrials)
    response_times: np.ndarray
    contrast_left: np.ndarray  # [proportion] (nTrials)
    contrast_right: np.ndarray  # [proportion] (nTrials)
    visual_stim_times: np.ndarray
    intervals: np.ndarray  # (nTrials, 2)

@dataclass
class SparseNoise:
    positions: np.ndarray  # [degrees] (nStimuli, 2)
    times: np.ndarray

@dataclass
class PassiveStimuli:
    beep_times: np.ndarray
    valve_click_times: np.ndarray
    visual_contrast_left: np.ndarray
    visual_contrast_right: np.ndarray
    visual_times: np.ndarray
    white_noise_times: np.ndarray

@dataclass
class Channel:
    ccf_ap: float  # [µm]
    ccf_dv: float  # [µm]
    ccf_lr: float  # [µm]
    allen_ontology: str
    probe_index: int
    raw_row: int
    site: int
    site_position: Tuple[float, float]  # [µm] (x, y)

@dataclass
class Cluster:
    phy_annotation: int  # [0=noise, 1=MUA, 2=Good, 3=Unsorted]
    depth: float  # [µm]
    original_id: int
    peak_channel: int
    probe: int
    template_waveform_channels: np.ndarray  # (50,)
    template_waveforms: np.ndarray  # (82, 50)
    waveform_duration: float  # [s]

@dataclass
class Probe:
    description: str
    entry_point_rl: float
    entry_point_ap: float
    vertical_angle: float
    horizontal_angle: float
    axial_angle: float
    distance_advanced: float
    raw_filename: str
    site_positions: np.ndarray  # (nSites, 2)

@dataclass
class Spike:
    amp: float  # [µV]
    cluster: int
    depth: float  # [µm]
    time: float

@dataclass
class Session:
    eye: Eye
    face: Face
    lick_piezo: LickPiezo
    wheel: Wheel
    wheel_moves: WheelMove
    trials: Trial
    sparse_noise: SparseNoise
    passive_stimuli: PassiveStimuli
    channels: List[Channel]
    clusters: List[Cluster]
    probes: List[Probe]
    spikes: List[Spike]