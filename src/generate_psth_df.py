import argparse
import numpy as np
import pandas as pd
from src.session_metadata import Session
import os

data_directory = r'data\spikeAndBehavioralData'

def trial_subroutines(key, value):
    match key:
        case 'response':
            return value
        case 'response_time':
            return np.squeeze(value)
        case 'contrast_left':
            return value
        case 'contrast_right':
            return value
        case 'feedback_time':
            return np.squeeze(value)
        case 'feedback_type':
            return value
        case 'gocue':
            return np.squeeze(value)
        case 'prev_reward':
            return np.squeeze(value)
        case _:
            return None

def neuron_subroutines(key, value):
    match key:
        case 'brain_area':
            return value
        case _:
            return None

def dat_subroutines(dat):
    session_data = Session.from_tar(os.path.join(os.getcwd(),data_directory,dat['mouse_name']+"_"+dat['date_exp']+".tar"))
    session_data.cluster_df = session_data.cluster_df.loc[dat['cellid_orig']]

    cluster_subset = session_data.cluster_df[['depths', 'site', 'probe', 'template_waveforms',
                             'waveform_duration', 'peak_to_trough_duration',
                             'mouse_name', 'date_exp']]

    neuron_data = {}
    trial_data = {}
    for key, value in dat.items():
        neuron_result = neuron_subroutines(key, value)
        trial_result = trial_subroutines(key, value)

        # Only add to dictionary if the result is not
        if neuron_result is not None:
            neuron_data[key] = neuron_result
        if trial_result is not None:
            trial_data[key] = trial_result


    # Create DataFrames from the processed data
    neuron_df = pd.DataFrame(neuron_data)
    trial_df = pd.DataFrame(trial_data)


    # Add common fields
    for df in [neuron_df, trial_df]:
        if 'mouse_name' in dat:
            df['mouse_name'] = dat['mouse_name']
        if 'date_exp' in dat:
            df['date_exp'] = pd.to_datetime(dat['date_exp'])

    neuron_df[[*dat['ccf_axes']]] = dat['ccf']
    trial_df[['reaction_time','reaction_type']] = dat['reaction_time']

    trial_df['pupil_area'] = list(dat['pupil'][0,:,:])
    trial_df['pupil_x'] = list(dat['pupil'][1,:,:])
    trial_df['pupil_y'] = list(dat['pupil'][2,:,:])
    trial_df['face'] = list(np.squeeze(dat['face']))
    trial_df['licks'] = list(np.squeeze(dat['licks']))
    trial_df['wheel'] = list(np.squeeze(dat['wheel']))
    trial_df['average_pupil_speed'] = list(np.sqrt(((np.diff(dat['pupil'][1,:,:], axis=1)/0.01)**2)+((np.diff(dat['pupil'][2,:,:], axis=1)/0.01)**2)))

    # Add index columns
    neuron_df['neuron_id'] = range(len(neuron_df))
    trial_df['trial_id'] = range(len(trial_df))

    # Create cross join between neurons and trials
    # First, add a temporary key for merging
    neuron_df['_merge_key'] = 1
    trial_df['_merge_key'] = 1

    # Perform the merge
    merged_df = pd.merge(
        neuron_df,
        trial_df,
        on='_merge_key',
        suffixes=('_neuron', '_trial')
    ).drop('_merge_key', axis=1)

    # Deduplicate common columns (mouse_name and date_exp)
    for col in merged_df.columns:
        if col.endswith('_neuron') and col.replace('_neuron', '_trial') in merged_df.columns:
            base_col = col.replace('_neuron', '')
            if (merged_df[col] == merged_df[col.replace('_neuron', '_trial')]).all():
                merged_df[base_col] = merged_df[col]
                merged_df = merged_df.drop([col, col.replace('_neuron', '_trial')], axis=1)

    # Reset the index
    merged_df = merged_df.reset_index(drop=True).merge(cluster_subset, left_on='neuron_id', right_index=True, how='left', suffixes=('', '_del'))
    spikes = dat['spks'].reshape(-1,250)

    merged_df['spks'] = list(spikes)

    columns_to_drop = ['mouse_name_del', 'date_exp_del']

    merged_df = merged_df.drop(columns=[col for col in columns_to_drop if col in merged_df.columns])

    return merged_df


def process_data(alldat):
    dfs = [dat_subroutines(dat) for dat in alldat]

    concat_df = pd.concat(dfs, ignore_index=True)

    dfs_select = []
    for area in concat_df.brain_area.unique():
        selected_df = concat_df.query(f'brain_area == "{area}"')
        # Create a mapping dictionary from old to new IDs
        unique_neurons = selected_df.neuron_id.unique()
        new_neuron_ids = np.arange(len(unique_neurons))
        id_mapping = dict(zip(unique_neurons, new_neuron_ids))

        # Map old IDs to new IDs
        selected_df['neuron_id'] = selected_df['neuron_id'].map(id_mapping)
        dfs_select.append(selected_df)

    concat_df = pd.concat(dfs_select)

    # Convert unordered pairs to ordered pairs
    concat_df['contrast_pair'] = concat_df.apply(
        lambda row: tuple(sorted([row['contrast_left'], row['contrast_right']], reverse=True)), axis=1)

    concat_df['trial_outcome'] = concat_df.apply(lambda row: 'left_reward' if (row['contrast_left'] > row['contrast_right']) & (row['response'] == 1) & (row['feedback_type'] == 1)
    else 'right_reward' if (row['contrast_right'] > row['contrast_left']) & (row['response'] == -1) & (row['feedback_type'] == 1)
                                                 else 'left_penalty' if (row['contrast_left'] > row['contrast_right']) & (row['response'] == -1) & (row['feedback_type'] == -1)
                                                 else 'right_penalty'  if (row['contrast_right'] > row['contrast_left']) & (row['response'] == 1) & (row['feedback_type'] == -1)
                                                 else 'nogo',
                                                 axis=1)
    concat_df['neuron_type'] = concat_df['peak_to_trough_duration'].apply(
        lambda x: 'inhibitory' if x < 0.5 else 'excitatory')

    return concat_df
