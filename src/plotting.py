import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def binarize_spikes(spike_data, threshold=0):
    """
    Binarize spike data by converting to binary events.

    Parameters:
    spike_data (np.ndarray): 2D array of spike data
    threshold (float): Threshold for spike detection

    Returns:
    list of lists: Binary spike times for each trial
    """
    binary_spikes = []
    time_points = np.arange(-495, 2005, 10)

    for trial in spike_data:
        # Find indices where spikes occur (above threshold)
        spike_indices = np.where(trial > threshold)[0]

        # Convert indices to actual time points
        spike_times = time_points[spike_indices]
        binary_spikes.append(spike_times)

    return binary_spikes

def plot_event_contrast_grid(data_df: pd.DataFrame, area: str):
    """
    Create a 2x2 grid of event plots with their PSTHs, sorted by contrast difference
    using matplotlib

    Parameters:
    data_df: DataFrame containing trial data
    area: str, the brain area to filter by
    neuron_no: neuron identifier

    Returns:
    matplotlib Figure object
    """

    sort_df = data_df.sort_values('contrast_pair', ascending=True)
    unique_pairs = sort_df['contrast_pair'].unique()

    diffs = np.squeeze(np.array([np.diff(x) for x in unique_pairs]) != 0)

    # Define readable colors from matplotlib's named colors
    color_list = [
        'red', 'blue', 'red', 'green', 'blue',
        'green']

    # Create a dictionary mapping unique combinations to colors
    color_dict = {combo: color for combo, color in zip(unique_pairs[np.squeeze(diffs)], color_list)}
    print(color_dict)

    # Define parameters and filter conditions
    titles_list = ['Left Contrast Reward', 'Left Contrast Penalty', 'Right Contrast Reward', 'Right Contrast Penalty']
    right_response = -1
    left_response = 1
    correct_feedback = 1
    incorrect_feedback = -1

    # Color palette
    stimulus_color = 'red'
    colors = color_dict

    left_correct_reward = data_df.query(f'(contrast_left > contrast_right) & (response == @left_response) & (feedback_type == @correct_feedback)')
    left_correct_penalty = data_df.query('(contrast_left > contrast_right) & (response == @right_response) & (feedback_type == @incorrect_feedback)')
    right_correct_reward = data_df.query('(contrast_left < contrast_right) & (response == @right_response) & (feedback_type == @correct_feedback)')
    right_correct_penalty = data_df.query('(contrast_left < contrast_right) & (response == @left_response) & (feedback_type == @incorrect_feedback)')


    df_list = [left_correct_reward, left_correct_penalty, right_correct_reward, right_correct_penalty]
    df_filtered = [x.query(f'brain_area == "{area}"') for x in df_list]

    # Sort each DataFrame by contrast difference and store contrast differences
    sorted_df_filtered = []
    contrasts_list = []
    for i, df in enumerate(df_filtered):
        sorted_df = df.sort_values('contrast_pair', ascending=True)
        sorted_df_filtered.append(sorted_df)
        contrasts_list.append(sorted_df['contrast_pair'].values)

    spikes_list = [np.array(df.spks.to_list()) for df in sorted_df_filtered]

    avg_speed = [np.array(df.average_pupil_speed.to_list()) for df in sorted_df_filtered]

    wheel_speed = [np.array(df.wheel.to_list()) for df in sorted_df_filtered]

    pupil_area = [np.array(df.pupil_area.to_list()) for df in sorted_df_filtered]



    # Time points for x-axis
    time_points = np.arange(-495, 2005, 10)

    # Create figure
    fig = plt.figure(figsize=(15, 42))  # Increased height to accommodate title
    gs = fig.add_gridspec(10, 2, height_ratios=[0.7, 0.5, 0.5, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5])
    axs = []
    for i in range(10):
        for j in range(2):
            axs.append(fig.add_subplot(gs[i, j]))
    axs = np.array(axs).reshape(10, 2)

    # Add figure title with proper spacing
    fig.suptitle(f'{area} Spike Train Event Plots with PSTH',
                fontsize=14,
                y=0.98)  # Adjusted y position for better spacing

    psth_ylims = []
    wheel_ylims = []
    speed_ylims = []
    pupil_area_ylims = []
    df_contrast_pair = left_correct_reward.contrast_pair.unique()
    for idx,(spike_data, pupil_speed, pupil_size, wheel, df) in enumerate(zip(spikes_list, avg_speed, pupil_area, wheel_speed, sorted_df_filtered)):

        contrast_pair_array = df.contrast_pair.to_numpy()

        changes = np.where(contrast_pair_array[:-1] != contrast_pair_array[1:])[0].astype(int)  + 1




        spike_colors = []
        color_map ={}

        for id, c_val  in enumerate(df.contrast_pair.unique()):
            color_map[c_val] = colors[c_val]
            if id == 0:
                spike_colors = spike_colors + [colors[c_val] for _ in range(changes[id])]

            elif id == len(df.contrast_pair.unique()) - 1:
                spike_colors = spike_colors + [colors[c_val] for _ in range(len(contrast_pair_array) - changes[id-1])]

            else:
                spike_colors = spike_colors + [colors[c_val] for _ in range(changes[id] - changes[id-1])]


        if idx < 2:
            raster_ax = axs[idx*5, 0]
            psth_ax = axs[(idx*5)+1, 0]
            wheel_ax = axs[(idx*5)+2, 0]
            speed_ax = axs[(idx*5)+3, 0]
            pupil_area_ax = axs[(idx*5)+4, 0]

            raster_ax.set_ylabel('Trials')
            psth_ax.set_ylabel('Firing rate (Hz)')
            wheel_ax.set_ylabel('Speed')
            speed_ax.set_ylabel('Speed (m/s) #Placeholder Dimensions') #Placeholder dimensions
        elif idx >= 2:
            raster_ax = axs[(idx-2)*5, 1]
            psth_ax = axs[((idx-2)*5)+1, 1]
            wheel_ax = axs[((idx-2)*5)+2, 1]
            speed_ax = axs[((idx-2)*5)+3, 1]
            pupil_area_ax = axs[((idx-2)*5)+4, 1]

        # Remove grid lines
        raster_ax.grid(False)
        psth_ax.grid(False)
        wheel_ax.grid(False)
        speed_ax.grid(False)
        pupil_area_ax.grid(False)

        # If insufficient data
        if spike_data.shape[0] < 10:
            raster_ax.text(0.5, 0.5, f"Not enough trial data",
                           horizontalalignment='center', verticalalignment='center')
            psth_ax.text(0.5, 0.5, f"Not enough trial data",
                         horizontalalignment='center', verticalalignment='center')
            continue

        # Binarize spike data
        binary_spikes = binarize_spikes(spike_data)

        # Eventplot (raster equivalent)
        raster_ax.eventplot(binary_spikes, color=spike_colors, linewidths=1)

        # Stimulus onset line
        raster_ax.axvline(x=0, color=stimulus_color, linestyle='--', linewidth=2, label ='stimulus onset')
        psth_ax.axvline(x=0, color=stimulus_color, linestyle='--', linewidth=2)
        wheel_ax.axvline(x=0, color=stimulus_color, linestyle='--', linewidth=2)
        speed_ax.axvline(x=0, color=stimulus_color, linestyle='--', linewidth=2)
        pupil_area_ax.axvline(x=0, color=stimulus_color, linestyle='--', linewidth=2)

        psth_lim = []
        wheel_lim = []
        speed_lim = []
        pupil_area_lim = []
        for id,(change, (k_, v_)) in enumerate(zip(changes, color_map.items())):
            contrast_key = list(color_map.keys())[-1]
            if len(changes) == 1:
                    # PSTH
                    psth = np.mean(spike_data[:change], axis=0) / (10 / 1000)
                    psth_ax.plot(time_points, psth, color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')
                    psth_lim.append(np.max(psth))
                    psth = np.mean(spike_data[change:], axis=0) / (10 / 1000)
                    psth_ax.plot(time_points, psth, color=list(color_map.values())[1], linewidth=2, label= f'contrast_pair {str(list(color_map.keys())[1])}')
                    psth_lim.append(np.max(psth))

                    # Wheel
                    wheel_ax.plot(time_points, np.abs(np.mean(wheel[:change], axis=0)), color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')
                    wheel_lim.append(np.max(np.abs(np.mean(wheel[:change]))))
                    wheel_ax.plot(time_points, np.abs(np.mean(wheel[change:], axis=0)), color=list(color_map.values())[1], linewidth=2, label= f'contrast_pair {str(list(color_map.keys())[1])}')
                    wheel_lim.append(np.max(np.abs(np.mean(wheel[change:]))))

                    # Average Pupil Speed
                    speed_ax.plot(time_points[1:], np.median(pupil_speed[:change], axis=0), color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')
                    speed_lim.append(np.max(np.median(pupil_speed[:change])))
                    speed_ax.plot(time_points[1:], np.median(pupil_speed[change:], axis=0), color=list(color_map.values())[1], linewidth=2, label= f'contrast_pair {str(list(color_map.keys())[1])}')
                    speed_lim.append(np.max(np.median(pupil_speed[:change])))

                    # Average Pupil Area Change
                    pupil_area_ax.plot(time_points, np.median(pupil_size[:change], axis=0), color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')
                    pupil_area_lim.append(np.max(np.median(pupil_size[:change])))
                    pupil_area_ax.plot(time_points, np.median(pupil_size[change:], axis=0), color=list(color_map.values())[1], linewidth=2, label= f'contrast_pair {str(list(color_map.keys())[1])}')
                    pupil_area_lim.append(np.max(np.median(pupil_size[change:])))

            else:
                if id == 0:
                    # PSTH
                    psth = np.mean(spike_data[:change], axis=0) / (10 / 1000)
                    psth_ax.plot(time_points, psth, color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')

                    psth_lim.append(np.max(psth))

                    # Wheel
                    wheel_ax.plot(time_points, np.abs(np.mean(wheel[:change], axis=0)), color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')
                    wheel_lim.append(np.max(np.abs(np.mean(wheel[:change]))))

                    # Average Pupil Speed
                    speed_ax.plot(time_points[1:], np.median(pupil_speed[:change], axis=0), color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')
                    speed_lim.append(np.max(np.median(pupil_speed[:change])))

                    # Average Pupil Area Change
                    pupil_area_ax.plot(time_points, np.median(pupil_size[:change], axis=0), color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')
                    pupil_area_lim.append(np.max(np.median(pupil_size[:change])))

                elif id == len(changes)-1:
                    psth_pre = np.mean(spike_data[changes[id-1]:change], axis=0) / (10 / 1000)
                    psth =np.mean(spike_data[change:], axis=0) / (10 / 1000)
                    # PSTH
                    psth_ax.plot(time_points, psth_pre, color=v_, linewidth=2, label=f'contrast_pair {str(k_)}')
                    psth_ax.plot(time_points, psth, color=colors[contrast_key], linewidth=2, label= f'contrast_pair {str(contrast_key)}')
                    psth_lim.append(np.max(psth_pre))
                    psth_lim.append(np.max(psth))

                    # Wheel
                    wheel_ax.plot(time_points, np.abs(np.mean(wheel[changes[id-1]:change], axis=0)), color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')
                    wheel_ax.plot(time_points, np.abs(np.mean(wheel[change:], axis=0)), color=colors[contrast_key], linewidth=2, label= f'contrast_pair {contrast_key}')
                    wheel_lim.append(np.max(np.abs(np.mean(wheel[changes[id-1]:change], axis=0))))
                    wheel_lim.append(np.max(np.abs(np.mean(wheel[change:],axis=0))))

                    # Average Pupil Speed
                    speed_ax.plot(time_points[1:], np.median(pupil_speed[changes[id-1]:change], axis=0), color=v_, linewidth=2, label = f'contrast_pair {str(k_)}')
                    speed_ax.plot(time_points[1:], np.median(pupil_speed[change:], axis=0), color=colors[contrast_key], linewidth=2, label= f'contrast_pair {contrast_key}')
                    speed_lim.append(np.max(np.median(pupil_speed[changes[id-1]:change],axis=0)))
                    speed_lim.append(np.max(np.median(pupil_speed[change:],axis=0)))

                    # Average Pupil Area Change
                    pupil_area_ax.plot(time_points, np.median(pupil_size[changes[id-1]:change], axis=0), color=v_, linewidth=2, label = f'contrast_pair {str(k_)}')
                    pupil_area_ax.plot(time_points, np.median(pupil_size[change:], axis=0), color=colors[contrast_key], linewidth=2, label= f'contrast_pair {contrast_key}')
                    pupil_area_lim.append(np.max(np.median(pupil_size[changes[id-1]:change],axis=0)))
                    pupil_area_lim.append(np.max(np.median(pupil_size[change:],axis=0)))

                else:
                    psth = np.mean(spike_data[changes[id-1]:change], axis=0) / (10 / 1000)
                    psth_ax.plot(time_points, np.mean(spike_data[changes[id-1]:change], axis=0) / (10 / 1000), color=v_, linewidth=2, label=f'contrast_pair {str(k_)}')
                    psth_lim.append(np.max(psth))

                    wheel_ax.plot(time_points, np.abs(np.mean(wheel[changes[id-1]:change], axis=0)), color=v_, linewidth=2, label= f'contrast_pair {str(k_)}')
                    wheel_lim.append(np.max(np.abs(np.mean(wheel[changes[id-1]:change],axis=0))))

                    speed_ax.plot(time_points[1:], np.median(pupil_speed[changes[id-1]:change], axis=0), color=v_, linewidth=2, label = f'contrast_pair {str(k_)}')
                    speed_lim.append(np.max(np.median(pupil_speed[changes[id-1]:change],axis=0)))

                    pupil_area_ax.plot(time_points, np.median(pupil_size[changes[id-1]:change], axis=0), color=v_, linewidth=2, label = f'contrast_pair {str(k_)}')
                    pupil_area_lim.append(np.max(np.median(pupil_size[changes[id-1]:change],axis=0)))

        psth_ylims.append(np.max(np.array(psth_lim)))
        wheel_ylims.append(np.max(np.array(wheel_lim)))
        speed_ylims.append(np.max(np.array(speed_lim)))
        pupil_area_ylims.append(np.max(np.array(pupil_area_lim)))



        # Axis labels and titles
        raster_ax.set_title(titles_list[idx])

        psth_ax.set_title('PSTH')

        wheel_ax.set_title('Average Wheel Speed Across Trials')

        speed_ax.set_title('Average Pupil Speed Across Trials')

        pupil_area_ax.set_title('Pupil Area Change Across Trials')
        pupil_area_ax.set_xlabel('Time (ms)')



        # X-axis limits
        raster_ax.set_xlim(-500, 2000)
        psth_ax.set_xlim(-500, 2000)
        wheel_ax.set_xlim(-500, 2000)
        speed_ax.set_xlim(-500, 2000)
        pupil_area_ax.set_xlim(-500, 2000)

        # Set y-axis limits for
        raster_ax.set_ylim(-1, len(binary_spikes))


        #Legends
        raster_ax.legend(loc='upper right')
        psth_ax.legend(loc='upper right')
        wheel_ax.legend(loc='upper right')
        speed_ax.legend(loc='upper right')
        pupil_area_ax.legend(loc='upper right')
        wheel_ax.legend(loc='upper right')

    axs[1,0].set_ylim(0, np.ceil(max(psth_ylims[::2])))
    axs[6,0].set_ylim(0, np.ceil(max(psth_ylims[1::2])))
    axs[1,1].set_ylim(0, np.ceil(max(psth_ylims[::2])))
    axs[6,1].set_ylim(0, np.ceil(max(psth_ylims[1::2])))


    axs[2,0].set_ylim(0, np.ceil(max(wheel_ylims[::2])))
    axs[7,0].set_ylim(0, np.ceil(max(wheel_ylims[1::2])))
    axs[2,1].set_ylim(0, np.ceil(max(wheel_ylims[::2])))
    axs[7,1].set_ylim(0, np.ceil(max(wheel_ylims[1::2])))

    axs[3,0].set_ylim(0, np.ceil(max(speed_ylims[::2])))
    axs[8,0].set_ylim(0, np.ceil(max(speed_ylims[1::2])))
    axs[3,1].set_ylim(0, np.ceil(max(speed_ylims[::2])))
    axs[8,1].set_ylim(0, np.ceil(max(speed_ylims[1::2])))

    axs[4,0].set_ylim(0, max(pupil_area_ylims[::2])+0.01)
    axs[9,0].set_ylim(0, max(pupil_area_ylims[1::2])+0.01)
    axs[4,1].set_ylim(0, max(pupil_area_ylims[::2])+0.01)
    axs[9,1].set_ylim(0, max(pupil_area_ylims[1::2])+0.01)

    fig.suptitle(f'{area}  Spike Train PSTH, Events contrast pair {df_contrast_pair}', fontsize=12)
    # Replace tight_layout with manual spacing adjustment
    plt.subplots_adjust(
        top=0.96,      # Increased to leave space for title
        bottom=0.05,
        left=0.1,
        right=0.9,
        hspace=0.4,    # Adjusted for better vertical spacing between subplots
        wspace=0.3     # Space between columns
    )

    return fig

def plot_psth_response_rank(df, figure_title):
    df = df.query('brain_area != "root"')

    # Brain Region Information
    regions = ["vis ctx", "thal", "hipp", "other ctx", "midbrain", "basal ganglia", "cortical subplate", "other"]
    region_colors = ['blue', 'red', 'green', 'darkblue', 'violet', 'lightblue', 'orange', 'gray']

    brain_groups = [
        ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],  # visual cortex
        ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"],  # thalamus
        ["CA", "CA1", "CA2", "CA3", "DG", "SUB", "POST"],  # hippocampus
        ["ACA", "AUD", "COA", "DP", "ILA", "MOp", "MOs", "OLF", "ORB", "ORBm", "PIR", "PL", "SSp", "SSs", "RSP", "TT"],  # non-visual cortex
        ["APN", "IC", "MB", "MRN", "NB", "PAG", "RN", "SCs", "SCm", "SCig", "SCsg", "ZI"],  # midbrain
        ["ACB", "CP", "GPe", "LS", "LSc", "LSr", "MS", "OT", "SNr", "SI"],  # basal ganglia
        ["BLA", "BMA", "EP", "EPd", "MEA"]  # cortical subplate
    ]

    # Create region-to-color dictionary
    region_color_dict = dict(zip(regions, region_colors))
    brain_region_color_dict = {subregion: region_color_dict[region] for region, group in zip(regions, brain_groups) for subregion in group}

    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 20))
    grid_widths = [0.15, 1, 0.3]
    gs = plt.GridSpec(2, 9, width_ratios=grid_widths * 3)

    im_list = []  # Store imshow plots for colorbar

    for i, pair in enumerate(df.contrast_pair.unique()[::-1]):
        data_df = df.query('contrast_pair == @pair').sort_values('spks_latency_response')
        data = np.stack(data_df.spks_psth.values)[:, 50:]
        response_latencies = np.stack(data_df.spks_latency_response.values)
        brain_areas = np.stack(data_df.brain_area.values)
        neuron_count = data_df.neuron_id_neuron_count.values

        row = 0 if i < 3 else 1
        label_column = i * 3 if i < 3 else (i * 3) - 9
        data_column = label_column + 1
        neuron_count_column = label_column + 2

        # Brain region labels
        ax_labels = plt.subplot(gs[row, label_column])
        ax_labels.set_xlim(0, 1)
        ax_labels.set_ylim(0, len(brain_areas))

        for j, area in enumerate(brain_areas):
            ax_labels.text(0.5, len(brain_areas) - j - 0.5, area, ha='right', va='center', color=brain_region_color_dict[area])

        ax_labels.axis('off')

        # Activity plot
        ax_data = plt.subplot(gs[row, data_column])
        imshow_data =(data / np.mean(data, axis=1).reshape(data.shape[0], 1))
        im = ax_data.imshow(imshow_data[:,:30], aspect='auto', cmap='inferno')
        im_list.append(im)  # Store for colorbar
        ax_data.set_xlabel('Time from stimulus (ms)')
        ax_data.set_yticks([])
        ax_data.xlim(0,300)
        ax_data.set_xticks(np.arange(0, 300, 30))
        ax_data.set_title(f'Contrast {pair}')
        ax_data.grid(False)

        # Horizontal lines between region groups
        for j in range(len(data)):
            ax_data.hlines(y=j + 0.5, xmin=-0.5, xmax=199.5, color='gray', linestyle='-', alpha=0.5)

        # Neuron count plot
        ax_neuron = plt.subplot(gs[row, neuron_count_column])
        y_positions = np.arange(len(brain_areas))
        # ax_neuron.barh(y_positions, neuron_count, color='gray')
        ax_neuron.barh(y_positions, response_latencies, color='gray')
        ax_neuron.set_yticks(y_positions)
        ax_neuron.set_yticklabels([])
        ax_neuron.set_ylim(len(brain_areas) - 0.5, -0.5)
        ax_neuron.set_xlim(0, 900)
        # ax_neuron.set_title('Number of Neurons')
        ax_neuron.set_title('Latency responses')

    # Add a single colorbar at the bottom left (horizontal)
    cbar_ax = fig.add_axes([0.85, 0.95, 0.1, 0.01])  # [left, bottom, width, height]
    cbar = fig.colorbar(im_list[-1], cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Average Normalized Firing Rate')


    fig.suptitle(figure_title)
    plt.show()
