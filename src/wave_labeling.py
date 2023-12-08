import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy.signal import savgol_filter, find_peaks

def load_ecg_data(file_path):
    with open(file_path, "r") as in_data:
        full_dataset = pd.read_csv(in_data)
    return pd.Series(full_dataset['ECG_I_filtered'].values)

def process_ecg_data(file_path):
    ecg_data = load_ecg_data(file_path)
    smoothed_heartbeats = scipy.signal.savgol_filter(ecg_data, window_length=20, polyorder=2)

    # Find the R waves' peaks
    r_waves, _ = scipy.signal.find_peaks(ecg_data, height=0.275)

    waves_list = []

    index_range = len(smoothed_heartbeats) - 1
    r_waves_indexed = enumerate(r_waves)

    # Total time duration (in seconds)
    total_duration = 20

    # Total number of data points
    total_data_points = 10000

    # Calculate the time increment per data point
    time_increment = total_duration / total_data_points

    # Time values based on the index
    time_values = np.round(np.arange(len(ecg_data)) * time_increment, decimals=4)

    # Loop through the RR intervals
    for (index, r) in r_waves_indexed:
        # Get the RR interval
        start_index = r  # the index of the current R wave
        try:
            end_index = r_waves[index + 1]  # the index of the next R wave
        except IndexError:
            end_index = index_range  # have hit the end of the heartbeat array
        temp_rr_interval = smoothed_heartbeats[start_index:end_index]

        p_and_t_waves, metadata = scipy.signal.find_peaks(temp_rr_interval,
                                                          height=[-0.1, 0.325],
                                                          distance=200)

        if not p_and_t_waves.any():
            p_wave = q_wave = r_wave = s_wave = t_wave = None
        else:
            p_wave = p_and_t_waves[-1]
            t_wave = p_and_t_waves[0]

            # Label the Q wave
            q_area = temp_rr_interval[-16:]
            q_area = pd.Series(q_area)
            q_wave = q_area.idxmin(axis=0)

            # Label the S wave
            s_area = temp_rr_interval[0:16]
            s_area = pd.Series(s_area)
            s_wave = s_area.idxmin(axis=0)

            r_wave = len(temp_rr_interval) // 2

            # Increment index all of the newly found waves
            q_wave += r + len(temp_rr_interval) - 15
            s_wave += r

            # Add the waves to the list
            waves_list.extend([
                {'ECG_I_filtered': smoothed_heartbeats[i], 'wave_label': 'p'} for i in range(max(0, p_wave - 1), min(len(smoothed_heartbeats), p_wave + 2))
            ])
            waves_list.extend([
                {'ECG_I_filtered': smoothed_heartbeats[i], 'wave_label': 'q'} for i in range(max(0, q_wave - 1), min(len(smoothed_heartbeats), q_wave + 2))
            ])
            waves_list.extend([
                {'ECG_I_filtered': smoothed_heartbeats[i], 'wave_label': 'r'} for i in range(max(0, r + r_wave - 1), min(len(smoothed_heartbeats), r + r_wave + 2))
            ])
            waves_list.extend([
                {'ECG_I_filtered': smoothed_heartbeats[i], 'wave_label': 's'} for i in range(max(0, s_wave - 1), min(len(smoothed_heartbeats), s_wave + 2))
            ])
            waves_list.extend([
                {'ECG_I_filtered': smoothed_heartbeats[i], 'wave_label': 't'} for i in range(max(0, t_wave - 1), min(len(smoothed_heartbeats), t_wave + 2))
            ])

    waves_df = pd.DataFrame(waves_list)

    return waves_df

def run(input_data_directory):
    interim_data_directory = './data/interim/ecg-id-database-2.0.0/'

    for file_name in os.listdir(input_data_directory):
        print(file_name)
        if file_name.endswith(".csv"):
            file_path = os.path.join(input_data_directory, file_name)
            waves_df = process_ecg_data(file_path)

            if interim_data_directory:
                os.makedirs(interim_data_directory, exist_ok=True)

                interim_file_name = f"{os.path.splitext(file_name)[0]}_interim.csv"
                interim_file_path = os.path.join(interim_data_directory, interim_file_name)
                waves_df.to_csv(interim_file_path, index=False)
