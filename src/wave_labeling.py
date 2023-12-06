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
    r_waves, metadata = scipy.signal.find_peaks(ecg_data, height=0.275)

    p_waves = []
    q_waves = []
    s_waves = []
    t_waves = []

    waves_list = []

    index_range = len(smoothed_heartbeats) - 1 # helps us not overshoot last element's index
    r_waves_indexed = enumerate(r_waves)       # adds a separate index number to each R wave

    waves_df = pd.DataFrame(columns=['time', 'p', 'q', 'r', 's', 't'])

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
        start_index = r                    # the index of the current R wave
        try:
            end_index = r_waves[index + 1] # the index of the next R wave
        except IndexError:
            end_index = index_range        # have hit the end of the heartbeat array
        temp_rr_interval = smoothed_heartbeats[start_index:end_index]

        p_and_t_waves, metadata = scipy.signal.find_peaks(temp_rr_interval,
                                                          height=[-0.1, 0.325],
                                                          distance=200)

        if not p_and_t_waves.any():
            p_wave = t_wave = None
        else:
            p_wave = p_and_t_waves[-1]
            t_wave = p_and_t_waves[0]

            p_wave += r
            t_wave += r
        
        # Label the Q wave
        q_area = temp_rr_interval[-16:]
        q_area = pd.Series(q_area)
        q_wave = q_area.idxmin(axis=0)
    
        # Label the S wave
        s_area = temp_rr_interval[0:16]
        s_area = pd.Series(s_area)
        s_wave = s_area.idxmin(axis=0)
        
        '''Increment index all of the newly found waves, since each for loop
        causes the RR interval's index to start at 0. For example, if the 2nd RR
        interval's Q wave is found at index 12, yet that RR interval actually
        starts at 100, then the Q wave's actual index is 112.
        '''
        q_wave += r + len(temp_rr_interval) - 15
        s_wave += r

        # Calculate time for the QRS complex
        qrs_start_time = time_values[start_index]

        # Add the waves to the DataFrame
        waves_list.append({
            'time': qrs_start_time,
            'p': p_wave,
            'q': q_wave,
            'r': r,
            's': s_wave,
            't': t_wave
        })

    return pd.DataFrame(waves_list)

def run(input_data_directory):
    interim_data_directory = './data/interim/ecg-id-database-1.0.0/'

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
