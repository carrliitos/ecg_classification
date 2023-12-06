import os
import wfdb
import pandas as pd

def rename_and_move_files(root_path):
    for person_folder in os.listdir(root_path):
        person_folder_path = os.path.join(root_path, person_folder)
        
        if os.path.isdir(person_folder_path):
            for csv_file in os.listdir(person_folder_path):
                if csv_file.endswith(".csv"):
                    old_path = os.path.join(person_folder_path, csv_file)
                    new_name = f"{csv_file}"
                    new_path = os.path.join(root_path, new_name)
                    
                    os.rename(old_path, new_path)

            os.rmdir(person_folder_path)

def preprocess_ecg_data(raw_data_directory):
    processed_data_directory = './data/processed/ecg-id-database-1.0.0/'
    person_directories = [d for d in os.listdir(raw_data_directory) if os.path.isdir(os.path.join(raw_data_directory, d))]

    for person_dir in person_directories:
        person_processed_directory = os.path.join(processed_data_directory, person_dir)
        os.makedirs(person_processed_directory, exist_ok=True)

        dat_files = [f for f in os.listdir(os.path.join(raw_data_directory, person_dir)) if f.endswith('.dat')]

        for dat_file in dat_files:
            dat_file_path = os.path.join(raw_data_directory, person_dir, dat_file)

            record = wfdb.rdrecord(os.path.splitext(dat_file_path)[0])

            signals = record.p_signal
            signal_names = record.sig_name

            ecg_filtered_index = signal_names.index("ECG I filtered")

            ecg_filtered_signal = signals[:, ecg_filtered_index]

            df = pd.DataFrame(data=ecg_filtered_signal, columns=["ECG_I_filtered"])

            new_csv_file_name = f"{person_dir}_{os.path.splitext(dat_file)[0]}.csv"
            new_csv_file_path = os.path.join(person_processed_directory, new_csv_file_name)

            df.to_csv(new_csv_file_path, index=False)

    rename_and_move_files(processed_data_directory)

    return processed_data_directory
