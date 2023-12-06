from src import pre_proc, wave_labeling

def main():
    raw_data_directory = './data/raw/ecg-id-database-1.0.0/'
    processed_data_directory = './data/processed/ecg-id-database-1.0.0/'

    # Pre-process raw data and get the processed data directory
    processed_data_directory = pre_proc.preprocess_ecg_data(raw_data_directory)

    # Run wave labeling on the processed data
    wave_labeling.run(processed_data_directory)

if __name__ == '__main__':
    main()
