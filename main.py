from src.a01 import main as ecg_main
from src.a02 import main as process_main
from src.a03_prep import main as prep_main
from src.a04_training import main as training_main

if __name__ == "__main__":
    # Run ECG data loading and plotting
    ecg_main()

    # Run ECG data processing
    process_main()

    # Run data preparation
    prep_main()

    # Train the model
    training_main()
