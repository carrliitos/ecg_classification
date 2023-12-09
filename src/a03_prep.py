from glob import glob
import numpy as np

def load_processed_data(directory='../data/processed/mitdb'):
    """
    Load processed ECG data from CSV files in a specified directory.

    Parameters:
    - directory (str): Path to the directory containing CSV files.

    Returns:
    - alldata (numpy.ndarray): Concatenated ECG data from all CSV files.
    """
    alldata = np.empty(shape=[0, 188])
    paths = glob(f'{directory}/*.csv')
    
    for path in paths:
        print('Loading ', path)
        csvrows = np.loadtxt(path, delimiter=',')
        alldata = np.append(alldata, csvrows, axis=0)

    return alldata

def shuffle_and_separate(data):
    """
    Shuffle and separate ECG data into training, validation, and testing sets.

    Parameters:
    - data (numpy.ndarray): Concatenated ECG data.

    Returns:
    - train_data (numpy.ndarray): Training set.
    - test_data (numpy.ndarray): Testing set.
    - validate_data (numpy.ndarray): Validation set.
    """
    np.random.shuffle(data)
    totrows = len(data)
    trainrows = int((totrows * 3 / 5) + 0.5)  # 60%
    testrows = int((totrows * 1 / 5) + 0.5)  # 20%
    validaterows = totrows - trainrows - testrows  # 20%
    mark1 = trainrows
    mark2 = mark1 + testrows

    return data[:mark1], data[mark1:mark2], data[mark2:]

def save_data(train_data, test_data, validate_data, directory='../data/interim/mitdb'):
    """
    Save ECG data into separate CSV files for training, testing, and validation.

    Parameters:
    - train_data (numpy.ndarray): Training set.
    - test_data (numpy.ndarray): Testing set.
    - validate_data (numpy.ndarray): Validation set.
    - directory (str): Path to the directory to save the CSV files.
    """
    with open(f'{directory}/train.csv', "wb") as fin:
        np.savetxt(fin, train_data, delimiter=",", fmt='%f')

    with open(f'{directory}/test.csv', "wb") as fin:
        np.savetxt(fin, test_data, delimiter=",", fmt='%f')

    with open(f'{directory}/validate.csv', "wb") as fin:
        np.savetxt(fin, validate_data, delimiter=",", fmt='%f')

def main():
    alldata = load_processed_data()
    train_data, test_data, validate_data = shuffle_and_separate(alldata)
    save_data(train_data, test_data, validate_data)

if __name__ == "__main__":
    main()
