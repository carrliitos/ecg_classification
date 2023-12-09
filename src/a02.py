import sys
import os
import wfdb as wf
import numpy as np
from scipy import signal
from src.datasets import mitdb as dm
from biosppy.signals import ecg

def load_data(path):
    """
    Load ECG data from WFDB files and extract relevant information.

    Parameters:
    - path (str): Path to the WFDB file.

    Returns:
    - data (numpy.ndarray): ECG data.
    - rates (numpy.ndarray): Classification rates.
    - record_info (dict): Information about the record.
    """
    record = wf.rdsamp(path)
    annotation = wf.rdann(path, 'atr')
    data = record[0].transpose()
    cat = np.array(annotation.symbol)
    rate = np.zeros_like(cat, dtype='float')

    for catid, catval in enumerate(cat):
        if catval == 'N':
            rate[catid] = 1.0  # Normal
        elif catval in realbeats:
            rate[catid] = 2.0  # Abnormal

    rates = np.zeros_like(data[0], dtype='float')
    rates[annotation.sample] = rate

    return data, rates, record[1]

def find_rpeaks(channel):
    """
    Find R-peaks in ECG data using the Biosppy library.

    Parameters:
    - channel (numpy.ndarray): ECG signal data for a single channel.

    Returns:
    - rpeaks (numpy.ndarray): R-peak locations.
    - rpeak_indices (numpy.ndarray): Indices of R-peaks in the original signal.
    """
    out = ecg.ecg(signal=channel, sampling_rate=360, show=False)
    rpeaks = np.zeros_like(channel, dtype='float')
    rpeaks[out['rpeaks']] = 1.0
    return rpeaks, out['rpeaks']

def process_beats(beats, rates, rpeaks):
    """
    Process individual heartbeats, normalize, resample, and append classifications.

    Parameters:
    - beats (numpy.ndarray): Array of individual heartbeats.
    - rates (numpy.ndarray): Classification rates.
    - rpeaks (numpy.ndarray): Indices of R-peaks in the original signal.

    Returns:
    - processed_beats (numpy.ndarray): Processed heartbeats with classifications.
    """
    beatstoremove = np.array([0])
    
    for idx, idxval in enumerate(rpeaks):
        firstround = idx == 0
        lastround = idx == len(beats) - 1

        if firstround or lastround:
            continue

        fromidx = 0 if idxval < 10 else idxval - 10
        toidx = idxval + 10
        catval = rates[fromidx:toidx].max()

        if catval == 0.0:
            beatstoremove = np.append(beatstoremove, idx)
            continue

        catval -= 1.0
        beats[idx] = np.append(beats[idx], beats[idx + 1][:40])
        beats[idx] = (beats[idx] - beats[idx].min()) / beats[idx].ptp()

        newsize = int((beats[idx].size * 125 / 360) + 0.5)
        beats[idx] = signal.resample(beats[idx], newsize)

        if beats[idx].size > 187:
            beatstoremove = np.append(beatstoremove, idx)
            continue

        zerocount = 187 - beats[idx].size
        beats[idx] = np.pad(beats[idx], (0, zerocount), 'constant', constant_values=(0.0, 0.0))
        beats[idx] = np.append(beats[idx], catval)

    beatstoremove = np.append(beatstoremove, len(beats) - 1)
    return np.array([beat for idx, beat in enumerate(beats) if idx not in beatstoremove])

def save_to_csv(beats, path, filename, chname):
    """
    Save processed heartbeats to a CSV file.

    Parameters:
    - beats (numpy.ndarray): Processed heartbeats with classifications.
    - path (str): Path to save the CSV file.
    - filename (str): Original filename.
    - chname (str): ECG channel type.
    """
    savedata = np.array(list(beats[:]), dtype=float)
    outfn = os.path.join(path, f"{filename}_{chname}.csv")
    print('    Generating ', outfn)
    
    with open(outfn, "wb") as fin:
        np.savetxt(fin, savedata, delimiter=",", fmt='%f')

def main():
    sys.path.append("/home/carlitos/Documents/Projects/ecg_classification")
    records = dm.get_records("../data/raw/mitdb")
    print('Total files: ', len(records))

    realbeats = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
                 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']

    for path in records:
        pathpts = path.split('/')
        fn = pathpts[-1]
        print('Loading file:', path)

        data, rates, record_info = load_data(path)
        print('    Sampling frequency used for this record:', record_info.get('fs'))
        print('    Shape of loaded data array:', data.shape)
        print('    Number of loaded annotations:', len(rates))

        for channelid, channel in enumerate(data):
            chname = record_info.get('sig_name')[channelid]
            print('    ECG channel type:', chname)

            rpeaks, rpeak_indices = find_rpeaks(channel)
            beats = np.split(channel, rpeak_indices)
            processed_beats = process_beats(beats, rates, rpeak_indices)
            save_to_csv(processed_beats, "../data/processed/mitdb", fn, chname)

if __name__ == "__main__":
    main()
