import mne
import yasa
import numpy as np
from pathlib import Path
import os
from multiprocessing import Pool, cpu_count
from mne.filter import filter_data
import matplotlib.pyplot as plt
from mne.time_frequency import psd_array_welch

from settings import base_path, data_path
from sleep.sleep_classification import sleep_scoring_from_file

EVENT_ID = {'WAKE': 0, 'IS/QW': 1, 'NREM': 2, 'SWS': 3, 'REM': 4, "artifact": 5}

thresholds_csv = base_path / 'metadata' / 'biotrial_sleep_thresholds.csv'


def preprocess(args):
    filename, save_folder, data_folder = args
    # Load the data
    raw = mne.io.read_raw_fif(filename, preload=True)
    # raw.plot(n_channels=30, block=True, scalings='auto')

    # Apply a high pass filter at 0.5 Hz
    raw.filter(l_freq=0.5, h_freq=None)
    # raw.plot(n_channels=30, block=True, scalings='auto')

    # Apply a notch filter at 50 and 60 Hz
    freqs = [50, 60]
    raw.notch_filter(freqs=freqs)
    # raw.plot(n_channels=30, block=True, scalings='auto')

    # Apply a low pass filter at 100 Hz
    raw.filter(l_freq=None, h_freq=60)
    # raw.plot(n_channels=30, block=True, scalings='auto')

    # Create epochs of 4s
    epoch_duration = 4.
    epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, preload=True)
    # sleep staging
    x_hypno, y_hypno, Sleep_stages = sleep_scoring_from_file(filename,
                                                             thresholds_csv,
                                                             epoch_duration=epoch_duration)

    # do that for a sleep-wake classification only
    #y_hypno[y_hypno == 1] = 0
    #y_hypno[y_hypno == 2] = 1
    Sleep_stages = {'WAKE': 0, 'IS/QW': 1, 'NREM': 2, 'SWS': 3, 'REM': 4}

    # reverse dict
    stage_from_number = {v: k for k, v in Sleep_stages.items()}

    # Initialize a list to store events
    events = []
    for i in range(len(epochs)):
        epoch = epochs[i]

        # Get EEG and EMG data
        eeg_data = epoch.get_data(picks='EEG')
        emg_data = epoch.get_data(picks='EMG')

        # Convert thresholds to μV
        eeg_threshold = 1000000e-6  # biotrial 1000000e-6
        emg_threshold = 1000000e-6  # biotrial 100000e-6

        # Check if EEG or EMG data exceed threshold or EEG segment is flat
        if (np.any(np.abs(eeg_data) > eeg_threshold) or np.any(np.abs(emg_data) > emg_threshold) or
                np.max(eeg_data) == np.min(eeg_data)):
            # Add an annotation to mark this epoch as 'BAD'
            raw.annotations.append(epoch.times[0], 4, 'BAD')
            events.append([epoch.events[0, 0], 0, EVENT_ID["artifact"]])

        else:  # if good, sleep stage
            sleep_stage = stage_from_number.get(y_hypno[i], np.nan)
            raw.annotations.append(epoch.times[0], 4, sleep_stage)
            events.append([epoch.events[0, 0], 0, EVENT_ID[sleep_stage]])

    # Convert events list to numpy array
    events = np.array(events, dtype=int)
    from collections import Counter
    counter = Counter(events[:, 2])

    print(counter)
    # Create an Epochs array from the raw data and the events
    epochs_with_events = mne.Epochs(raw, events, tmin=0, tmax=4, preload=True, baseline=(0, 0))

    # Save the preprocessed data with annotations and the Epochs array
    relative_path = os.path.relpath(filename, data_folder)
    new_path = Path(os.path.join(save_folder, relative_path))  # Convert new_path to Path object

    # The stem + new suffix
    raw_filename = new_path.with_name(new_path.stem + '_preprocessed_raw').with_suffix('.fif')
    epochs_filename = new_path.with_name(new_path.stem + '_preprocessed_epochs').with_suffix('.fif')

    os.makedirs(raw_filename.parent, exist_ok=True)  # Use .parent to get directory
    os.makedirs(epochs_filename.parent, exist_ok=True)  # Use .parent to get directory
    raw.save(raw_filename, overwrite=True)

    epochs_with_events.save(epochs_filename, overwrite=True)


def main(num_processes=4):
    data_folder = data_path / 'biotrial_data' / 'mne_raw'
    save_folder = data_path / 'biotrial_data' / 'preprocessed'
    fif_files = list(data_folder.glob('**/*.fif'))

    # Define the number of processes to spawn
    # num_processes = min(len(fif_files), cpu_count())

    # Package arguments for the preprocess function
    args = [(fif_file, save_folder, data_folder) for fif_file in fif_files]

    # Initialize a Pool of processes
    with Pool(num_processes) as p:
        p.map(preprocess, args)


if __name__ == "__main__":
    main(num_processes=20)

