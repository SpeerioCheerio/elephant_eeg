import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mne.filter import filter_data

import sys
sys.path.append('/')

from settings import data_path
from sleep.utils import convert_seconds, avg_value_per_epochs, value_distribution


def load_record(filename):
    if 'biotrial' in str(filename):
        lab_name = 'biotrial'
    else:
        lab_name = 'crl'

    # load h5 file
    with h5py.File(filename, 'r') as hf:
        eeg_data = hf['eeg'][:]
        sfreq = hf['eeg'].attrs['sfreq']
        emg_data = hf['emg'][:]

        # info
        timestamps_eeg = np.arange(len(eeg_data)) / sfreq
        record_duration = convert_seconds(max(timestamps_eeg))
        info = {
            'rat_id': hf.attrs['rat_id'],
            'condition_id': hf.attrs['condition_id'],
            'lab': lab_name,
            'session': hf.attrs['session_id'],
            'sfreq': sfreq,
            'record_duration': record_duration
        }
        if lab_name == 'biotrial':
            info['treatment_data'] = hf.attrs['treatment_data']
            info['dosing_offset_sec'] = hf.attrs['dosing_offset_sec']
        elif lab_name == 'crl':
            info['treatment'] = hf.attrs['treatment']
            info['dosing_offset_sec'] = None

    return eeg_data, emg_data, info


def process_data(filename1, filename2=None):
    # params
    epoch_duration = 4.

    # load
    eeg_data, emg_data, info = load_record(filename1)
    if filename2:
        if filename2.exists():
            eeg_data_m, emg_data_m, info = load_record(filename2)
            # concatenate
            emg_data = np.concatenate((emg_data, emg_data_m))
            eeg_data = np.concatenate((eeg_data, eeg_data_m))
    sfreq = info['sfreq']

    # filtering and rms
    filtered_emg = filter_data(data=emg_data.astype(np.float64), sfreq=sfreq,
                               l_freq=10, h_freq=80,
                               method='fir', fir_design='firwin',
                               phase='zero', fir_window='hamming')

    filtered_eeg = filter_data(data=eeg_data.astype(np.float64), sfreq=sfreq,
                               l_freq=5, h_freq=10,
                               method='fir', fir_design='firwin',
                               phase='zero', fir_window='hamming')
    rms_emg = pd.Series(filtered_emg ** 2).rolling(window=1000).mean()
    rms_eeg = pd.Series(filtered_eeg ** 2).rolling(window=1000).mean()

    # data for phase graph and distrib
    data = np.array([rms_emg, rms_eeg]).T
    x_epochs, y_epochs = avg_value_per_epochs(data, sfreq, epoch_duration=epoch_duration)
    y_epochs = y_epochs[~np.isnan(y_epochs).all(axis=1)]
    # histogram of values
    epsilon = 0.000000000000001
    epochs_emg = y_epochs[:, 0] + epsilon
    epochs_eeg = y_epochs[:, 1] + epsilon
    X_emg, Y_emg = value_distribution(epochs_emg)
    X_eeg, Y_eeg = value_distribution(epochs_eeg)

    epochs_emg = np.log(epochs_emg)
    epochs_eeg = np.log(epochs_eeg)

    return {
        'info': info,
        'epoch_duration': epoch_duration,
        'epochs_emg': epochs_emg,
        'epochs_eeg': epochs_eeg,
        'X_eeg': X_eeg,
        'Y_eeg': Y_eeg,
        'X_emg': X_emg,
        'Y_emg': Y_emg,
    }


def plot_graph(processed_data):

    # input data
    epochs_emg = processed_data['epochs_emg']
    epochs_eeg = processed_data['epochs_eeg']
    X_eeg = processed_data['X_eeg']
    Y_eeg = processed_data['Y_eeg']
    X_emg = processed_data['X_emg']
    Y_emg = processed_data['Y_emg']

    # Plot
    fig = plt.figure(figsize=(14, 7))
    fig.set_facecolor('white')
    fig.set_edgecolor('white')
    axes = {
        "emg_distrib": plt.axes([0.05, 0.35, 0.2, 0.6]),
        "eeg_distrib": plt.axes([0.3, 0.05, 0.65, 0.25]),
        "phase_graph": plt.axes([0.3, 0.35, 0.65, 0.6]),
    }

    # EMG distrib
    ax = axes['emg_distrib']
    ax.plot(Y_emg, X_emg, linewidth=4)
    ax.set_ylabel('EMG')
    ylim = ax.get_ylim()
    # EEG distrib
    ax = axes['eeg_distrib']
    ax.plot(X_eeg, Y_eeg, linewidth=4)
    ax.set_ylabel('EEG')
    xlim = ax.get_xlim()
    # scatter plot
    ax = axes['phase_graph']
    ax.scatter(epochs_eeg, epochs_emg, s=0.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


if __name__ == '__main__':
    # parameters
    lab_name = "crl"
    rat_id = 12
    condition = 1

    record_dir = data_path / f'{lab_name}_data' / 'sleep_raw'
    figures_dir = data_path / f'{lab_name}_data' / 'thresholds'

    # filename
    if lab_name == "biotrial":
        filename = record_dir / f'condition-{condition}' / f'rat_{rat_id}.h5'
        print(filename)
        processed_data = process_data(filename)
    else:
        filename_base = record_dir / f'condition-{condition}' / f'rat_{rat_id}_baseline.h5'
        filename_manip = record_dir / f'condition-{condition}' / f'rat_{rat_id}_manipulation.h5'
        print(filename_base)
        processed_data = process_data(filename_base, filename_manip)

    # plot
    plot_graph(processed_data)
