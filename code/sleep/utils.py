from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import mne
from mne.filter import filter_data


condition_map = {
    "biotrial": {
            '0': 'Saline',
            '1': 'Vehicle (5 mL/kg, ip)',
            '2': 'AC101 (5 mg/kg, ip)',
            '3': 'AC101 (15 mg/kg, ip)',
            '4': 'Basmisanil (5 mg/kg, ip)',
            '5': 'MRK016 (3 mg/kg, ip)',
            '6': 'Diazepam (2 mg/kg, ip)'
        },
    "crl": {
        '1': 'Vehicle',
        '2': 'AC101 1(mg / mL)',
        '3': 'AC101 3(mg / mL)',
        '4': 'AC101 10(mg / mL)',
        '5': 'Diazepam 0.5(mg / mL)',
        '6': 'Diazepam 1(mg / mL)'
    }
}


def convert_seconds(seconds):
    """
    Convert seconds to hh:mm:ss
    :param seconds: (int, float)
    :return:
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


def load_record(filename):
    filename = Path(filename)

    if 'biotrial' in str(filename):
        lab_name = 'biotrial'
    else:
        lab_name = 'crl'

    extra_signals = {}

    # load h5 file
    if filename.suffix == '.h5':
        with h5py.File(filename, 'r') as hf:
            eeg_data = hf['eeg'][:]
            sfreq = hf['eeg'].attrs['sfreq']
            emg_data = hf['emg'][:]
            extra_signals['activity'] = hf['activity'][:]
            extra_signals['temp'] = hf['temp'][:]

            # info
            timestamps = np.arange(len(eeg_data)) / sfreq
            record_duration_sec = max(timestamps)
            record_duration = convert_seconds(record_duration_sec)
            info = {
                'rat_id': hf.attrs['rat_id'],
                'condition_id': hf.attrs['condition_id'],
                'lab': lab_name,
                'session': hf.attrs['session_id'],
                'sfreq': sfreq,
                'record_duration': record_duration,
                'record_duration_sec': record_duration_sec
            }
            if lab_name == 'biotrial':
                info['treatment'] = hf.attrs['treatment_data']
                info['dosing_offset_sec'] = hf.attrs['dosing_offset_sec']
            elif lab_name == 'crl':
                info['treatment'] = hf.attrs['treatment']
                info['dosing_offset_sec'] = None

    # load fif File with mne
    elif filename.suffix == '.fif':
        raw = mne.io.read_raw_fif(filename, preload=True)
        sfreq = raw.info['sfreq']
        eeg_data = raw.get_data(picks='EEG')
        emg_data = raw.get_data(picks='EMG')
        if eeg_data.ndim > 1:
            eeg_data = eeg_data[0]
        if emg_data.ndim > 1:
            emg_data = emg_data[0]
        # info
        record_duration_sec = max(raw.times)
        record_duration = convert_seconds(record_duration_sec)
        info = {
            'sfreq': sfreq, 'record_duration': record_duration,
            'lab': lab_name, 'record_duration_sec': record_duration_sec
        }
        if raw.info['description']:
            pairs_desc = raw.info['description'].split(', ')
            raw_info = {pair.split(': ')[0]: pair.split(': ')[1] for pair in pairs_desc}
            info = {**info, **raw_info}

    # check if rat_id and condition_id in info
    if "condition_id" not in info:
        condition_id = int(filename.parent.stem.replace('_','-').split('-')[1])
        info["condition_id"] = condition_id
    if "rat_id" not in info:
        rat_id = int(filename.stem.replace('_','-').split('-')[1])
        info["rat_id"] = rat_id

    # add the name of the condition
    info["condition_name"] = condition_map[lab_name][str(info["condition_id"])]

    return eeg_data, emg_data, info, extra_signals


def bandpass_filter(data, lowcut, highcut, sfreq, order=5):
    nyquist = 0.5 * sfreq
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def get_frequency_amplitude(data, sfreq, lowcut, highcut, smoothing_window=3000):
    filtered_data = filter_data(data=data.astype(np.float64), sfreq=sfreq,
                                l_freq=lowcut, h_freq=highcut,
                                method='fir', fir_design='firwin',
                                phase='zero', fir_window='hamming')
    amplitude = pd.Series(filtered_data ** 2).rolling(window=smoothing_window).mean()
    amplitude = amplitude.combine_first(pd.Series(filtered_data ** 2))
    return amplitude.to_numpy()


def visit_h5_datasets(filename):
    def print_datasets(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name)

    with h5py.File(filename, 'r') as f:
        f.visititems(print_datasets)


def avg_value_per_epochs(data, sfreq, epoch_duration=4.):
    # unsqueeze if 2D
    if data.ndim == 1:
        data = data[:, np.newaxis]

    window_size = int(epoch_duration * sfreq)
    data = data[:(data.shape[0] // window_size) * window_size, :]  # Trim the array
    epoch_avgs = data.reshape(-1, window_size, data.shape[1]).mean(axis=1)

    # x and y
    x_epochs = epoch_duration * np.arange(0, epoch_avgs.shape[0]) + epoch_duration / 2

    return x_epochs, epoch_avgs


def value_distribution(filtered_data, bins=1000, cleaning=True):
    if isinstance(filtered_data, np.ndarray):
        signal_values = filtered_data[~np.isnan(filtered_data)]
    else:
        signal_values = filtered_data.dropna().values
    signal_values = signal_values[signal_values > 0]
    signal_values = np.log(signal_values)
    # Creating histogram
    Y, X = np.histogram(signal_values, bins=bins, density=True)
    X = (X[1:] + X[:-1]) / 2  # Get bin centers

    # cleaning
    if cleaning:
        X, Y = X[2:-2], Y[2:-2]
        threshold = 0.015 * np.max(Y)
        idx_min, idx_max = np.argmax(Y > threshold), np.where(Y > threshold)[0][-1]
        X, Y = X[idx_min:idx_max + 1], Y[idx_min:idx_max + 1]

    return X, Y
