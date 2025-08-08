import pandas as pd
import numpy as np
from mne.filter import filter_data

import sys

sys.path.append('/')

from sleep.utils import avg_value_per_epochs, load_record, get_frequency_amplitude


def prepare_data_for_sleep_scoring(signals, sfreq, epoch_duration=4.):
    # input data
    emg_data = signals['emg']
    eeg_data = signals['eeg']

    # filtering and rms
    rms_emg = get_frequency_amplitude(emg_data, sfreq, 10, 45, smoothing_window=3000)
    rms_eeg = get_frequency_amplitude(eeg_data, sfreq, 0.4, 4, smoothing_window=3000)

    # data per epochs
    data = np.array([rms_emg, rms_eeg]).T
    x_epochs, y_epochs = avg_value_per_epochs(data, sfreq, epoch_duration=epoch_duration)
    epochs_emg = np.log(y_epochs[:, 0])
    epochs_eeg = np.log(y_epochs[:, 1])

    return epochs_eeg, epochs_emg


def sleep_scoring(signals, sfreq, thresholds, epoch_duration=4., start_wake=False):
    epochs_eeg, epochs_emg = prepare_data_for_sleep_scoring(signals, sfreq, epoch_duration=epoch_duration)
    x_epochs = epoch_duration * np.arange(0, len(epochs_eeg)) + epoch_duration / 2
    # behavioral classification
    Sleep_stages = {'WAKE': 0, 'IS/QW': 1, 'NREM': 2, 'SWS': 3, 'REM': 4}

    idx_wake = epochs_emg >= thresholds['emg_high']
    idx_isqw = np.logical_and(np.logical_and(
        epochs_emg < thresholds['emg_high'], epochs_emg >= thresholds['emg_low']), epochs_eeg >= thresholds['eeg_low'])
    idx_sws = np.logical_and(epochs_emg < thresholds['emg_high'], epochs_eeg >= thresholds['eeg_high'])
    idx_rem = np.logical_and(epochs_emg < thresholds['emg_low'], epochs_eeg < thresholds['eeg_low'])

    x_hypno = x_epochs.copy()
    y_hypno = np.ones_like(epochs_emg) * Sleep_stages['NREM']
    y_hypno[idx_wake] = Sleep_stages['WAKE']
    y_hypno[idx_isqw] = Sleep_stages['IS/QW']
    y_hypno[idx_sws] = Sleep_stages['SWS']
    y_hypno[idx_rem] = Sleep_stages['REM']

    if start_wake:
        mask = y_hypno != Sleep_stages['WAKE']
        y_hypno[:np.argmax(mask)] = Sleep_stages['WAKE']

    return x_hypno, y_hypno, Sleep_stages


def sleep_scoring_from_file(filename, thresholds_csv, epoch_duration=4.):
    eeg_data, emg_data, info, _ = load_record(filename)

    sfreq = int(info['sfreq'])
    rat_id = int(info['rat_id'])
    condition = int(info['condition_id'])

    # thresholds
    df_thresholds = pd.read_csv(thresholds_csv)
    df = df_thresholds[(df_thresholds['Rat id'] == rat_id) & (df_thresholds['Condition'] == condition)]
    thresholds = {'emg_high': df['EMG_high'].item(),
                  'emg_low': df['EMG_low'].item(),
                  'eeg_high': df['EEG_high'].item(),
                  'eeg_low': df['EEG_low'].item()}

    # signals
    signals = {'emg': emg_data, 'eeg': eeg_data}

    # sleep scoring
    x_hypno, y_hypno, Sleep_stages = sleep_scoring(signals, sfreq, thresholds,
                                                   epoch_duration=epoch_duration,
                                                   start_wake=True)

    return x_hypno, y_hypno, Sleep_stages


if __name__ == '__main__':
    from settings import data_path, base_path

    # params
    epoch_duration = 4.

    lab_name = "crl"
    condition = 3
    rat_id = 4

    record_dir = data_path / f'{lab_name}_data' / 'mne_raw'
    thresholds_csv = base_path / 'metadata' / f'{lab_name}_sleep_thresholds.csv'

    filename = record_dir / f'condition-{condition}' / f'rat_{rat_id}_manipulation.fif'
    if filename.exists():
        x_hypno, y_hypno, Sleep_stages = sleep_scoring_from_file(filename,
                                                                 thresholds_csv,
                                                                 epoch_duration=4.)
