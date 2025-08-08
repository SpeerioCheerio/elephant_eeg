import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from mne.filter import filter_data

import sys
sys.path.append('/')

from settings import data_path
from sleep.spectrograms import compute_spectrogram_lspopt, plot_spectrogram
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
            info['treatment'] = hf.attrs['treatment_data']
            info['dosing_offset_sec'] = hf.attrs['dosing_offset_sec']
        elif lab_name == 'crl':
            info['treatment'] = hf.attrs['treatment']
            info['dosing_offset_sec'] = None

    return eeg_data, emg_data, info


def generate_record_data(filename_base, filename_manip):
    """
    Generates the data used in the record description figure
    :param filename_base:
    :param filename_manip:
    :return:
    """
    # params
    epoch_duration = 4.
    # load
    eeg_data, emg_data, info = load_record(filename_base)
    sfreq = info['sfreq']
    if filename_manip.exists():
        eeg_data_m, emg_data_m, info = load_record(filename_manip)
        info['dosing_offset_sec'] = len(emg_data) / sfreq
        # concatenate
        emg_data = np.concatenate((emg_data, emg_data_m))
        eeg_data = np.concatenate((eeg_data, eeg_data_m))

    # spectrograms
    params = {"win_sec": 10, "fmin": 1, "fmax": 150}
    specg_emg, t_emg, freq_emg, _ = compute_spectrogram_lspopt(
        emg_data, sfreq, transformation="log", parameters=params)

    params = {"win_sec": 30, "fmin": 0.5, "fmax": 30}
    specg_eeg, t_eeg, freq_eeg, _ = compute_spectrogram_lspopt(
        eeg_data, sfreq, transformation="log", parameters=params)

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
        'spectrogram_emg': [specg_emg, t_emg, freq_emg],
        'spectrogram_eeg': [specg_eeg, t_eeg, freq_eeg],
        'epochs_emg': epochs_emg,
        'epochs_eeg': epochs_eeg,
        'X_eeg': X_eeg,
        'Y_eeg': Y_eeg,
        'X_emg': X_emg,
        'Y_emg': Y_emg,
    }


def plot_description_record(processed_data, save_path):
    """
    Plot the record description figure from the processed data

    :param processed_data: dict
    :return:
    """
    # params
    rescale = 3600

    # input data
    info = processed_data['info']
    epoch_duration = processed_data['epoch_duration']
    specg_emg, t_emg, freq_emg = processed_data['spectrogram_emg']
    specg_eeg, t_eeg, freq_eeg = processed_data['spectrogram_eeg']
    epochs_emg = processed_data['epochs_emg']
    epochs_eeg = processed_data['epochs_eeg']
    X_eeg = processed_data['X_eeg']
    Y_eeg = processed_data['Y_eeg']
    X_emg = processed_data['X_emg']
    Y_emg = processed_data['Y_emg']

    # Figure
    fig = plt.figure(figsize=(14, 7))
    fig.set_facecolor('white')
    fig.set_edgecolor('white')
    axes = {
        "info": plt.axes([0.05, 0.74, 0.25, 0.21]),
        "spectrogram_emg": plt.axes([0.05, 0.29, 0.9, 0.2]),
        "spectrogram_eeg": plt.axes([0.05, 0.05, 0.9, 0.2]),
        "emg_distrib": plt.axes([0.55, 0.66, 0.08, 0.3]),
        "eeg_distrib": plt.axes([0.67, 0.52, 0.3, 0.12]),
        "phase_graph": plt.axes([0.67, 0.66, 0.3, 0.3]),
    }

    # info
    infos_text = info["lab"] + "\n"
    infos_text += f'Rat: {info["rat_id"]}' + "\n"
    infos_text += f'Condition: {info["condition_id"]}' + "\n"
    if "treatment" in info:
        infos_text += f'Treatment: {info["treatment"]}' + "\n"
    infos_text += f'Session: {info["session"]}' + "\n"
    infos_text += f'Sampling Frequency: {info["sfreq"]} Hz' + "\n"
    infos_text += f'Epoch duration: {epoch_duration} sec' + "\n"
    infos_text += f'Record duration: {info["record_duration"]}' + "\n"
    TextBox(axes["info"], "", initial=infos_text)

    # spectrogram EMG
    ax = axes['spectrogram_emg']
    plot_spectrogram(specg_emg, t_emg, freq_emg, axe_plot=ax,
                     rescale=rescale, start_time=0, title='EMG')
    if 'dosing_offset_sec' in info:
        if info['dosing_offset_sec'] is not None:
            offset = info['dosing_offset_sec'] / rescale
            ax.axvline(x=offset, color='k', linewidth=2)
    ax.set_xticks([])

    # spectrogram EEG
    ax = axes['spectrogram_eeg']
    plot_spectrogram(specg_eeg, t_eeg, freq_eeg, axe_plot=ax,
                     rescale=rescale, start_time=0, title='EEG')
    if 'dosing_offset_sec' in info:
        if info['dosing_offset_sec'] is not None:
            offset = info['dosing_offset_sec'] / rescale
            ax.axvline(x=offset, color='k', linewidth=2)

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

    # save fig
    title = f'rat {info["rat_id"]} / condition {info["condition_id"]}'
    fig.suptitle(title)
    plt.savefig(save_path)
    plt.close()
    return 1


if __name__ == '__main__':
    overwrite = False

    # parameters
    lab_names = ["crl"]
    conditions = range(1, 7)
    rats = range(1, 13)
    for lab_name in lab_names:
        record_dir = data_path / f'{lab_name}_data' / 'sleep_raw'
        figures_dir = data_path / f'{lab_name}_data' / 'thresholds'
        for condition in conditions:
            for rat_id in rats:

                # filename
                filename_base = record_dir / f'condition-{condition}' / f'rat_{rat_id}_baseline.h5'
                filename_manip = record_dir / f'condition-{condition}' / f'rat_{rat_id}_manipulation.h5'
                if filename_base.exists():
                    # figure - check if exists
                    figure_save_path = figures_dir / f'rat_{rat_id}_cond_{condition}.png'
                    if overwrite or not figure_save_path.exists():
                        # process and plot
                        print(filename_base)
                        processed_data = generate_record_data(filename_base, filename_manip)
                        plot_description_record(processed_data, save_path=figure_save_path)
