import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.filter import filter_data

from settings import data_path
from sleep.threshold.thresold_estimation import threshold_estimator, show_threshold_estimation
from sleep.spectrograms import compute_spectrogram_lspopt, plot_spectrogram


def classify_in_epochs(data_to_threshold,
                       threshold,
                       sfreq,
                       default_value=1,
                       epoch_duration=4.):
    # values
    below_value = 0
    above_value = 1
    nan_value = default_value

    # sleep epochs
    window_size = int(4 * sfreq)
    data = data_to_threshold.to_numpy()
    data = data[:(len(data) // window_size) * window_size]  # Trim the array
    epoch_avgs = data.reshape(-1, window_size).mean(axis=1)

    # x and y
    x_epochs = epoch_duration * np.arange(0, len(epoch_avgs)) + epoch_duration / 2
    y_epochs = np.ones_like(epoch_avgs) * below_value
    y_epochs[epoch_avgs > threshold] = above_value
    y_epochs[np.isnan(epoch_avgs)] = nan_value

    return x_epochs, y_epochs


def sleep_staging(eeg_data, emg_data, sfreq,
                  epoch_duration=4.,
                  theta_threshold=None, emg_threshold=None):
    filtered_emg_data = filter_data(data=emg_data, sfreq=sfreq,
                                    l_freq=10, h_freq=100,
                                    method='fir', fir_design='firwin',
                                    phase='zero', fir_window='hamming')

    eeg_theta = filter_data(eeg_data, sfreq=sfreq, l_freq=5, h_freq=10, method='fir',
                            fir_design='firwin', phase='zero', fir_window='hamming')

    # rms values
    rms_emg_data = pd.Series(filtered_emg_data ** 2).rolling(window=500).mean()
    rms_eeg_theta = pd.Series(eeg_theta ** 2).rolling(window=500).mean()

    # EMG sleep epochs
    if emg_threshold is None:
        emg_log_threshold, _, _, _ = threshold_estimator(rms_emg_data)
        emg_threshold = np.exp(emg_log_threshold)
    x_sleep, y_sleep = classify_in_epochs(rms_emg_data, emg_threshold,
                                          sfreq, epoch_duration=epoch_duration)

    # EEG theta epochs
    if theta_threshold is None:
        theta_log_threshold, _, _, _  = threshold_estimator(rms_eeg_theta)
        theta_threshold = np.exp(theta_log_threshold)
    x_theta, y_theta = classify_in_epochs(rms_eeg_theta, theta_threshold, sfreq=sfreq,
                                          default_value=0, epoch_duration=epoch_duration)

    # hypnogram
    Sleep_stages = {'NREM': 0, 'REM': 1, 'Wake': 2}

    x_hypno = x_sleep.copy()
    y_hypno = y_sleep.copy()
    y_hypno[y_theta == 1] = Sleep_stages['REM']
    y_hypno[y_sleep == 1] = Sleep_stages['Wake']

    return x_hypno, y_hypno, Sleep_stages


if __name__ == '__main__':
    # parameters
    epoch_duration = 4.

    # file
    lab_name = 'biotrial'
    condition = 2
    rat_id = 7
    session = "baseline"

    # filename
    record_dir = data_path / f'{lab_name}_data' / 'mne_raw'
    filename = record_dir / f'condition-{condition}' / f'rat_{rat_id}_{session}.fif'

    # load a signal
    raw = mne.io.read_raw_fif(filename, preload=True)
    sfreq = raw.info['sfreq']
    timestamps = raw.times
    emg_data = raw.get_data()[1]

    # rms values
    filtered_emg_data = filter_data(data=emg_data, sfreq=sfreq,
                                    l_freq=10, h_freq=100,
                                    method='fir', fir_design='firwin',
                                    phase='zero', fir_window='hamming')
    rms_emg_data = pd.Series(filtered_emg_data ** 2).rolling(window=500).mean()

    # EMG sleep epochs
    emg_log_threshold, X, Y, params_optimized = threshold_estimator(rms_emg_data)
    emg_threshold = np.exp(emg_log_threshold)
    x_sleep, y_sleep = classify_in_epochs(rms_emg_data, emg_threshold,
                                          sfreq, epoch_duration=epoch_duration)

    # spectrograms
    params = {"win_sec": 30, "fmin": 1, "fmax": 250}
    specg_emg, t_emg, freq_emg, _ = compute_spectrogram_lspopt(
        emg_data, sfreq, transformation="log", parameters=params)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(15, 8))
    axs = np.ravel(axs)
    rescale = 60
    plt.rcParams.update({'font.size': 12})

    # spectrogram EMG
    img_emg = plot_spectrogram(specg_emg, t_emg, freq_emg, axe_plot=axs[0], rescale=rescale,
                               start_time=0, title='EMG')

    # hypnogram
    ytick_substage = [0, 1]  # position of the stages in the graph, down to up
    ylabel_substage = ['Sleep', 'Wake']  # stages in the graph, down to up
    ax = axs[1]
    ax.step(x_sleep/rescale, y_sleep, 'r', linewidth=0.8)
    ax.set_yticks(np.sort(ytick_substage))
    ax.set_yticklabels(ylabel_substage)

    # x_lim
    x_lim = tuple([0, 120])
    for ax in axs:
        ax.set_xlim(x_lim)

    ax.set_xlabel('Time (min)')
    plt.show()

    # Threshold curves
    show_threshold_estimation(X, Y, emg_log_threshold, params_optimized)

