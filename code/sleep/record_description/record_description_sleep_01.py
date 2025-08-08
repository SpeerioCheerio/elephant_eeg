import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

import sys
sys.path.append('/')
sys.path.append('/home/karim/Sama/Damona')

from settings import data_path
from sleep.spectrograms import compute_spectrogram_lspopt, plot_spectrogram
from sleep.utils import load_record
from sleep.sleep_classification import sleep_scoring
from sleep.hypnogram import plot_sleep_stage_transitions, compute_sleep_stage_metrics, \
    plot_stage_duration, plot_stage_percentage


def process_data(filename1, thresholds, filename2=None):
    """
    Generates the data used in the record description figure

    :param filename:
    :return:
    """
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


    # info

    # spectrograms
    params = {"win_sec": 30, "fmin": 1, "fmax": 170}
    specg_emg, t_emg, freq_emg, _ = compute_spectrogram_lspopt(
        emg_data, sfreq, transformation="log", parameters=params)

    params = {"win_sec": 30, "fmin": 0.5, "fmax": 35}
    specg_eeg, t_eeg, freq_eeg, _ = compute_spectrogram_lspopt(
        eeg_data, sfreq, transformation="log", parameters=params)

    # Hypnogram
    signals = {'emg': emg_data, 'eeg': eeg_data}
    x_hypno, y_hypno, Sleep_stages = sleep_scoring(signals, sfreq, thresholds,
                                                   epoch_duration=epoch_duration)

    return {
        'info': info,
        'epoch_duration': epoch_duration,
        'spectrogram_emg': [specg_emg, t_emg, freq_emg],
        'spectrogram_eeg': [specg_eeg, t_eeg, freq_eeg],
        'x_hypno': x_hypno,
        'y_hypno': y_hypno,
        'Sleep_stages': Sleep_stages
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
    record_duration = info['record_duration']
    specg_emg, t_emg, freq_emg = processed_data['spectrogram_emg']
    specg_eeg, t_eeg, freq_eeg = processed_data['spectrogram_eeg']
    x_hypno = processed_data['x_hypno']
    hypnogram = processed_data['y_hypno']
    Sleep_stages = processed_data['Sleep_stages']

    # Figure
    fig = plt.figure(figsize=(14, 7))
    fig.set_facecolor('white')
    fig.set_edgecolor('white')
    axes = {
        "info": plt.axes([0.05, 0.74, 0.25, 0.21]),
        "stage_duration": plt.axes([0.05, 0.38, 0.18, 0.22]),
        "stage_percentage": plt.axes([0.05, 0.05, 0.13, 0.22]),
        "transitions": plt.axes([0.21, 0.05, 0.16, 0.22]),
        "spectrogram_emg": plt.axes([0.42, 0.65, 0.54, 0.25]),
        "spectrogram_eeg": plt.axes([0.42, 0.32, 0.54, 0.25]),
        "hypnogram": plt.axes([0.42, 0.07, 0.54, 0.2])
    }

    # info in Text Box
    infos_text = info["lab"] + "\n"
    infos_text += f'Rat: {info["rat_id"]}' + "\n"
    infos_text += f'Condition: {info["condition_name"]}' + "\n"
    if "treatment" in info:
        infos_text += f'Treatment: {info["treatment"]}' + "\n"
    infos_text += f'Session: {info["session"]}' + "\n"
    infos_text += f'Sampling Frequency: {info["sfreq"]} Hz' + "\n"
    infos_text += f'Epoch duration: {epoch_duration} sec' + "\n"
    infos_text += f'Record duration: {record_duration}' + "\n"
    TextBox(axes["info"], "", initial=infos_text)

    # stage metrics
    stage_duration, stage_percentage = compute_sleep_stage_metrics(hypnogram, epoch_duration)
    plot_stage_duration(stage_duration, axe_plot=axes["stage_duration"])
    plot_stage_percentage(stage_percentage, axe_plot=axes["stage_percentage"])

    # transitions hypnogram
    plot_sleep_stage_transitions(hypnogram, epoch_duration,
                                 axe_plot=axes["transitions"],
                                 title="Transitions")

    # spectrogram EMG
    plot_spectrogram(specg_emg, t_emg, freq_emg, axe_plot=axes['spectrogram_emg'],
                     rescale=rescale, start_time=0, title='EMG', colourbar=False)

    # spectrogram EEG
    plot_spectrogram(specg_eeg, t_eeg, freq_eeg, axe_plot=axes['spectrogram_eeg'],
                     rescale=rescale, start_time=0, title='EEG', colourbar=False)

    # hypnogram
    ax = axes['hypnogram']
    ax.step(x_hypno / rescale, hypnogram, 'k', linewidth=0.3)

    ytick_substage = list(Sleep_stages.values())
    ylabel_substage = list(Sleep_stages)
    ax.set_yticks(np.sort(ytick_substage))
    ax.set_yticklabels(ylabel_substage)
    ax.set_xlabel('Time (h)')

    # x_lim
    x_lim = tuple([0, info['record_duration_sec'] / rescale])
    axes['spectrogram_emg'].set_xlim(x_lim)
    axes['spectrogram_eeg'].set_xlim(x_lim)
    axes['hypnogram'].set_xlim(x_lim)

    # save fig
    title = filename.stem + ' / ' + filename.parent.stem
    fig.suptitle(title)
    plt.savefig(save_path)
    plt.close()
    return 1


if __name__ == '__main__':
    overwrite = False

    # parameters
    lab_names = ["biotrial", "crl"]
    conditions = range(7)
    rats = range(1, 13)
    for lab_name in lab_names:
        record_dir = data_path / f'{lab_name}_data' / 'sleep_raw'
        figures_dir = data_path / f'{lab_name}_data' / 'record_descriptions'

        session_csv = data_path / f'{lab_name}_info' / 'behavior_thresholds.csv'

        thresholds_csv = data_path / f'{lab_name}_info' / 'behavior_thresholds.csv'
        df_thresholds = pd.read_csv(thresholds_csv)

        for condition in conditions:
            for rat_id in rats:
                df = df_thresholds[
                    (df_thresholds['Rat id'] == rat_id) & (df_thresholds['Condition'] == condition)
                    ]

                # figure - check if exists
                figure_save_path = figures_dir / f'rat_{rat_id}_cond_{condition}.png'
                if overwrite or not figure_save_path.exists():
                    if lab_name == "biotrial":
                        filename = record_dir / f'condition-{condition}' / f'rat_{rat_id}.h5'
                        if not filename.exists():
                            continue
                        print(filename)

                        thresholds = {'emg': df['EMG'].item(), 'eeg': df['EEG'].item()}
                        processed_data = process_data(filename, thresholds)
                        plot_description_record(processed_data, figure_save_path)
                    else:
                        filename_base = record_dir / f'condition-{condition}' / f'rat_{rat_id}_baseline.h5'
                        filename_manip = record_dir / f'condition-{condition}' / f'rat_{rat_id}_manipulation.h5'
                        if not filename_base.exists():
                            continue
                        print(filename_base)

                        thresholds = {'emg': df['EMG'].item(), 'eeg': df['EEG'].item()}
                        processed_data = process_data(filename_base, thresholds, filename2=filename_manip)
                        plot_description_record(processed_data, figure_save_path)
