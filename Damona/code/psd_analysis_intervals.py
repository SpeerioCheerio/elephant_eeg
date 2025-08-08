import os
import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mne.time_frequency import psd_array_welch

def get_condition_map(cro):
    if cro == 'biotrial':
        condition_map = {
            '0': 'Saline',
            '1': 'Vehicle (5 mL/kg, ip)',
            '2': 'AC101 (5 mg/kg, ip)',
            '3': 'AC101 (15 mg/kg, ip)',
            '4': 'Basmisanil (5 mg/kg, ip)',
            '5': 'MRK016 (3 mg/kg, ip)',
            '6': 'Diazepam (2 mg/kg, ip)'
        }
    elif cro == 'crl':
        condition_map = {
            '1': 'Vehicle',
            '2': 'AC101 1(mg / mL)',
            '3': 'AC101 3(mg / mL)',
            '4': 'AC101 10(mg / mL)',
            '5': 'Diazepam 0.5(mg / mL)',
            '6': 'Diazepam 1(mg / mL)'
        }
    else:
        condition_map = {}

    return condition_map

def power_spectral_analysis(raw, fmin=0.5, fmax=30):
    psds, freqs = psd_array_welch(raw.get_data()[0], sfreq=raw.info['sfreq'], fmin=fmin, fmax=fmax)
    psd_normalized = psds / np.sum(psds)

    return freqs, psd_normalized


import concurrent.futures


def process_file(file, interval_seconds=1800):
    results = {}

    raw = mne.io.read_raw_fif(file, preload=True)
    description = raw.info['description']

    # Parse description
    desc_dict = {}
    for item in description.split(', '):
        key, val = item.split(': ')
        desc_dict[key] = val

    rat_id = desc_dict['rat_id']
    condition = desc_dict['condition_id']
    manipulation_type = 'baseline' if 'baseline' in file.stem.lower() else 'manipulation'

    # Split raw into intervals
    interval_samples = int(interval_seconds * raw.info['sfreq'])
    for start_sample in range(0, len(raw.times), interval_samples):
        end_sample = start_sample + interval_samples
        interval = start_sample // interval_samples

        max_time = raw.times[-1]  # get the maximum time in the raw data

        # Adjust tmax if it exceeds the maximum time in the raw data
        if end_sample / raw.info['sfreq'] > max_time:
            end_sample = int(max_time * raw.info['sfreq'])

        raw_interval = raw.copy().crop(tmin=start_sample / raw.info['sfreq'],
                                       tmax=end_sample / raw.info['sfreq'])

        freqs, psds_db = power_spectral_analysis(raw_interval)

        key = (rat_id, condition, manipulation_type, interval)  # key now includes the interval
        if key not in results:
            results[key] = []
        results[key].append((freqs, psds_db))

    return results


def calculate_spectra(condition_folders, condition_map, interval_seconds=1800, n_jobs=6):
    all_results = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for condition_folder in condition_folders:
            raw_files = [file for file in condition_folder.glob('*_raw.fif')]

            # Executing the function asynchronously
            future_to_file = {executor.submit(process_file, file, interval_seconds): file for file in raw_files}

            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]

                try:
                    file_results = future.result()
                    for key, value in file_results.items():
                        if key not in all_results:
                            all_results[key] = []
                        all_results[key].extend(value)

                except Exception as exc:
                    print(f'{file!r} generated an exception: {exc}')

    return all_results

def calculate_mean_and_std(results, interval_seconds=1800):
    # First, gather all PSDs for each condition, manipulation type, and interval
    condition_results = {}
    for key, values in results.items():
        rat_id, condition_id, manipulation, interval = key
        if (condition_id, manipulation, interval) not in condition_results:
            condition_results[(condition_id, manipulation, interval)] = []
        condition_results[(condition_id, manipulation, interval)].extend(values)

    # Then calculate the mean and standard deviation
    mean_results = {}
    std_results = {}

    for key, values in condition_results.items():
        all_freqs = [value[0] for value in values]
        all_psds_db = [value[1] for value in values]

        # check if all frequency arrays are equal (they should be)
        if not all(np.array_equal(all_freqs[0], freq) for freq in all_freqs):
            print("Frequency arrays are not equal for condition {}. Cannot calculate mean and std.".format(key[0]))
            return

        mean_psds_db = np.mean(all_psds_db, axis=0)
        std_psds_db = np.std(all_psds_db, axis=0) / np.sqrt(len(all_psds_db))

        mean_results[key] = (all_freqs[0], mean_psds_db)
        std_results[key] = std_psds_db

    return mean_results, std_results

import scipy.stats


def plot_spectral_data(mean_results, std_results, condition_map):
    # Determine the unique manipulations (baseline or manipulation) and intervals
    unique_manipulations = sorted(
        set(key[1] for key in mean_results.keys()))  # 'baseline' first and then 'manipulation'
    unique_intervals = sorted(set(key[2] for key in mean_results.keys()))

    # Create subplots for each interval and manipulation
    fig, axs = plt.subplots(len(unique_intervals), len(unique_manipulations), figsize=(10, 5 * len(unique_intervals)),
                            squeeze=False)
    z_score_95 = 1  # z-score for 95% confidence interval

    # To handle the case where a manipulation doesn't exist for an interval
    if len(unique_manipulations) == 1:
        axs = np.reshape(axs, (-1, 1))

    for key, (freqs, mean_psds_db) in mean_results.items():
        std_psds_db = std_results[key]
        manipulation_index = unique_manipulations.index(key[1])  # find the index of the manipulation
        interval_index = unique_intervals.index(key[2])  # find the index of the interval

        axs[interval_index][manipulation_index].plot(freqs, mean_psds_db,
                                                     label=f'Condition {condition_map.get(str(key[0]), "Unknown")}')
        axs[interval_index][manipulation_index].fill_between(freqs, mean_psds_db - z_score_95 * std_psds_db,
                                                             mean_psds_db + z_score_95 * std_psds_db, alpha=0.3)
        axs[interval_index][manipulation_index].set_title(
            f'Interval {unique_intervals[interval_index]} - {unique_manipulations[manipulation_index].title()}')
        axs[interval_index][manipulation_index].legend()

        if interval_index == len(unique_intervals) - 1:  # if it's the last row, set the xlabel
            axs[interval_index][manipulation_index].set_xlabel('Frequency (Hz)')

    axs[3][0].set_ylabel('Power Spectral Density (relative)')  # set ylabel on the middle subplot

    # removing empty subplots
    for i in range(len(unique_intervals)):
        for j in range(len(unique_manipulations)):
            if not axs[i][j].lines:  # if there is no data for the subplot, remove it
                fig.delaxes(axs[i][j])

    plt.tight_layout()
    plt.show()

def plot_spectral_data_select(mean_results, std_results, condition_map, selected_conditions=None):
    # Determine the unique manipulations (baseline or manipulation) and intervals
    unique_manipulations = sorted(set(key[1] for key in mean_results.keys()))
    unique_intervals = sorted(set(key[2] for key in mean_results.keys()))

    # Create subplots for each interval and manipulation
    fig, axs = plt.subplots(len(unique_intervals), len(unique_manipulations), figsize=(10, 5 * len(unique_intervals)), squeeze=False)
    z_score_95 = 1  # z-score for 95% confidence interval

    # To handle the case where a manipulation doesn't exist for an interval
    if len(unique_manipulations) == 1:
        axs = np.reshape(axs, (-1, 1))

    for key, (freqs, mean_psds_db) in mean_results.items():
        # only plot the selected conditions
        if selected_conditions is not None and str(key[0]) not in selected_conditions:
            continue

        std_psds_db = std_results[key]
        manipulation_index = unique_manipulations.index(key[1])  # find the index of the manipulation
        interval_index = unique_intervals.index(key[2])  # find the index of the interval

        axs[interval_index][manipulation_index].plot(freqs, mean_psds_db, label=f'Condition {condition_map.get(str(key[0]), "Unknown")}')
        axs[interval_index][manipulation_index].fill_between(freqs, mean_psds_db - z_score_95 * std_psds_db, mean_psds_db + z_score_95 * std_psds_db, alpha=0.3)
        axs[interval_index][manipulation_index].set_title(f'Interval {unique_intervals[interval_index]} - {unique_manipulations[manipulation_index].title()}')
        axs[interval_index][manipulation_index].legend()

        if interval_index == len(unique_intervals) - 1:  # if it's the last row, set the xlabel
            axs[interval_index][manipulation_index].set_xlabel('Frequency (Hz)')

    axs[3][0].set_ylabel('Power Spectral Density (relative)')  # set ylabel on the middle subplot

    # removing empty subplots
    for i in range(len(unique_intervals)):
        for j in range(len(unique_manipulations)):
            if not axs[i][j].lines:  # if there is no data for the subplot, remove it
                fig.delaxes(axs[i][j])

    plt.tight_layout()
    plt.show()

def main(n_jobs=1, cro='biotrial'):
    raw_folder = Path('/home/lucky/PycharmProjects/Damona/data/biotrial_data/preprocessed/')
    condition_folders = [folder for folder in raw_folder.iterdir() if folder.is_dir()]

    condition_map = get_condition_map(cro)

    results = calculate_spectra(condition_folders, condition_map)

    mean_results, std_results = calculate_mean_and_std(results)

    plot_spectral_data(mean_results, std_results, condition_map)
    plot_spectral_data_select(mean_results, std_results, condition_map, selected_conditions=['0', '1', '2', '3', '6'])

    print(2)

if __name__ == '__main__':
    main(n_jobs=1)
