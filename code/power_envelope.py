import os
import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from mne.time_frequency import psd_array_welch
from scipy.signal import hilbert
import seaborn as sns
import numpy as np
from scipy.ndimage import convolve1d


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


def calculate_spectra(condition_folders, bands=[(5, 10), (10, 15)]):
    results = {}
    baseline_envelopes = {}

    for condition_folder in condition_folders:
        raw_files = [file for file in condition_folder.glob('*_raw.fif')]

        # First, get baseline power envelopes
        for file in raw_files:
            if 'baseline' in file.stem.lower():
                raw = mne.io.read_raw_fif(file, preload=True)
                raw.resample(125)
                description = raw.info['description']
                desc_dict = {item.split(': ')[0]: item.split(': ')[1] for item in description.split(', ')}
                rat_id = desc_dict['rat_id']
                condition = desc_dict['condition_id']
                for band in bands:
                    power_envelope_df = power_envelope(raw, band[0], band[1])
                    key = (rat_id, condition, 'baseline', band)
                    if key not in baseline_envelopes:
                        baseline_envelopes[key] = []
                    baseline_envelopes[key].append(power_envelope_df)

        # Then, get and normalize manipulation power envelopes
        for file in raw_files:
            if 'manipulation' in file.stem.lower():
                raw = mne.io.read_raw_fif(file, preload=True)
                raw.resample(125)
                description = raw.info['description']
                desc_dict = {item.split(': ')[0]: item.split(': ')[1] for item in description.split(', ')}
                rat_id = desc_dict['rat_id']
                condition = desc_dict['condition_id']
                for band in bands:
                    power_envelope_df = power_envelope(raw, band[0], band[1])
                    baseline_key = (rat_id, condition, 'baseline', band)
                    if baseline_key in baseline_envelopes:
                        baseline_power_envelope = np.mean(baseline_envelopes[baseline_key], axis=0)
                        normalized_power_envelope = normalize_power_envelope(power_envelope_df, baseline_power_envelope)
                        key = (rat_id, condition, 'manipulation', band)
                        if key not in results:
                            results[key] = []
                        results[key].append(normalized_power_envelope)

    return results



def power_envelope(raw, fmin, fmax):
    raw_bandpass = raw.copy().filter(fmin, fmax)
    analytic_signal = hilbert(raw_bandpass.get_data()[0])
    power_envelope = np.abs(analytic_signal)**2
    return power_envelope

def normalize_power_envelope(power_envelope, baseline_power_envelope):
    baseline_mean = np.mean(baseline_power_envelope)
    baseline_std = np.std(baseline_power_envelope)
    normalized_power_envelope = (power_envelope - baseline_mean) / baseline_std
    return normalized_power_envelope


def plot_mean_and_error(results, condition_map, bands, sampling_rate, window_size):
    window_size_samples = int(window_size * sampling_rate)  # convert window size to samples

    for band in bands:
        plt.figure(figsize=(10, 6))
        for condition, condition_name in condition_map.items():
            condition_results = [val for key, val in results.items() if
                                 key[1] == condition and key[2] == 'manipulation' and key[3] == band]
            if not condition_results:
                continue  # skip if there are no results for this condition

            all_values = np.array(condition_results)
            mean_values = np.mean(all_values, axis=0)
            error = np.std(all_values, axis=0) / np.sqrt(all_values.shape[0])  # calculate standard error

            # Use cumulative sum to calculate moving averages
            cumsum = np.cumsum(np.insert(mean_values, 0, 0))
            mean_values = (cumsum[window_size_samples:] - cumsum[:-window_size_samples]) / float(window_size_samples)
            mean_values = np.append(mean_values,
                                    np.full(window_size_samples - 1, mean_values[-1]))  # pad end to original length

            cumsum = np.cumsum(np.insert(error, 0, 0))
            error = (cumsum[window_size_samples:] - cumsum[:-window_size_samples]) / float(window_size_samples)
            error = np.append(error, np.full(window_size_samples - 1, error[-1]))  # pad end to original length

            lower_bound = mean_values - error
            upper_bound = mean_values + error

            time = np.arange(0, mean_values.shape[0]) / sampling_rate  # create time array
            time = time / 60  # convert to minutes

            plt.plot(time, mean_values, label=condition_name)
            plt.fill_between(time, lower_bound, upper_bound, alpha=0.2)

        plt.title(f'Power envelope for band {band[0]}-{band[1]} Hz')
        plt.xlabel('Time (min)')
        plt.ylabel('Standard deviations from mean')
        plt.legend()
        plt.xlim(0, 4 * 60)  # set x limits to represent full 4 hours
        plt.show()

def main(n_jobs=1, cro='biotrial'):
    raw_folder = Path('/home/lucky/PycharmProjects/Damona/data/biotrial_data/preprocessed/')
    condition_folders = [folder for folder in raw_folder.iterdir() if folder.is_dir()]
    condition_map = get_condition_map(cro)
    bands = [(10, 20)]  # frequency bands

    results = calculate_spectra(condition_folders, bands)
    sampling_rate = 125
    window_size = 480
    plot_mean_and_error(results, condition_map, bands, sampling_rate, window_size)


if __name__ == '__main__':
    main(n_jobs=1)
