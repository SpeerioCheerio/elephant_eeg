import os
import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mne.time_frequency import psd_array_welch


def power_spectral_analysis(raw, fmin=0.5, fmax=30):
    psds, freqs = psd_array_welch(raw.get_data()[0], sfreq=raw.info['sfreq'], fmin=fmin, fmax=fmax)
    psd_normalized = psds / np.sum(psds)

    return freqs, psd_normalized


def calculate_spectra(condition_folders, condition_map):
    results = {}

    for condition_folder in condition_folders:
        raw_files = [file for file in condition_folder.glob('*_raw.fif')]

        for file in raw_files:
            raw = mne.io.read_raw_fif(file, preload=True)
            description = raw.info['description']
            # raw.plot(n_channels=30, block=True, scalings='auto')
            #fig = raw.plot_psd(fmin=1, fmax=50)
            print(raw.info['description'])

            # Parse description
            desc_dict = {}
            for item in description.split(', '):
                key, val = item.split(': ')
                desc_dict[key] = val

            rat_id = desc_dict['rat_id']
            condition = desc_dict['condition_id']  # Now we get condition from the description
            print("COND",condition)
            manipulation_type = 'baseline' if 'baseline' in file.stem.lower() else 'manipulation'

            freqs, psds_db = power_spectral_analysis(raw)

            key = (rat_id, condition, manipulation_type)  # key now starts with rat_id
            if key not in results:
                results[key] = []
            results[key].append((freqs, psds_db))

    return results
def calculate_mean_and_std(results):
    # First, gather all PSDs for each condition and manipulation type
    condition_results = {}
    for key, values in results.items():
        rat_id, condition_id, manipulation = key
        if (condition_id, manipulation) not in condition_results:
            condition_results[(condition_id, manipulation)] = []
        condition_results[(condition_id, manipulation)].extend(values)

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

def plot_spectral_data(mean_results, std_results, condition_map):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # two subplots: baseline and manipulation
    z_score_95 = 1  # z-score for 95% confidence interval

    for key, (freqs, mean_psds_db) in mean_results.items():
        std_psds_db = std_results[key]
        axs[0 if key[1] == 'baseline' else 1].plot(freqs, mean_psds_db, label=f'Condition {condition_map.get(str(key[0]), "Unknown")}')
        axs[0 if key[1] == 'baseline' else 1].fill_between(freqs, mean_psds_db - z_score_95 * std_psds_db, mean_psds_db + z_score_95 * std_psds_db, alpha=0.3)

    axs[0].set_title('Baseline')
    axs[1].set_title('Manipulation')
    for ax in axs:
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (dB)')
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_spectral_data_selected(mean_results, std_results, condition_map, selected_conditions=None):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # two subplots: baseline and manipulation
    z_score_95 = 1  # z-score for 95% confidence interval

    for key, (freqs, mean_psds_db) in mean_results.items():
        # only plot the selected conditions
        if selected_conditions is not None and str(key[0]) not in selected_conditions:
            continue

        std_psds_db = std_results[key]
        axs[0 if key[1] == 'baseline' else 1].plot(freqs, mean_psds_db,
                                                   label=f'Condition {condition_map.get(str(key[0]), "Unknown")}')
        axs[0 if key[1] == 'baseline' else 1].fill_between(freqs, mean_psds_db - z_score_95 * std_psds_db,
                                                           mean_psds_db + z_score_95 * std_psds_db, alpha=0.3)

    axs[0].set_title('Baseline')
    axs[1].set_title('Manipulation')
    for ax in axs:
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (dB)')
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_spectral_data_per_rat(results, condition_map, legend_all=True):
    # Get unique rat IDs and conditions
    rats = sorted(set(key[0] for key in results.keys()))
    conditions = sorted(set(str(key[1]) for key in results.keys()))  # condition treated as string

    num_rats = len(rats)
    fig, axs = plt.subplots(num_rats, 2, figsize=(10, num_rats * 5))  # Create subplots with two columns

    # Adjust for case of single rat
    if num_rats == 1:
        axs = np.array([axs])

    # Create a list to store line objects for the legend
    lines = []

    for i, rat_id in enumerate(rats):
        max_psd = float('-inf')
        min_psd = float('inf')
        max_freq = 0

        for condition in conditions:
            for j, manipulation in enumerate(['baseline', 'manipulation']):
                key = (rat_id, condition, manipulation)
                if key in results:
                    for freqs, psd in results[key]:
                        line, = axs[i, j].plot(freqs, psd, label=f'{condition_map.get(condition, "Unknown")}')
                        print('###', f'{condition_map.get(condition, "Unknown")}', condition)
                        if max(psd) > max_psd:
                            max_psd = max(psd)
                        if min(psd) < min_psd:
                            min_psd = min(psd)
                        if max(freqs) > max_freq:
                            max_freq = max(freqs)
                        if i == j == 0:  # store only one line object per condition
                            lines.append(line)

            if legend_all:
                axs[i, 0].legend()
                axs[i, 1].legend()

        for j in range(2):
            axs[i, j].set_ylim([min_psd, max_psd])  # set the y limit
            axs[i, j].set_xlim([0, max_freq])  # set the x limit
            axs[i, j].set_title(f'Rat {rat_id} - {"baseline" if j == 0 else "manipulation"}')
            if i == num_rats - 1:  # x-label only for the last subplot in each column
                axs[i, j].set_xlabel('Frequency (Hz)')

    # y-labels for each column
    fig.text(0.04, 0.5, 'Power Spectral Density (dB)', va='center', rotation='vertical')  # y-label for left column
    fig.text(0.96, 0.5, 'Power Spectral Density (dB)', va='center', rotation='vertical')  # y-label for right column

    # Create the legend from the line objects
    if not legend_all:
        fig.legend(lines, [condition_map.get(condition, "Unknown") for condition in conditions], loc='upper right')

    plt.tight_layout()
    plt.show()




def main(n_jobs=1, cro='biotrial'):
    raw_folder = Path('/home/lucky/PycharmProjects/Damona/data/biotrial_data/preprocessed/')
    condition_folders = [folder for folder in raw_folder.iterdir() if folder.is_dir()]

    condition_map = get_condition_map(cro)

    results = calculate_spectra(condition_folders, condition_map)

    mean_results, std_results = calculate_mean_and_std(results)

    plot_spectral_data(mean_results, std_results, condition_map)
    plot_spectral_data_selected(mean_results, std_results, condition_map, selected_conditions=['0', '1', '2', '3', '6'])

    plot_spectral_data_per_rat(results, condition_map)





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


if __name__ == '__main__':
    main(n_jobs=1)
