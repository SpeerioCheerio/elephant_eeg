import os
import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mne.time_frequency import psd_array_welch


def power_spectral_analysis(epochs, fmin=0.5, fmax=30):
    # Check if the desired condition '1' is in epochs.event_id
    if '1' not in epochs.event_id:
        return None, None  # Returning None to indicate no proper data

    # Otherwise, proceed with the analysis
    epochs_wake = epochs['1']
    epochs_data = epochs_wake.get_data(picks='EEG')

    # Check if epochs_data is not empty
    if epochs_data.size == 0:
        print("WARNING: Empty epochs data encountered. Skipping.")
        return None, None

    psds, freqs = psd_array_welch(epochs_data[:, 0, :], sfreq=epochs_wake.info['sfreq'], fmin=fmin, fmax=fmax)

    psds_normalized = psds / np.sum(psds, axis=-1, keepdims=True)

    return freqs, psds_normalized


def main(n_jobs=1, cro='biotrial'):

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

    raw_folder = Path('/home/lucky/PycharmProjects/Damona/data/biotrial_data/preprocessed/')
    condition_folders = [folder for folder in raw_folder.iterdir() if folder.is_dir()]

    # Prepare figure for the group plot
    fig, axs = plt.subplots(2, figsize=(14, 8))  # One plot for baseline and one for manipulation

    group_data = {
        'baseline': {},
        'manipulation': {}
    }

    for condition_folder in condition_folders:
        condition = condition_folder.name.replace("condition-", "")
        baseline_files = sorted(condition_folder.glob('*baseline_preprocessed_epochs.fif'))
        manipulation_files = sorted(condition_folder.glob('*manipulation_preprocessed_epochs.fif'))

        psds_baseline_condition = []
        psds_manipulation_condition = []

        for baseline_file, manipulation_file in zip(baseline_files, manipulation_files):
            epochs_baseline = mne.read_epochs(baseline_file)
            epochs_manipulation = mne.read_epochs(manipulation_file)

            freqs, psds_baseline = power_spectral_analysis(epochs_baseline)
            _, psds_manipulation = power_spectral_analysis(epochs_manipulation)

            if freqs is not None and psds_baseline is not None:
                psds_baseline_condition.append(psds_baseline.mean(0))  # Average across epochs for each rat

            if psds_manipulation is not None:
                psds_manipulation_condition.append(psds_manipulation.mean(0))  # Average across epochs for each rat

        # Save condition data to the group_data dictionary
        group_data['baseline'][condition] = np.array(psds_baseline_condition).mean(0)
        group_data['manipulation'][condition] = np.array(psds_manipulation_condition).mean(0)

    # Plotting group PSDs
    for condition, psd in group_data['baseline'].items():
        axs[0].plot(freqs, psd, label=condition_map.get(condition, 'Unknown'))  # Plot baseline PSD

    for condition, psd in group_data['manipulation'].items():
        axs[1].plot(freqs, psd, label=condition_map.get(condition, 'Unknown'))  # Plot manipulation PSD



    axs[0].set_title('Group Average (Baseline)')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Normalized Power Spectral Density')
    axs[0].legend()

    axs[1].set_title('Group Average (Manipulation)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Normalized Power Spectral Density')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('group_spectra.png')
    plt.show()

if __name__ == '__main__':
    main(n_jobs=1)

