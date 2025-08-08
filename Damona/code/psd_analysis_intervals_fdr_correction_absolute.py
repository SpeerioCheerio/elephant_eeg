import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import concurrent.futures
from pathlib import Path
from mne.time_frequency import psd_array_welch
import scipy.stats as stats
from scipy.stats import permutation_test
from statsmodels.stats.multitest import multipletests
import matplotlib.lines as mlines
from matplotlib.cm import get_cmap

def get_condition_map(cro):
    if cro == 'biotrial':
        condition_map = {
            '0': 'Saline',
            '1': 'Vehicle (5 mL/kg, ip)',
            '2': 'DPX-101 (5 mg/kg, ip)',
            '3': 'DPX-101 (15 mg/kg, ip)',
            '4': 'Basmisanil (5 mg/kg, ip)',
            '5': 'MRK016 (3 mg/kg, ip)',
            '6': 'Diazepam (2 mg/kg, ip)'
        }
    elif cro == 'crl':
        condition_map = {
            '1': 'Vehicle',
            '2': 'DPX-101 1(mg / mL)',
            '3': 'DPX-101 3(mg / mL)',
            '4': 'DPX-101 10(mg / mL)',
            '5': 'Diazepam 0.5(mg / mL)',
            '6': 'Diazepam 1(mg / mL)'
        }
    else:
        condition_map = {}

    return condition_map

def power_spectral_analysis(raw, fmin=0.5, fmax=75):
    psds, freqs = psd_array_welch(raw.get_data()[0], sfreq=raw.info['sfreq'], fmin=fmin, fmax=fmax)

    return freqs, psds


import concurrent.futures

def statistic(sample1, sample2, axis=0):
    # Compute the statistic across the specified axis
    return np.mean(sample1, axis=axis) - np.mean(sample2, axis=axis)
def process_file(file, interval_seconds=3600):
    baseline_results = {}
    manipulation_results = {}

    raw = mne.io.read_raw_fif(file, preload=True)
    description = raw.info['description']

    # Parse description
    desc_dict = {}
    for item in description.split(', '):
        key, val = item.split(': ')
        desc_dict[key] = val

    rat_id = desc_dict['rat_id']
    condition = desc_dict['condition_id']
    is_baseline = 'baseline' in file.stem.lower()

    if is_baseline:
        # Process the entire baseline data as one interval
        freqs, psds_db = power_spectral_analysis(raw)
        key = (rat_id, condition, 0)  # Using 0 as the interval number for baseline
        baseline_results[key] = psds_db
    else:
        # Split manipulation data into intervals
        interval_samples = int(interval_seconds * raw.info['sfreq'])
        for start_sample in range(0, len(raw.times), interval_samples):
            end_sample = start_sample + interval_samples
            interval = start_sample // interval_samples

            # Adjust for the maximum time in the raw data
            max_time = raw.times[-1]
            if end_sample / raw.info['sfreq'] > max_time:
                end_sample = int(max_time * raw.info['sfreq'])

            raw_interval = raw.copy().crop(tmin=start_sample / raw.info['sfreq'], tmax=end_sample / raw.info['sfreq'])
            freqs, psds_db = power_spectral_analysis(raw_interval)

            key = (rat_id, condition, interval)
            if key not in manipulation_results:
                manipulation_results[key] = []
            manipulation_results[key].append(psds_db)

    return baseline_results, manipulation_results, freqs

def calculate_spectra(condition_folders, condition_map, interval_seconds=3600, n_jobs=6):
    all_baseline_results = {}
    all_manipulation_results = {}
    difference_results = {}
    p_values=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for condition_folder in condition_folders:
            raw_files = [file for file in condition_folder.glob('*_raw.fif')]

            # Executing the function asynchronously
            future_to_file = {executor.submit(process_file, file, interval_seconds): file for file in raw_files}

            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]

                try:
                    baseline_results, manipulation_results, freqs = future.result()

                    # Aggregate results
                    for key, value in baseline_results.items():
                        if key not in all_baseline_results:
                            all_baseline_results[key] = []
                        all_baseline_results[key].extend(value)

                    for key, value in manipulation_results.items():
                        if key not in all_manipulation_results:
                            all_manipulation_results[key] = []
                        all_manipulation_results[key].extend(value)

                except Exception as exc:
                    print(f'{file!r} generated an exception: {exc}')

    # Calculate differences
    for key, manipulation_values in all_manipulation_results.items():
        # Find the corresponding baseline key (same rat and condition, but interval 0)
        baseline_key = (key[0], key[1], 0)

        if baseline_key in all_baseline_results:
            # Compute mean PSD for the baseline
            baseline = all_baseline_results[baseline_key]

            for manipulation_interval_data in manipulation_values:
                psds_db = manipulation_interval_data

                # Ensure baseline_mean is not zero to avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    percentage_change = (psds_db - baseline) / baseline * 100

                # Handling any potential NaNs due to zero division
                percentage_change = np.nan_to_num(percentage_change, nan=0.0)

                # Store the percentage change in difference_results
                if key not in difference_results:
                    difference_results[key] = []
                difference_results[key].append(percentage_change)
        else:
            print(f"No baseline data found for rat {baseline_key[0]}, condition {baseline_key[1]}.")

    # 'all_manipulation_results' and 'all_baseline_results' are dictionaries
    # with keys (ratid, condition, interval) and arrays of spectra as values

    significance_results = {}
    all_p_values  = {}

    # Group data by condition and interval across all rats
    grouped_manipulation_data = {}
    grouped_baseline_data = {}

    for key, values in all_manipulation_results.items():
        condition, interval = key[1], key[2]
        if (condition, interval) not in grouped_manipulation_data:
            grouped_manipulation_data[(condition, interval)] = []
        grouped_manipulation_data[(condition, interval)].extend(values)

    for key, values in all_baseline_results.items():
        condition = key[1]
        if (condition, 0) not in grouped_baseline_data:
            grouped_baseline_data[(condition, 0)] = []
        grouped_baseline_data[(condition, 0)].append(values)

    # Perform permutation tests for each condition and interval
    for key in grouped_manipulation_data:
        condition, interval = key
        baseline_key = (condition, 0)
        p_values=[]

        if baseline_key in grouped_baseline_data:
            # Pool data across rats for manipulation and baseline
            manipulation_data = np.vstack(grouped_manipulation_data[key])
            baseline_data = np.vstack(grouped_baseline_data[baseline_key])

            for i in range(manipulation_data.shape[1]):
                freq_manipulation_data = manipulation_data[:, i]
                freq_baseline_data = baseline_data[:, i]

                # Debugging print statements
                print(
                    f"Frequency index {i}: manipulation data shape {freq_manipulation_data.shape}, baseline data shape {freq_baseline_data.shape}")

                p_value = permutation_test((
                    freq_manipulation_data, freq_baseline_data),
                    statistic=statistic,
                    permutation_type='independent',
                    n_resamples=10000,
                    vectorized=True
                ).pvalue
                p_values.append(p_value)

            # Apply FDR correction
            p_adj = multipletests(p_values, method='fdr_bh')[1]

            # Determine significance
            interval_significance = np.array(p_adj) < 0.05

            # Store results
            significance_results[key] = interval_significance
            all_p_values[key] = p_adj

    return difference_results, significance_results, freqs, all_p_values


def plot_group_spectra_with_ci(data, condition_map, freqs, cro, significance_results):
    """
    Plots group level difference spectra with 95% confidence intervals, grouped by condition and interval.
    Uses condition names from the provided CRO map.

    Parameters:
    data (dict): A dictionary where keys are tuples representing (rat, condition, interval)
                 and values are arrays of spectra.
    cro (str): The name of the CRO for condition mapping.
    """

    # Group data by condition and interval
    grouped_data = {}
    for (rat, condition, interval), spectra in data.items():
        key = (condition, interval)
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(spectra)

    # Calculate means and standard errors for each group
    group_means = {}
    group_sems = {}
    for key, spectra_group in grouped_data.items():
        stacked_spectra = np.vstack(spectra_group)
        group_means[key] = np.mean(stacked_spectra, axis=0)
        group_sems[key] = stats.sem(stacked_spectra, axis=0)

    # Plotting
    x_values = freqs
    for key, mean in group_means.items():
        condition_name = condition_map.get(key[0], f"Unknown Condition {key[0]}")
        sem = group_sems[key]
        plt.plot(x_values, mean, label=f'{condition_name}, Interval {key[1]}')
        plt.fill_between(x_values, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.2)

    plt.xlabel('Frequency')
    plt.ylabel('Difference Spectrum Magnitude')
    plt.title(f'Group Level Difference Spectra by Condition and Interval with 95% CI ({cro})')
    plt.legend()
    plt.show()





def plot_group_spectra_with_individual_plots(data, condition_map, freqs, cro, significance_results, all_p_values):
    """
    Plots group level difference spectra with 95% confidence intervals and significance indicators for each interval as an individual figure.
    """
    # Time range dictionary
    time_range_dict = {
        0: "0-2 hours",
        1: "2-4 hours",
        2: "4-6 hours",
        3: "6-8 hours",
        4: "8-10 hours",
        5: "10-12 hours",
        6: "12-14 hours",
        7: "14-16 hours",
        8: "16-18 hours",
        9: "18-20 hours"
    }

    # Group data by condition and interval
    grouped_data = {}
    intervals = set()
    conditions = set()
    for (rat, condition, interval), spectra in data.items():
        intervals.add(interval)
        conditions.add(condition)
        key = (condition, interval)
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(spectra)

    # Calculate means and standard errors for each group
    group_means = {}
    group_sems = {}
    for key, spectra_group in grouped_data.items():
        stacked_spectra = np.vstack(spectra_group)
        group_means[key] = np.mean(stacked_spectra, axis=0)
        group_sems[key] = stats.sem(stacked_spectra, axis=0)

    # Sorting intervals and conditions for consistent plotting
    sorted_intervals = sorted(intervals)
    sorted_conditions = sorted(conditions)

    # Define Miami Vice theme colors and assign them to conditions
    miami_vice_colors = ['#0055B7', '#BEBEBE', '#808080']  # '#BEBEBE' instead of light gray
    color_map = {condition: color for condition, color in zip(sorted_conditions, miami_vice_colors)}

    # Define significance symbols
    significance_symbols = {
        0.01: '**',
        0.05: '*'
    }

    # Define the condition for which to show significance
    significance_condition = '3'

    for interval in sorted_intervals:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # Quadratic scaling

        for condition in sorted_conditions:
            key = (condition, interval)
            if key in group_means:
                mean = group_means[key]
                sem = group_sems[key]
                color = color_map.get(condition, '#808080')
                alpha = 0.2  # Alpha for the filled region

                # Plotting the mean spectrum for the condition
                ax.plot(freqs, mean, color=color, label=condition_map.get(condition))

                # Filling between the SEM
                ax.fill_between(freqs, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=alpha)

                # Show significance only for the specified condition
                if condition == significance_condition and key in all_p_values:
                    p_values = all_p_values[key]
                    significant_indices = np.where((p_values < max(significance_symbols.keys())) & significance_results[key])[0]

                    if significant_indices.size > 0:
                        # Grouping continuous indices
                        grouped_indices = np.split(significant_indices, np.where(np.diff(significant_indices) != 1)[0] + 1)

                        for group in grouped_indices:
                            start, end = freqs[group[0]], freqs[group[-1]]
                            ax.hlines(0, start, end, colors='black', lw=4)

                            # Adding significance stars
                            for threshold, symbol in significance_symbols.items():
                                if any(p_values[group] < threshold):
                                    ax.text((start + end) / 2, 0, symbol, color='black', ha='center', va='bottom')

        ax.axhline(y=0, color='gray', linestyle='--')
        interval_title = f'{time_range_dict[interval]} Interval'
        ax.set_title(interval_title, fontsize=14)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Absolute power change to baseline (%)')

        # Add legend for conditions
        ax.legend(loc='upper left')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f'Group Level Change to Baseline Spectra by Condition with 95% CI ({cro})', fontsize=16, y=1.05)

        # Save each plot as an independent figure
        plt.savefig(f'absolute_group_spectra_interval_{interval}.png')
        plt.close(fig)
def plot_group_spectra_with_separate_figures(data, condition_map, freqs, cro):
    """
    Plots group level difference spectra with 95% confidence intervals, separate figures for each condition,
    and each figure contains subplots for each interval. Uses condition names from the provided CRO map.

    Parameters:
    data (dict): A dictionary where keys are tuples representing (rat, condition, interval)
                 and values are arrays of spectra.
    cro (str): The name of the CRO for condition mapping.
    """

    # Group data by condition and interval
    grouped_data = {}
    intervals = set()
    conditions = set()
    for (rat, condition, interval), spectra in data.items():
        intervals.add(interval)
        conditions.add(condition)
        key = (condition, interval)
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(spectra)

    # Calculate means and standard errors for each group
    group_means = {}
    group_sems = {}
    for key, spectra_group in grouped_data.items():
        stacked_spectra = np.vstack(spectra_group)
        group_means[key] = np.mean(stacked_spectra, axis=0)
        group_sems[key] = stats.sem(stacked_spectra, axis=0)

    # Sorting intervals for consistent plotting
    sorted_intervals = sorted(intervals)

    # Creating separate figures for each condition
    for condition in sorted(conditions):
        fig, axs = plt.subplots(len(sorted_intervals), 1, figsize=(10, len(sorted_intervals) * 5))
        if len(sorted_intervals) == 1:
            axs = [axs]  # Make axs a list if only one interval

        for i, interval in enumerate(sorted_intervals):
            key = (condition, interval)
            if key in group_means:
                mean = group_means[key]
                sem = group_sems[key]
                axs[i].plot(freqs, mean, label='Mean Spectrum')
                axs[i].fill_between(freqs, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.2)

            axs[i].set_title(f'Interval {interval}')
            axs[i].set_xlabel('Frequency')
            axs[i].set_ylabel('Difference Spectrum Magnitude')

        plt.tight_layout()
        condition_name = condition_map.get(condition, f"Unknown Condition {condition}")
        plt.suptitle(f'{condition_name}: Group Level Difference Spectra with 95% CI ({cro})', fontsize=16)
        plt.subplots_adjust(top=0.95)  # Adjust the top to fit the suptitle
        plt.show()


def plot_group_spectra_with_subplots_percondition(data, condition_map, freqs, cro):

        """
        Plots group level change to baseline spectra, each condition in a separate figure,
        and each line within a figure representing a different interval with a color gradient.
        Uses condition names from the provided CRO map and saves figures in the specified path.

        Parameters:
        data (dict): A dictionary where keys are tuples representing (rat, condition, interval)
                     and values are arrays of spectra.
        cro (str): The name of the CRO for condition mapping.
        save_path (str): The directory path where the figures will be saved.
        """
        time_range_dict = {
            0: "0-2 hours",
            1: "2-4 hours",
            2: "4-6 hours",
            3: "6-8 hours",
            4: "8-10 hours",
            5: "10-12 hours",
            6: "12-14 hours",
            7: "14-16 hours",
            8: "16-18 hours",
            9: "18-20 hours"
        }
        # Group data by condition and interval
        grouped_data = {}
        intervals = set()
        conditions = set()
        for (rat, condition, interval), spectra in data.items():
            intervals.add(interval)
            conditions.add(condition)
            key = (condition, interval)
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(spectra)

        # Calculate means for each group
        group_means = {}
        for key, spectra_group in grouped_data.items():
            stacked_spectra = np.vstack(spectra_group)
            group_means[key] = np.mean(stacked_spectra, axis=0)

        # Sorting intervals and conditions for consistent plotting
        sorted_intervals = sorted(intervals)
        sorted_conditions = sorted(conditions)

        # Creating a color map for intervals
        cmap = get_cmap('Blues')
        interval_colors = cmap(np.linspace(0.2, 1, len(sorted_intervals)))  # Avoid very light colors

        # Creating individual figures for each condition
        for condition in sorted_conditions:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))  # Quadratic dimension figure

            for interval_index, interval in enumerate(sorted_intervals):
                if (condition, interval) in group_means:
                    mean = group_means[(condition, interval)]
                    interval_label = time_range_dict.get(interval, f'Unknown Interval {interval}')
                    ax.plot(freqs, mean, label=interval_label, color=interval_colors[interval_index])

            ax.axhline(y=0, color='gray', linestyle='--')  # Adding a horizontal line at y=0
            condition_name = condition_map.get(condition, f"Unknown Condition {condition}")
            ax.set_title(f'{condition_name}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Absolute power change to baseline (%)')
            ax.legend()

            # Save the figure
            fig.savefig(f"tmp/{cro}_absolute_condition_{condition}.png", bbox_inches='tight')

            plt.close(fig)  # Close the figure after saving

def main(n_jobs=1, cro='biotrial'):
    # ... [Same as before] ...
    raw_folder = Path('/home/lucky/PycharmProjects/Sama/Damona/data/biotrial_data/preprocessed/')
    condition_folders = [folder for folder in raw_folder.iterdir() if folder.is_dir()]
    # Conditions you are interested in
    desired_conditions = {'condition-3', 'condition-4', 'condition-6'}

    # Filtering folders based on the condition
    condition_folders2 = []
    for folder in raw_folder.iterdir():
        if folder.is_dir():
            # Extract the condition from the folder name
            folder_name = folder.name
            if folder_name in desired_conditions:
                condition_folders2.append(folder)



    condition_map = get_condition_map(cro)
    difference_results, significance_results, freqs, all_p_values = calculate_spectra(condition_folders2, condition_map)

    # Plot the results
    plot_group_spectra_with_individual_plots(difference_results, condition_map, freqs, cro, significance_results, all_p_values)
    plot_group_spectra_with_subplots_percondition(difference_results, condition_map, freqs, cro)

    plot_group_spectra_with_separate_figures(difference_results, condition_map, freqs, cro)
if __name__ == '__main__':
    main(n_jobs=1)
