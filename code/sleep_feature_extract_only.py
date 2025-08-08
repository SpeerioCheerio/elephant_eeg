from tqdm import tqdm
import pandas as pd
import numpy as np
import yasa
import features
import argparse
import mne
import glob
import os
from typing import List, Tuple
import re
from concurrent.futures import ProcessPoolExecutor
import utils
from settings import base_path, data_path
from concurrent.futures import ProcessPoolExecutor, as_completed
import behavioral_features

def process_file(args, root, file):
    # Determine if the file is baseline or manipulation
    if "baseline_preprocessed_raw" in file:
        stage = "baseline"
    elif "manipulation_preprocessed_raw" in file:
        stage = "manipulation"
    else:
        return None  # Skip files that are not baseline or manipulation

    # Extract the condition from the directory name
    condition = os.path.basename(root)
    # Extract the rat id from the file name
    match = re.match(r"rat_(\d+)_.*_preprocessed_raw.fif", file)
    final_feature_df_list = []

    if match:
        rat_id = int(match.group(1))
        # Construct the full file path
        file_path = os.path.join(root, file)

        # Load the data
        mne_file = mne.io.read_raw_fif(file_path)

        start_time, end_time = mne_file.times[0], mne_file.times[-1]

        description = mne_file.info['description']
        # raw.plot(n_channels=30, block=True, scalings='auto')
        # fig = raw.plot_psd(fmin=1, fmax=50)
        print(mne_file.info['description'])

        # Parse description
        desc_dict = {}
        for item in description.split(', '):
            key, val = item.split(': ')
            desc_dict[key] = val

        condition_number = desc_dict['condition_id']
        session_id = desc_dict['session_id']


        # Checking if the file has zero duration
        if start_time == end_time:
            print(f"{mne_file}This file has zero duration, skipping...")


        if args.intervals is True:

            # get the signal length [seconds]
            signal_length = utils.extract_mne_signal_length_in_seconds(mne_file)
            # return the signal intervals for a given window length
            intervals = utils.crop_signal(signal_length, args.window)

            # Run, the feature extraction per interval
<<<<<<< HEAD
            for num_interval, interval in enumerate(intervals):
=======
            for count, interval in enumerate(intervals):
>>>>>>> b48c844 (add interval)
                # cast the interval to the args
                args.interval = interval
                if interval[1] > mne_file.times[-1]:  # Check if tmax exceeds the limit
                    print(f'Skipping interval {interval} because tmax exceeds the file duration.')
                    continue  # Skip the rest of this iteration and proceed with the next one
                tmp_mne_file = mne_file.copy().crop(interval[0], interval[1])

                if tmp_mne_file.n_times == 0 or tmp_mne_file.get_data().size == 0:
                    print(f'Skipping interval {interval} because it results in an empty data object.')
                    continue

                max_time = tmp_mne_file.times[-1]  # Getting the maximum time in the mne file
                # Calculate the number of samples for window minutes
                samples_per_second = args.window * tmp_mne_file.info["sfreq"]
                if samples_per_second / mne_file.info['sfreq'] > max_time:
                    print(f"{mne_file}tmax is too large for this file, skipping...")
                    continue  # This will skip the rest of the current loop iteration

                if mne_file.annotations.duration.size != 0:
                    if mne_file.annotations.duration.min() != mne_file.annotations.duration.max():
                        print(f"{mne_file} Error: Not all epochs have the same duration")
                    behavior_interval_in_seconds = int(mne_file.annotations.duration.min())

                    # Get start and end "indexes" of your interval in seconds
                    start_index, end_index = int(interval[0] / behavior_interval_in_seconds), int(
                        interval[1] / behavior_interval_in_seconds)

                    # Get the descriptions of these annotations
                    descriptions_in_interval = mne_file.annotations.description[start_index:end_index]

                    # Perform value_counts on your data
                    value_counts_series = pd.Series(descriptions_in_interval).value_counts()

                    # Compute sleep wake transitions (fragmentation)
                    transition_count = behavioral_features.count_transitions(pd.Series(descriptions_in_interval))

                    # Predefined Series with specific index labels
                    predefined_series = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                                  index=['WAKE', 'IS/QW', 'NREM', 'SWS', 'REM', 'BAD'])
                    # Update the predefined series with the value_counts result
                    predefined_series.update(value_counts_series)
                    behavior_df = pd.DataFrame(predefined_series).T
                    behavior_df['behavior_state_transitions'] = transition_count
                else:
                    print("mne_file.annotations.duration is empty.")
                    predefined_series = pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                                                  index=['WAKE', 'IS/QW', 'NREM', 'SWS', 'REM', 'BAD'])
                    behavior_df = pd.DataFrame(predefined_series).T
                    behavior_df['behavior_state_transitions'] = np.nan


                try:



                    behavior_df = behavior_df.reset_index(drop=True)
                    behavior_df['condition'] = condition_number
                    behavior_df['rat_id'] = rat_id
                    behavior_df['stage'] = stage
                    behavior_df['session'] = session_id
<<<<<<< HEAD
                    behavior_df['interval'] = num_interval

=======
                    behavior_df['interval']  = count
                    behavior_df['interval_time'] = interval
>>>>>>> b48c844 (add interval)

                    neural_features = behavior_df
                except ValueError:
                    print('value Error', file_path)

                final_feature_df_list.append(neural_features)
            neural_features = pd.concat(final_feature_df_list)
            path2results = os.path.join(
                args.path_to_results,
                "features",
                condition,
            )
            neural_agg_fname = os.path.join(path2results,
                                            f"rat_{rat_id}_cond_{condition}_stage_{stage}_neural_features_interval.csv")
            neural_features.to_csv(neural_agg_fname, header=False, index=False)
            print(neural_features)


        else:


            # Predefined Series with specific index labels
            predefined_series = pd.Series([np.nan, np.nan, np.nan], index=['sleep', 'BAD', 'wake'])

            # Perform value_counts on your data
            value_counts_series = pd.Series(mne_file.annotations.description).value_counts()

            # Update the predefined series with the value_counts result
            predefined_series.update(value_counts_series)
            behavior_df = pd.DataFrame(predefined_series).T
            # Compute sleep wake transitions (fragmentation)
            transition_count = behavioral_features.count_transitions(pd.Series(mne_file.annotations.description))

            behavior_df['sleep_wake_transitions'] = transition_count

            behavior_df['condition'] = condition_number
            behavior_df['rat_id'] = rat_id
            behavior_df['stage'] = stage
            behavior_df['session'] = session_id

            neural_features = behavior_df

        return neural_features


def process_directory(args):
    final_feature_df_list = []
    # Create a ProcessPoolExecutor with the specified number of processes
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        # Get a list of all files to process
        files_to_process = []
        for root, dirs, files in os.walk(args.directory):
            for file in files:
                files_to_process.append((root, file))

        # Create a progress bar
        pbar = tqdm(total=len(files_to_process), desc="Processing files")

        # Process each file in a separate process
        futures = [executor.submit(process_file, args, root, file) for root, file in files_to_process]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                final_feature_df_list.append(result)
            pbar.update(1)

        # Close the progress bar
        pbar.close()

    # Concatenate all DataFrames in the list
    final_df = pd.concat(final_feature_df_list)
    final_df_fname = os.path.join(args.path_to_results, f"neural_features_intervals.csv")

    # Save the DataFrame to a CSV file
    final_df.to_csv(final_df_fname, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_duration", default=240, type=int)
    parser.add_argument("-i", "--intervals", default=True, action="store_true")
    parser.add_argument("-w", "--window", default=1800, type=int)

    results_folder = data_path / 'biotrial_results' / 'features'
    save_folder = data_path / 'biotrial_data' / 'preprocessed'
    parser.add_argument("-p", "--path_to_results", default=results_folder)
    parser.add_argument("-b", "--bands", nargs="+", default=[
        ("Delta", 0.5, 5),
        ("Theta", 5, 10),
        ("Alpha", 10, 13),
        ("Beta1", 14, 18),
        ("Beta2", 19, 30),
        ("Gamma1", 30, 60),
        ("Gamma2", 60, 100),
    ])
    parser.add_argument("-n", "--num_processes", default=20, type=int, help="The number of processes to use")
    parser.add_argument("-d", "--directory", default=save_folder)

    args = parser.parse_args()

    process_directory(args)
