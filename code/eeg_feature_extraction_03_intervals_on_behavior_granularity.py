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
def extract_features(mne_file, epochs_duration: int, bands: List[Tuple[str, float, float]],
                     intervals: bool, window: int, path_to_results: str, subject: str,
                     session: str, condition: str, rat_id: str, stage: str):

    description = mne_file.info['description']

    match = re.search(r'session_id:\s*(\d+)', description)
    if match:
        session_id = int(match.group(1))


    # Access the annotations
    annotations = mne_file.annotations

    # Create a new list of annotations, excluding the 'BAD' ones
    new_annotations = mne.Annotations(onset=[annot['onset'] for annot in annotations if annot['description'] != 'BAD'],
                                      duration=[annot['duration'] for annot in annotations if
                                                annot['description'] != 'BAD'],
                                      description=[annot['description'] for annot in annotations if
                                                   annot['description'] != 'BAD'])

    # Set the annotations in the Raw file to the new annotations
    mne_file.set_annotations(new_annotations)

    # UNIVARIATE FEATURES

    # e.g: bandpower per sensor, kyrtosis per sensor

    # Extract univariate, low-level, signal processing features
    try:
        feature_dataframe = features.extract_low_level_features(mne_file, epochs_duration)
    except RuntimeError as e:
        print(e)
        return None  # or an appropriate 'skip' signal for your use case

    sensor_df_exponent = features.spectral_exponent_IAF_TF_and_A3_A2_ratio(mne_file)

    iaf_theta_df = features.savgol_iaf(mne_file, 'theta', picks=None,  # noqa: C901
               fmin=4, fmax=10,
               resolution=0.1,
               average=True,
               ax=None,
               window_length=11, polyorder=5,
               pink_max_r2=1)

    iaf_alpha_df = features.savgol_iaf(mne_file, 'alpha', picks=None,  # noqa: C901
               fmin=10, fmax=20,
               resolution=0.1,
               average=True,
               ax=None,
               window_length=11, polyorder=5,
               pink_max_r2=1)


    feature_dataframe = pd.concat([feature_dataframe.squeeze(), sensor_df_exponent.squeeze(), iaf_theta_df.squeeze(), iaf_alpha_df.squeeze()], axis=0)

    # Extract the Spectral Centroid and Bandwidth + Features and add them
    # to the feature matrix
    spectral_features = features.calculate_spectral_centroid_and_bandwidth(mne_file)
    # add them to the feature matrix
    feature_dataframe = pd.concat([feature_dataframe, spectral_features.squeeze()], axis=0)


    # Extract the PEC per SENSOR
    pac_df = features.calculate_CFC(mne_file, intervals, window, epochs_duration)

    # add them to the feature matrix
    feature_dataframe = pd.concat(
        [feature_dataframe, spectral_features.squeeze(), pac_df.squeeze()], axis=0
    )

    # get the band names and do an ugly hack because YASA wants the
    # tuple reversed
    band_names = [bands[i][0] for i in range(0, len(bands))]
    lower_bands = [bands[i][1] for i in range(0, len(bands))]
    upper_bands = [bands[i][2] for i in range(0, len(bands))]
    band_collector = []
    for low, high, band_name in zip(lower_bands, upper_bands, band_names):
        band_collector.append(tuple([low, high, band_name]))

    # Extract the ABSOLUTE bandpower per sensor and frequency band
    absolute_bandpower = yasa.bandpower(
        mne_file, relative=False, bands=band_collector
    )
    # Extract the RELATIVE bandpower per sensor and frequency band
    relative_bandpower = yasa.bandpower(
        mne_file, relative=True, bands=band_collector
    )

    # calculate the cordance for all bands
    cordance_dict = features.extract_cordance(absolute_bandpower, relative_bandpower, bands)
    new_cordance_dict = {}
    for k, v in cordance_dict.items():
        for k_, v_ in cordance_dict[k].items():
            new_cordance_dict[f"cordance_{k}_{k_}"] = v_

    absolute_bandpower.columns = 'absolute_' + absolute_bandpower.columns
    relative_bandpower.columns = 'relative_' + relative_bandpower.columns
    # and append to the feature dataframe
    sensor_feature_dataframe = pd.concat(
        [feature_dataframe, absolute_bandpower.iloc[:,0:len(band_names)].squeeze(), relative_bandpower.iloc[:,0:len(band_names)].squeeze()], axis=0
    )


    # append to the combined dataframe
    # create the non-sensor specific features
    cordance_features = pd.DataFrame.from_dict(cordance_dict).T
    cordance_features.rename(columns={0: ""}, inplace=True)
    cordance_features.columns = 'Cordance_' + cordance_features.columns

    neural_features = pd.concat(
        [cordance_features.squeeze(), sensor_feature_dataframe], axis=0
    )
    neural_features = pd.DataFrame(neural_features.T)
    neural_features = neural_features.T

    # SAVE FEATURES

    path2results = os.path.join(
        path_to_results,
        "features",
        session,
    )

    # AGGREGATED NEURAL FEATURES

    if not os.path.exists(path2results):
        os.makedirs(path2results)

    # Save the Neural Features
    neural_agg_fname = os.path.join(path2results, f"rat_{subject}_cond_{condition}_stage_{stage}_neural_features_interval.csv")
    match = re.match(r"condition-(\d+)", condition)
    if match:
        condition_number = int(match.group(1))
    else:
        raise Exception(f"No match found for condition in {condition}. Skipping saving.")

    neural_features['condition'] = condition_number
    neural_features['rat_id'] = rat_id
    neural_features['stage'] = stage
    neural_features['session'] = session_id

    neural_features.to_csv(neural_agg_fname, header=False, index=False)

    return neural_features


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

        # Checking if the file has zero duration
        if start_time == end_time:
            print(f"{mne_file}This file has zero duration, skipping...")


        if args.intervals is True:

            # get the signal length [seconds]
            signal_length = utils.extract_mne_signal_length_in_seconds(mne_file)
            # return the signal intervals for a given window length
            intervals = utils.crop_signal(signal_length, args.window)

            # Run, the feature extraction per interval
            for interval in intervals:
                # cast the interval to the args
                args.interval = interval
                if interval[1] > mne_file.times[-1]:  # Check if tmax exceeds the limit
                    print(f'Skipping interval {interval} because tmax exceeds the file duration.')
                    continue  # Skip the rest of this iteration and proceed with the next one
                tmp_mne_file = mne_file.copy().crop(interval[0], interval[1])

                # Calculate the number of samples for window minutes
                samples_per_second = args.window * mne_file.info["sfreq"]  # replace with actual sampling rate if different

                # Crop the file
                tmp_mne_file = tmp_mne_file.crop(tmin=0, tmax=samples_per_second / tmp_mne_file.info['sfreq'])
                # Load the MNE file
                tmp_mne_file = tmp_mne_file.pick_types(eeg=True)

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

                # extract and save the features


                neural_features = extract_features(tmp_mne_file, epochs_duration=args.epochs_duration,
                                                   bands=args.bands, intervals=args.intervals, window=args.window,
                                                   path_to_results=args.path_to_results,
                                                   subject=f"rat_{rat_id}", session=condition, condition=condition,
                                                   rat_id=rat_id, stage=stage)
                neural_features['interval_start'] = interval[0]
                neural_features['interval_end'] = interval[1]
                try:
                    neural_features = neural_features.reset_index(drop=True)
                    behavior_df = behavior_df.reset_index(drop=True)
                    neural_features = pd.concat([neural_features, behavior_df], axis=1)
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
            # extract and save the features
            neural_features = extract_features(mne_file, epochs_duration=args.epochs_duration,
                                               bands=args.bands, intervals=args.intervals, window=args.window,
                                               path_to_results=args.path_to_results,
                                               subject=f"rat_{rat_id}", session=condition, condition=condition,
                                               rat_id=rat_id, stage=stage)

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
            neural_features = pd.concat([neural_features, behavior_df], axis=1)

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
    parser.add_argument("--epochs_duration", default=4, type=int)
    parser.add_argument("-i", "--intervals", default=True, action="store_true")
    parser.add_argument("-w", "--window", default=4, type=int)

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
    parser.add_argument("-n", "--num_processes", default=100, type=int, help="The number of processes to use")
    parser.add_argument("-d", "--directory", default=save_folder)

    args = parser.parse_args()

    process_directory(args)