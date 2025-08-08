
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
import matplotlib.pyplot as plt
def extract_features(mne_file_path: str, epochs_duration: int, bands: List[Tuple[str, float, float]],
                     intervals: bool, window: int, path_to_results: str, subject: str,
                     session: str, condition: str, rat_id: str, stage: str):
    # Load the MNE file
    mne_file = mne.io.read_raw_fif(mne_file_path)
    mne_file = mne_file.pick_types(eeg=True)

    description = mne_file.info['description']

    match = re.search(r'session_id:\s*(\d+)', description)
    if match:
        session_id = int(match.group(1))

    # Calculate the number of samples for 10 minutes
    samples_per_second = mne_file.info["sfreq"]  # replace with actual sampling rate if different
    samples_per_ten_minutes = samples_per_second * 60 * 10  # 10 minutes

    # Crop the file
    mne_file = mne_file.crop(tmin=0, tmax=samples_per_ten_minutes / mne_file.info['sfreq'])

    #Todo: how do we do with epoched data? concat and detrend?

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


    sensor_df_exponent = features.spectral_exponent_IAF_TF_and_A3_A2_ratio(mne_file)
    # Creating a new figure and an Axes instance
    iaf_df = features.savgol_iaf(mne_file, picks=None,  # noqa: C901
               fmin=4, fmax=15,
               resolution=0.25,
               average=True,
               ax=None,
               window_length=11, polyorder=5,
               pink_max_r2=0.9)

    # UNIVARIATE FEATURES

    # e.g: bandpower per sensor, kyrtosis per sensor

    # Extract univariate, low-level, signal processing features
    try:
        feature_dataframe = features.extract_low_level_features(mne_file, epochs_duration)
    except RuntimeError as e:
        print(e)
        return None  # or an appropriate 'skip' signal for your use case
    feature_dataframe = pd.concat([feature_dataframe.squeeze(), sensor_df_exponent.squeeze(), iaf_df.squeeze()], axis=0)

    # Extract the Hurst Exponenent on the Analytical Signal of the whole
    # whole duration and per 4 frequency bands
    hurst_df = features.calculate_hurst_exp(mne_file, bands, epochs_duration)
    # Add the hurst_exponent to the feature matrix
    feature_dataframe = pd.concat([feature_dataframe.squeeze(), hurst_df.squeeze()], axis=0)
    # Extract the Spectral Centroid and Bandwidth + Features and add them
    # to the feature matrix
    spectral_features = features.calculate_spectral_centroid_and_bandwidth(mne_file)
    # add them to the feature matrix
    feature_dataframe = pd.concat([feature_dataframe, spectral_features.squeeze()], axis=0)

    # Extract the LEMPEL-ZIV and Kolmogorov complexity per SENSOR
    lempel_ziv_df = features.calculate_lempel_ziv_and_Kolmogorov_complexity(mne_file,
        intervals, window, subject, epochs_duration)
    # Extract the PEC per SENSOR
    pac_df = features.calculate_CFC(mne_file, intervals, window, epochs_duration)

    # add them to the feature matrix
    feature_dataframe = pd.concat(
        [feature_dataframe, spectral_features.squeeze(), lempel_ziv_df.squeeze(), pac_df.squeeze()], axis=0
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
        subject,
        session,
    )

    # AGGREGATED NEURAL FEATURES

    if not os.path.exists(path2results):
        os.makedirs(path2results)

    # Save the Neural Features
    neural_agg_fname = os.path.join(path2results, f"rat_{subject}_cond_{condition}_stage_{stage}_neural_features.csv")
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
    if match:
        rat_id = int(match.group(1))
        # Construct the full file path
        file_path = os.path.join(root, file)
        # Call the feature extraction function
        neural_features = extract_features(mne_file_path=file_path, epochs_duration=args.epochs_duration, bands=args.bands,
                             intervals=args.intervals, window=args.window, path_to_results=args.path_to_results,
                             subject=f"rat_{rat_id}", session=condition, condition=condition, rat_id=rat_id,
                             stage=stage)  # Pass the condition type to extract_features
        return neural_features


def process_directory(args):
    final_feature_df_list = []
    # Create a ProcessPoolExecutor with the specified number of processes
    with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
        # Iterate through all the directories and files
        for root, dirs, files in os.walk(args.directory):
            # Process each file in a separate process
            futures = [executor.submit(process_file, args, root, file) for file in files]
            for future in futures:
                result = future.result()
                if result is not None:
                    final_feature_df_list.append(result)
    # Concatenate all DataFrames in the list
    final_df = pd.concat(final_feature_df_list)
    final_df_fname = os.path.join(args.path_to_results, f"neural_features.csv")

    # Save the DataFrame to a CSV file
    final_df.to_csv(final_df_fname, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_duration", default=60, type=int)
    parser.add_argument("-i", "--intervals", default=False, action="store_true")
    parser.add_argument("-w", "--window", default=30, type=int)
    parser.add_argument("-p", "--path_to_results", default="/home/lucky/PycharmProjects/Damona/data/biotrial_results")
    parser.add_argument("-b", "--bands", nargs="+", default=[
        ("Delta", 0.4, 5),
        ("Theta", 5, 8),
        ("Alpha", 8, 13),
        ("Beta1", 14, 18),
        ("Beta2", 19, 30),
        ("Gamma1", 30, 60),
        ("Gamma2", 60, 100),
    ])
    parser.add_argument("-n", "--num_processes", default=1, type=int, help="The number of processes to use")
    parser.add_argument("-d", "--directory", default="/home/lucky/PycharmProjects/Damona/data/biotrial_data/preprocessed")

    args = parser.parse_args()

    process_directory(args)