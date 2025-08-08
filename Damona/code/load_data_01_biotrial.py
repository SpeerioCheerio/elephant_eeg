import os
import mne
import numpy as np
import scipy.io as sio
import pandas as pd
from pathlib import Path
from scipy.signal import resample
from multiprocessing import Pool
import argparse

from settings import base_path


def map_sessions_to_id(sessions_encoded_csv):
    sessions_df = pd.read_csv(sessions_encoded_csv)
    sessions_df.set_index("Rat No.", inplace=True)
    session_id_dict = sessions_df.to_dict('index')
    return session_id_dict


class DataProcessor:
    CHANNEL_TYPE_DICT = {
        'EEG': 'eeg',
        'EMG': 'emg',
        'Temp': 'misc',  # Assign type "misc" to "Temp" channels
        'Activity': 'misc'  # "Activity" channel can be considered as 'misc'
    }
    TARGET_SFREQ = 500

    def __init__(self, mat_file: Path, save_folder: Path, session_id_dict: dict):
        self.mat_file = mat_file
        self.save_folder = save_folder
        self.session_id_dict = session_id_dict

    @staticmethod
    def repeat_upsample(data, target_len):
        old_len = len(data)
        repeat_times = target_len // old_len
        remainder = target_len % old_len
        repeated_data = np.repeat(data, repeat_times)
        # add remaining values at the end
        if remainder > 0:
            repeated_data = np.concatenate((repeated_data, data[:remainder]))
        # truncate or pad to achieve the target length
        if len(repeated_data) > target_len:
            repeated_data = repeated_data[:target_len]
        elif len(repeated_data) < target_len:
            padding = np.zeros(target_len - len(repeated_data))
            repeated_data = np.concatenate((repeated_data, padding))
        return repeated_data

    def process(self, store_misc=False):
        mat_data = sio.loadmat(self.mat_file)

        rec_data = mat_data['REC'][0][0]

        try:
            treatment_data = rec_data['Treatment'][0][0] if rec_data['Treatment'][0].size != 0 else "Saline"
        except IndexError:
            print(f"IndexError occurred while processing file: {self.mat_file}")
            treatment_data = "Saline"
        rat_id = rec_data['RatID'][0][0]
        session_id = rec_data['SessionID'][0][0]
        condition_id = self.session_id_dict.get(rat_id, {}).get(f'Session {session_id}', 'Unknown')
        if condition_id == "Unknown":
            print(rat_id, session_id, condition_id)
        dosing_offset_sec = rec_data['DosingOffsetSec'][0][0]

        ch_names = []
        ch_types = []
        eeg_data = []
        misc_data = []  # List to hold non-resampled misc channel data
        misc_ch_names = []  # List to hold channel names of misc data

        for record in rec_data:
            if record.dtype.names is not None:
                raw_name = str(record['label'][0][0][0])
                parsed_name_type = raw_name.split('-')[-1].strip()
                ch_type = parsed_name_type.split(' ')[-1].strip()
                ch_type = self.CHANNEL_TYPE_DICT.get(ch_type, 'misc')
                parsed_name = parsed_name_type.replace(ch_type, '').strip()
                parsed_name = parsed_name_type.split()[-1]
                sfreq = int(record['srate'][0][0][0])
                dosing_offset_samples = int(dosing_offset_sec * sfreq)

                data = record['data'][0][0].flatten()

                recording_length_hours = len(data) / sfreq / 3600
                print(f"Recording length: {recording_length_hours} hours")

                manipulation_window_h = 20# 20
                baseline_window_h = 2

                start_index = dosing_offset_samples - (baseline_window_h * 3600 * sfreq)
                end_index = dosing_offset_samples + (manipulation_window_h * 3600 * sfreq)
                if start_index < 0:
                    start_index = 0

                cut_data = data[start_index:end_index]  # slice the data

                if ch_type == 'misc':
                    # store non-resampled data for misc channels
                    misc_data.append(cut_data)
                    misc_ch_names.append(parsed_name)

                ch_names.append(parsed_name)
                ch_types.append(ch_type)

                recording_length_hours = len(cut_data) / sfreq / 3600
                print(f"Recording length after cut: {recording_length_hours} hours")
                if sfreq > self.TARGET_SFREQ:
                    num_samples = int(len(cut_data) * self.TARGET_SFREQ / sfreq)
                    data = resample(cut_data, num_samples)
                else:
                    data = self.repeat_upsample(cut_data, len(cut_data) * self.TARGET_SFREQ // sfreq)
                    data = self.repeat_upsample(data, len(eeg_data[0]))  # upsample to the length of the first item
                eeg_data.append(data)

        eeg_length = len(eeg_data[0])
        for i in range(len(eeg_data)):
            if len(eeg_data[i]) > eeg_length:
                eeg_data[i] = eeg_data[i][:eeg_length]

        if rat_id == 10:
            eeg_data[0], eeg_data[1] = eeg_data[1], eeg_data[0]

        info = mne.create_info(ch_names=ch_names, sfreq=self.TARGET_SFREQ, ch_types=ch_types)
        info['description'] = f'rat_id: {rat_id}, condition_id: {condition_id},' \
                              f' treatment: {treatment_data}, session_id: {session_id}'

        condition_folder = self.save_folder / f"condition-{condition_id}"
        condition_folder.mkdir(parents=True, exist_ok=True)

        # For baseline period
        baseline_data = np.array(eeg_data)[:, :baseline_window_h * 3600 * self.TARGET_SFREQ]
        raw_baseline = mne.io.RawArray(baseline_data, info)
        raw_baseline.save(condition_folder / f"rat_{rat_id}_baseline.fif", overwrite=True)
        # For manipulation period
        manipulation_data = np.array(eeg_data)[:, baseline_window_h * 3600 * self.TARGET_SFREQ:(manipulation_window_h+baseline_window_h) * 3600 * self.TARGET_SFREQ]
        raw_manipulation = mne.io.RawArray(manipulation_data, info)
        raw_manipulation.save(condition_folder / f"rat_{rat_id}_manipulation.fif", overwrite=True)

        # Save non-resampled misc data as two raw files
        misc_info = mne.create_info(ch_names=misc_ch_names, sfreq=sfreq, ch_types=['misc'] * len(misc_ch_names))
        misc_info['description'] = info['description']
        if store_misc == True:
            # For baseline period
            misc_baseline_data = np.array(misc_data)[:, :baseline_window_h * 3600 * sfreq]
            misc_raw_baseline = mne.io.RawArray(misc_baseline_data, misc_info)
            misc_raw_baseline.save(condition_folder / f"rat_{rat_id}_baseline_misc.fif", overwrite=True)

            # For manipulation period
            misc_manipulation_data = np.array(misc_data)[:, (manipulation_window_h-baseline_window_h) * 3600 * sfreq:]
            misc_raw_manipulation = mne.io.RawArray(misc_manipulation_data, misc_info)
            misc_raw_manipulation.save(condition_folder / f"rat_{rat_id}_manipulation_misc.fif", overwrite=True)


def process_rat_data(data_folder: Path, save_folder: Path, session_id_dict: dict, n_jobs=1):
    os.makedirs(save_folder, exist_ok=True)
    mat_files = list(data_folder.glob('*.mat'))

    with Pool(processes=n_jobs) as pool:
        processor_args = [(mat_file, save_folder, session_id_dict) for mat_file in mat_files]
        pool.starmap(process_mat_file, processor_args)


def process_mat_file(mat_file: Path, save_folder: Path, session_id_dict: dict):
    try:
        processor = DataProcessor(mat_file, save_folder, session_id_dict)
        processor.process()
    except sio.matlab._miobase.MatReadError:
        print(f"Unable to process {mat_file} because it appears to be empty.")


def main(n_jobs=4, base_path=base_path):
    sessions_encoded_csv = base_path / 'data' / 'biotrial_info' / 'sessions_encoded.csv'
    sessions_encoded_csv = str(sessions_encoded_csv)
    session_id_dict = map_sessions_to_id(sessions_encoded_csv)

    raw_folder = base_path / 'data' / 'biotrial_data' / 'raw'
    save_folder_base = base_path / 'data' / 'biotrial_data' / 'mne_raw'

    rat_folders = [folder for folder in raw_folder.iterdir() if folder.is_dir()]

    for rat_folder in rat_folders:
        save_folder = save_folder_base
        process_rat_data(rat_folder, save_folder, session_id_dict, n_jobs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process rat data.')
    parser.add_argument('--n_jobs', type=int, default=20, help='Number of parallel jobs to run.')
    args = parser.parse_args()

    main(n_jobs=args.n_jobs)
