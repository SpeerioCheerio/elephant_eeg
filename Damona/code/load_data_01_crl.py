import os
import mne
import numpy as np
from pathlib import Path
import argparse
import pandas as pd
import re
import datetime
from datetime import datetime as dt
from scipy.stats import zscore
from datetime import datetime as dt, timedelta
def detect_eeg_onset(data, window_size, z_threshold):
    """
    Detect the onset of the actual EEG recording using sliding window and z-score.

    data: array-like, the EEG data
    window_size: int, the size of the sliding window
    z_threshold: float, the z-score threshold to detect the onset of the signal
    """
    # Convert the data to a numpy array for processing
    data = np.array(data)

    # Calculate the z-score of the data
    z_scores = zscore(data)

    # Initialize a variable to keep track of the current position of the sliding window
    window_start = 0

    # Loop while the sliding window is within the bounds of the data
    while window_start + window_size <= len(data):
        # Calculate the maximum z-score within the sliding window
        max_z_score = np.max(z_scores[window_start:window_start+window_size])

        # If the maximum z-score is above the threshold, return the start of this window as the onset
        if max_z_score > z_threshold:
            return window_start

        # Move the sliding window
        window_start += 1

    # If no onset was found, return None
    return None

def map_sessions_to_id(sessions_encoded_csv):
    sessions_df = pd.read_csv(sessions_encoded_csv)
    session_id_dict = {}
    session_dose_dict = {}

    for i in range(0, len(sessions_df.columns), 3):
        rat_column = sessions_df.columns[i]
        condition_column = sessions_df.columns[i + 1]
        dose_column = sessions_df.columns[i + 2]

        if "Rat No. Session" in rat_column and "Condition" in condition_column:
            session_no = int(rat_column.replace("Rat No. Session ", ""))

            for rat_no, condition_id, dose_time in zip(sessions_df[rat_column], sessions_df[condition_column], sessions_df[dose_column]):
                session_id_dict.setdefault(rat_no, {})[session_no] = condition_id
                session_dose_dict.setdefault(rat_no, {})[session_no] = dose_time

    return session_id_dict, session_dose_dict


class DataProcessor:
    CHANNEL_TYPE_DICT = {
        'EEG': 'eeg',
        'EMG': 'emg',
        'Temp': 'misc',  # Assign type "misc" to "Temp" channels
        'SignalStr': 'misc'  # "SignalStr" channel can be considered as 'misc'
    }
    TARGET_SFREQ = 500

    def __init__(self, edf_file: Path, save_folder: Path, session_id_dict: dict, session_time_dict: dict):
        self.edf_file = edf_file
        self.save_folder = save_folder
        self.session_id_dict = session_id_dict
        self.session_time_dict = session_time_dict

    def process(self):
        # Extract rat_id from the filename
        rat_id = re.findall(r'\d+', self.edf_file.stem)[0].lstrip('0')  # Remove leading zeros

        # Extract session and treatment from the folder name
        folder_name_parts = self.edf_file.parent.name.split('_')
        session = folder_name_parts[0].replace("Session", "Session ")
        if 'Baseline' in self.edf_file.parent.name:
            treatment = 'baseline'
        elif 'Tx' in self.edf_file.parent.name:
            treatment = 'manipulation'
        else:
            treatment = 'Unknown'

        session_number = int(session.split(' ')[1])  # Extracting the session number from 'Session X'
        condition_id = self.session_id_dict.get(int(rat_id), {}).get(session_number, 'Unknown')

        print(f"Processing Rat: {rat_id}, Session: {session}, Condition ID: {condition_id}")

        raw = mne.io.read_raw_edf(self.edf_file)

        # n hours into seconds
        n_hours_in_seconds = 4 * 3600  # 2 hours into seconds
        # Define tmax in seconds, take twice the
        tmax = 10 * 3600

        # Check if the file's duration exceeds tmax
        if raw.times[-1] > tmax:

            # Crop the file
            raw.crop(tmin=0, tmax=tmax)

        # If raw.times[-1] is less than or equal to tmax, no cropping is needed
        else:
            print(f"{raw} The file's duration is less than or equal to 24 hours.")

        recording_info = raw.info

        # This will return the meas_date in seconds since 1970-01-01 00:00:00 (Unix time)
        meas_date = recording_info['meas_date']
        time_str = meas_date.strftime("%H:%M")

        dosing_onset = self.session_time_dict[int(rat_id)][session_number]

        # Parse the strings as time
        dosing_onset_time = dt.strptime(dosing_onset, "%H:%M")
        time_str_time = dt.strptime(time_str, "%H:%M")
        print(f'rat: {rat_id}, session no {session_number}, do onset from table {dosing_onset}, sess start {time_str} ')

        # Check if dosing onset time is less than the session start time
        if dosing_onset_time < time_str_time:
            # If it is, add one day to dosing onset time
            dosing_onset_time += timedelta(days=1)

        # Calculate the difference
        time_difference = dosing_onset_time - time_str_time
        time_difference = time_difference.seconds

        raw.info['description'] = f'rat_id: {rat_id}, condition_id: {condition_id},' \
                                  f' treatment: {treatment}, session_id: {session_number}'

        # Update the channel types
        channel_type_dict = {ch_name: self.CHANNEL_TYPE_DICT.get(ch_name, 'misc') for ch_name in raw.ch_names}
        raw.set_channel_types(channel_type_dict)

        # Resample the data
        if raw.info['sfreq'] > self.TARGET_SFREQ:
            raw.resample(self.TARGET_SFREQ)

        condition_folder = self.save_folder / f"condition-{condition_id}"
        condition_folder.mkdir(parents=True, exist_ok=True)

        if treatment == 'baseline':
            save_file = condition_folder / f"rat_{rat_id}_{treatment}_separate.fif"
            raw.save(save_file, overwrite=True)
        elif treatment == 'manipulation':
            # First, save the first hour (assumption: sfreq is in Hz, so 1 hour is self.TARGET_SFREQ * 3600 samples)
            raw_first_hour = raw.copy().crop(tmin=0, tmax=3600)
            save_file_first_hour = condition_folder / f"rat_{rat_id}_baseline.fif"  # saved as the original baseline file
            raw_first_hour.save(save_file_first_hour, overwrite=True)

            max_time = raw.times[-1]
            if time_difference + n_hours_in_seconds > max_time:
                tmax = max_time
            else:
                tmax = time_difference + n_hours_in_seconds
            raw_after_time_difference = raw.copy().crop(tmin=time_difference, tmax=tmax)

            # Detect the actual onset of the recording
            window_size = 100  # Change this based on your specific needs
            z_threshold = 3  # A common choice for the z-score threshold is 3, but you might need to tune this
            onset = detect_eeg_onset(raw_after_time_difference.get_data()[0], window_size, z_threshold)

            # If an onset was found, crop the data from this point onwards
            if onset is not None:
                additional_margin = 60  # in seconds
                raw_after_onset = raw_after_time_difference.crop(tmin=(onset / raw.info['sfreq']) + additional_margin)
                # Convert samples to seconds for the annotation
                onset_in_seconds = onset / raw.info['sfreq']
                # Add the detected onset as an annotation
                onset_annot = mne.Annotations(onset=[time_difference + onset_in_seconds - raw.first_samp / raw.info['sfreq']],
                                              duration=[0],
                                              description=['DetectedOnset'])
                save_file = condition_folder / f"rat_{rat_id}_{treatment}.fif"
                raw_after_onset.save(save_file, overwrite=True)

            else:
                print("No onset detected.")

            # Create an annotation at time_difference
            my_annot = mne.Annotations(onset=[time_difference - raw.first_samp / raw.info['sfreq']],
                                       # convert samples to seconds
                                       duration=[0],
                                       description=['Cutoff'])
            if onset is not None:
                raw.set_annotations(my_annot + onset_annot)

            else:
                raw.set_annotations(my_annot)

            # Plotting

            # Select only EEG channels
            raw.pick_types(meg=False, eeg=True, stim=False)
            # Plotting
            fig = raw.plot(start=time_difference-800, duration=1600, scalings="auto", block=True, show=False)

            # Save the plot
            fig.savefig(f'tmp/EEG_plot_rat_{rat_id}_sess_{session_number}.png')


def main(args):
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    session_id_dict, session_time_dict = map_sessions_to_id(args.sessions_encoded_csv)
    edf_files = list(input_folder.glob('**/*.edf'))

    for edf_file in edf_files:
        DataProcessor(edf_file, output_folder, session_id_dict, session_time_dict).process()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    from settings import data_path

    input_folder = data_path / 'crl_data' / 'raw'
    output_folder = data_path / 'crl_data' / 'mne_raw'
    sessions_encoded_csv = data_path / 'crl_info' / 'sessions_encoded.csv'

    parser.add_argument("--input_folder", help="Folder containing the EDF files.",
                        default=str(input_folder))
    parser.add_argument("--output_folder", help="Folder to save the processed data.",
                        default=str(output_folder))
    parser.add_argument("--sessions_encoded_csv", help="CSV file mapping session to encoded id.",
                        default=str(sessions_encoded_csv))
    args = parser.parse_args()

    main(args)
