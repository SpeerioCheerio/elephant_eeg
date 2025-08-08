import os
import numpy as np
import h5py
import scipy.io as sio
import pandas as pd
from pathlib import Path
import re
import mne


def map_sessions_to_id(sessions_encoded_csv):
    sessions_df = pd.read_csv(sessions_encoded_csv)
    session_id_dict = {}

    # Using 'range(0, len(sessions_df.columns), 2)' to process pairs of columns
    for i in range(0, len(sessions_df.columns), 2):
        rat_column = sessions_df.columns[i]
        session_column = sessions_df.columns[i + 1]

        # Check if the columns are correctly named
        if "Rat No. Session" in rat_column and "Session" in session_column:
            # Get the session number from the 'Rat No. Session' column name
            session_no = int(rat_column.replace("Rat No. Session ", ""))

            # Get the rat numbers and corresponding session IDs
            for rat_no, session_id in zip(sessions_df[rat_column], sessions_df[session_column]):
                session_id_dict.setdefault(rat_no, {})[session_no] = session_id
        else:
            print(f"Columns {rat_column} and {session_column} do not contain the required descriptors.")

    return session_id_dict


def parse_rec_label(label):
    label = label.lower()
    if 'eeg' in label:
        return 'eeg'
    if 'emg' in label:
        return 'emg'
    if 'temp' in label:
        return 'temp'
    if 'acti' in label:
        return 'activity'
    else:
        return 'unknown'


if __name__ == '__main__':
    # params
    overwrite = False

    # folders
    base_path = Path(Path.home(), 'Sama', 'Damona')
    crl_dir = base_path / 'data' / 'crl_data'
    raw_folder = crl_dir / 'raw'

    save_folder_base = crl_dir / 'sleep_raw'
    os.makedirs(save_folder_base, exist_ok=True)

    # encoded session csv
    sessions_encoded_csv = base_path / 'data' / 'crl_info' / 'sessions_encoded.csv'
    sessions_encoded_csv = str(sessions_encoded_csv)
    session_id_dict = map_sessions_to_id(sessions_encoded_csv)

    # Loop over sessions
    session_folders = [folder for folder in raw_folder.iterdir() if folder.is_dir()]

    for session_folder in session_folders:
        # session and condition
        session_folder_name = session_folder.stem
        folder_name_parts = session_folder_name.split('_')
        session = folder_name_parts[0].replace("Session", "Session ")
        session_id = int(session.split(' ')[1])  # Extracting the session number from 'Session X'
        if 'Baseline' in session_folder_name:
            treatment = 'baseline'
        elif 'Tx' in session_folder_name:
            treatment = 'manipulation'
        else:
            treatment = 'Unknown'

        # list of edf files
        edf_files = list(session_folder.glob('*.edf'))

        for edf_file in edf_files:
            # metadata
            rat_id = re.findall(r'\d+', edf_file.stem)[0].lstrip('0')  # Remove leading zeros
            condition_id = session_id_dict.get(int(rat_id), {}).get(session_id, 'Unknown')

            print(edf_file)
            print(f"Processing Rat: {rat_id}, Session: {session_id}, Condition ID: {condition_id}")

            # load edf file
            raw = mne.io.read_raw_edf(edf_file)

            # output file
            condition_folder = save_folder_base / f"condition-{condition_id}"
            condition_folder.mkdir(parents=True, exist_ok=True)
            output_h5 = condition_folder / f"rat_{rat_id}_{treatment}.h5"

            if output_h5.exists() and not overwrite:
                print(f'{str(output_h5)} already exists, not overwriting')
                continue

            print(f'Creating h5 file: {str(output_h5)}')

            # write h5 file
            with h5py.File(output_h5, 'w') as hf:
                # metadata
                hf.attrs['rat_id'] = rat_id
                hf.attrs['treatment'] = treatment
                hf.attrs['session_id'] = session_id
                hf.attrs['condition_id'] = condition_id
                hf.attrs['sfreq'] = raw.info['sfreq']

                # (resample and) save the time series
                for channel in raw.ch_names:
                    data = raw.get_data(channel).flatten().astype(np.float32)
                    ch_type = parse_rec_label(channel)
                    dataset = hf.create_dataset(
                        ch_type, data=data[:], compression='gzip')
                    dataset.attrs['sfreq'] = raw.info['sfreq']

