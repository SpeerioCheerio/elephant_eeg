import os
import numpy as np
import h5py
import scipy.io as sio
import pandas as pd
from pathlib import Path


def map_sessions_to_id(sessions_encoded_csv):
    sessions_df = pd.read_csv(sessions_encoded_csv)
    sessions_df.set_index("Rat No.", inplace=True)
    session_id_dict = sessions_df.to_dict('index')
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
    biotrial_dir = base_path / 'data' / 'biotrial_data'
    raw_folder = biotrial_dir / 'raw'

    save_folder_base = biotrial_dir / 'sleep_raw'
    os.makedirs(save_folder_base, exist_ok=True)

    # encoded session csv
    sessions_encoded_csv = base_path / 'data' / 'biotrial_info' / 'sessions_encoded.csv'
    sessions_encoded_csv = str(sessions_encoded_csv)
    session_id_dict = map_sessions_to_id(sessions_encoded_csv)

    # Loop over rats
    rat_folders = [folder for folder in raw_folder.iterdir() if folder.is_dir()]

    for rat_folder in rat_folders:
        mat_files = list(rat_folder.glob('*.mat'))

        for mat_file in mat_files:
            print(mat_file)

            # load mat file
            mat_data = sio.loadmat(mat_file)
            rec_data = mat_data['REC'][0][0]
            # metadata
            try:
                treatment_data = rec_data['Treatment'][0][0] if rec_data['Treatment'][0].size != 0 else "Saline"
            except IndexError:
                print(f"IndexError occurred while processing file: {mat_file}")
                treatment_data = "Saline"
            rat_id = rec_data['RatID'][0][0]
            session_id = rec_data['SessionID'][0][0]
            condition_id = session_id_dict.get(rat_id, {}).get(f'Session {session_id}', 'Unknown')
            if condition_id == "Unknown":
                print(rat_id, session_id, condition_id)
            dosing_offset_sec = rec_data['DosingOffsetSec'][0][0]

            # output file
            condition_folder = save_folder_base / f"condition-{condition_id}"
            condition_folder.mkdir(parents=True, exist_ok=True)
            output_h5 = condition_folder / f"rat_{rat_id}.h5"

            if output_h5.exists() and not overwrite:
                print(f'{str(output_h5)} already exists, not overwriting')
                continue

            print(f'Creating h5 file: {str(output_h5)}')

            # write h5 file
            with h5py.File(output_h5, 'w') as hf:
                # metadata
                hf.attrs['rat_id'] = rat_id
                hf.attrs['treatment_data'] = treatment_data
                hf.attrs['session_id'] = session_id
                hf.attrs['condition_id'] = condition_id
                hf.attrs['dosing_offset_sec'] = dosing_offset_sec

                # (resample and) save the time series
                for record in rec_data:
                    if record.dtype.names is not None:
                        raw_name = str(record['label'][0][0][0])
                        ch_type = parse_rec_label(raw_name)
                        sfreq = int(record['srate'][0][0][0])

                        data = record['data'][0][0].flatten().astype(np.float32)
                        # if sfreq > TARGET_SFREQ:
                        #     num_samples = int(len(data) * TARGET_SFREQ / sfreq)
                        #     data = resample(data, num_samples)

                        if rat_id == 10:
                            if ch_type == 'eeg':
                                ch_type = 'emg'
                            elif ch_type == 'emg':
                                ch_type = 'eeg'

                        dataset = hf.create_dataset(
                            ch_type, data=data[:], compression='gzip')
                        dataset.attrs['sfreq'] = sfreq

