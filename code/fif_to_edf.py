"""
Convert all our processed data to edf, for visualisation
"""
import numpy as np
from pathlib import Path
import mne

base_path = Path(Path.home(), 'Sama', 'Damona')
data_dir = base_path / 'data'

if __name__ == '__main__':
    # directories
    biotrial_mne_dir = data_dir / 'biotrial_data' / 'mne_raw'
    biotrial_edf_dir = data_dir / 'biotrial_data' / 'edf_raw'
    conditions = [0, 1, 2, 3, 4, 5, 6]
    sessions = ["baseline", "manipulation"]
    rats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    for condition in conditions:
        for rat_id in rats:
            for session in sessions:
                file_path = biotrial_mne_dir / f'condition-{condition}' / f'rat_{rat_id}_{session}.fif'
                if file_path.exists():
                    output_file = biotrial_edf_dir / f'condition-{condition}' / f'rat_{rat_id}_{session}.edf'
                    # load a signal
                    raw = mne.io.read_raw_fif(file_path)
                    # export to edf
                    mne.export.export_raw(output_file, raw, fmt='edf', add_ch_type=True)

