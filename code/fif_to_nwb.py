"""
Ongoing
Could be useful if we want to work with NWB libraries
"""
import mne
from pathlib import Path
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.ecephys import LFP, ElectricalSeries
from datetime import datetime
from dateutil.tz import tzlocal
from uuid import uuid4

base_path = Path(Path.home(), 'Sama', 'Damona')
data_dir = base_path / 'data'

if __name__ == '__main__':
    # filename
    biotrial_mne_dir = data_dir / 'biotrial_data' / 'mne_raw'
    biotrial_nwb_dir = data_dir / 'biotrial_data' / 'nwb_raw'
    condition = 0
    rat_id = 3
    input_file = biotrial_mne_dir / f'condition-{condition}' / f'rat_{rat_id}_manipulation.fif'
    output_file = biotrial_nwb_dir / 'test1.nwb'

    # load a signal
    raw = mne.io.read_raw_fif(input_file, preload=True)
    data = raw.get_data()[:2]

    # file metadata
    session_description = f'Rat {rat_id} - Session {condition} - manipulation'
    # create NWB
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=str(uuid4()),
        session_start_time=datetime.now(tzlocal()),
        session_id="session_1234",
        lab="Biotrial"
    )
    #device and electrodes
    device = nwbfile.create_device(
        name="biotrial", description="wires", manufacturer="homemade"
    )
    electrode_group_eeg = nwbfile.create_electrode_group(
        name="EEG", description="EEG", device=device, location="brain area")
    nwbfile.add_electrode(
        group=electrode_group_eeg, location="brain area")
    electrode_group_emg = nwbfile.create_electrode_group(
        name="EMG", description="EMG", device=device, location="body")
    nwbfile.add_electrode(
        group=electrode_group_emg, location="body")

    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(2)),  # reference row indices 0 to N-1
        description="all electrodes",
    )

    raw_electrical_series = ElectricalSeries(
        name="ElectricalSeries",
        data=data,
        electrodes=all_table_region,
        starting_time=0.0,  # timestamp of the first sample in seconds relative to the session start time
        rate=500.0,  # in Hz
    )
    nwbfile.add_acquisition(raw_electrical_series)

    with NWBHDF5IO(output_file, "w") as io:
        io.write(nwbfile)
