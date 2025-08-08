# Damona

## Biotrial
Study design:

![Study](data/biotrial_info/study.png)

![Study](data/biotrial_info/sessions.png)

## CRL
Study design:

![Study](data/crl_info/treatment_groups.png)

![Study](data/crl_info/sessions.png)

This repository provides a comprehensive data processing pipeline to convert raw EEG and other physiological data from MATLAB format to MNE Python compatible format.
Prerequisites

    Python 3.8+
    mne library for electrophysiological data processing
    numpy and scipy libraries for numerical operations
    scipy.io for MATLAB file loading


## load_data_01_biotrial.py

This script processes raw electrophysiology data in MATLAB format (.mat files) from rat studies, using libraries such as os, mne, numpy, scipy, pandas, and pathlib. It uses parallel processing to accelerate the computation when handling multiple files. Here's how it works:

    Initialization: It begins by setting up the necessary Python libraries for processing and storing the electrophysiology data.

    Mapping Sessions to IDs: A function (map_sessions_to_id) reads in a CSV file that provides a mapping between the session data and encoded IDs. It transforms this CSV into a dictionary structure for efficient lookups.

    Data Processor: The DataProcessor class is initialized with the .mat file, a directory to save the processed data, and the dictionary mapping sessions to IDs. Its primary function, process, handles the loading and processing of the .mat files. It extracts relevant metadata (such as treatment data, rat ID, session ID), processes different signal types, resamples them if necessary, and splits the data into baseline and manipulation periods based on the dosing offset. The processed data is then saved into .fif files (a format used by MNE-python for raw data).

    Upsampling Data: For signal data that is at a lower sampling rate than desired, the repeat_upsample function is used to increase its frequency by repeating elements.

    Processing All Data: The process_rat_data function is used to process all the .mat files in a given folder using parallel computation. It prepares the necessary arguments and uses a multiprocessing pool to process the files in parallel, which greatly speeds up the processing time if many files need to be processed.

    Error Handling: The process_mat_file function wraps around the creation of a DataProcessor object and the call to its process method, catching any errors related to empty or corrupted .mat files and printing a corresponding error message.

    Command-Line Interface: The main function serves as the entry point when the script is run from the command line. It takes an optional argument --n_jobs that specifies the number of parallel jobs to run for data processing. This number can be adjusted depending on the computational resources available on the machine running the script. The function handles the mapping of session IDs, sets up the necessary folder paths, and processes all rat data folders.


## load_data_01_crl.py
This script processes European Data Format (EDF) files and extracts relevant physiological data. This is particularly used for processing rat EEG data, which is stored in EDF format.

The script performs the following operations:

    Session Mapping: It reads a CSV file that maps each rat's session number to an encoded session ID. This CSV file should be formatted with pairs of columns representing the rat's number and the session ID, respectively.

    EDF File Processing: The script scans a specified input folder and its subfolders for EDF files. Each EDF file is processed independently, with the extraction of relevant data and metadata from the file.

    Data Resampling: The data is resampled to a target sampling frequency of 125 Hz if the original sampling frequency is higher.

    Metadata Extraction: It identifies each rat and their treatment condition from the file name and folder structure.

    Data Saving: The processed data, along with the extracted metadata, is saved in a Python-friendly MNE Raw format to a specified output folder, organized by rat number and condition ID.

You can run the script with command line arguments to specify the input EDF file directory, output directory, and the CSV file path for session ID mapping.

This script requires the MNE library to process EDF files and Pandas to handle CSV files and data organization.


## File structure
This dataset contains electrophysiological recordings of rat brain activity, organized by experimental conditions and individual rats. The recordings are in the MNE-Python compatible FIF format. Each individual rat's recordings are represented by two separate files: a "baseline" file and a "manipulation" file. The baseline file contains the normal state of the rat's brain activity, while the manipulation file contains the brain activity after some experimental manipulation.
Raw data is organized as it is (e.g. for crl_data; analog for biotrial data):
```
crl_data
│
└───raw
```
Data after loading and restructuring:
```
crl_data/
│
└───mne_raw/
│
├───condition-1/
│ ├───rat_2_baseline.fif
│ ├───rat_2_manipulation.fif
│ ├───rat_4_baseline.fif
│ ├───rat_4_manipulation.fif
│ ├───...
│
└───condition-2/
├───rat_2_baseline.fif
├───rat_2_manipulation.fif
├───rat_3_baseline.fif
├───rat_3_manipulation.fif
├───...
```




## Sama additions



# Repository Overview

This repository provides a full pipeline for transforming rodent EEG/physiology data from raw recordings into analytical and machine‑learning‑ready outputs. It includes:

- Scripts for data conversion from MATLAB or EDF formats into MNE FIF files
- Preprocessing and sleep scoring
- Feature extraction
- Statistical and spectral analyses
- Model training
- Configuration files and visual assets for two study cohorts (Biotrial and CRL)

---

## Dependencies

The project relies on the following libraries:

- `MNE`
- `YASA`
- `FOOOF`
- `H2O AutoML`
- Other analysis libraries

---

## Data & Metadata

- `data/biotrial_info` and `data/crl_info`: Study design figures, EEG placement images, and session mapping CSVs
- `metadata/biotrial_sleep_thresholds.csv` and `metadata/crl_sleep_thresholds.csv`: Per‑rat sleep scoring thresholds

---

## Key Components

### Data Loading

- **MATLAB to FIF**  
  `load_data_01_biotrial.py`  
  - Reads `.mat` files  
  - Maps session IDs  
  - Handles resampling/upsampling  
  - Splits recordings into baseline vs. manipulation segments

- **EDF to FIF**  
  `load_data_01_crl.py`  
  - Parses EDF files  
  - Detects recording onset via z‑score thresholds  
  - Resamples to 500 Hz  
  - Annotates events and saves segmented FIF files

---

### Preprocessing & Sleep Scoring

Scripts filter signals, create fixed‑length epochs, classify sleep stages, and mark artifacts.

- `preprocess_02_biotrial.py`  
  Uses YASA and custom thresholds to annotate wake/REM/NREM epochs and flag “BAD” segments

- `preprocess_02_crl.py`  
  Mirrors above steps for CRL data using cohort‑specific thresholds

- **Sleep package**:  
  - Hypnogram plotting  
  - Spectrogram computation  
  - Threshold estimation utilities

---

### Feature Extraction

- `features.py` implements EEG feature calculations:
  - Hurst exponent via band‑limited Hilbert transforms
  - Spectral centroid and bandwidth
  - Complexity metrics: Lempel‑Ziv, Kolmogorov
  - Cross‑frequency coupling
  - Bandpower and cordance

- `eeg_feature_extraction_03*.py`:  
  - Orchestrate feature extraction  
  - Interval-based analysis  
  - Merge behavioral annotations  
  - Save per‑rat feature tables

---

### Spectral & Statistical Analyses

- `psd_analysis*.py`:  
  - Compute power spectral density across conditions, time intervals, and behavioral events  
  - Supports FDR and cluster-based corrections  
  - Utilities for visualizing:
    - Bandpower changes  
    - Power envelopes  
    - Radar plots of feature clusters

---

### Modeling

- `modelling_07_binary.py`:  
  Trains binary H2O AutoML models on selected condition pairs with cross‑subject folds and confusion‑matrix visualization

- `modelling_07_multiclass.py`:  
  Extends above to multi‑class classification

- `utils.py`:  
  Helpers for:
  - PSD plotting  
  - Time utilities  
  - Signal interval cropping  
  - AutoML model selection

---

## Usage

The `README.md` outlines:

- Overall goals and prerequisites
- Conversion pipeline
- Session mapping
- Directory structure for processed data



#############################################################################


# Repository Analysis & Build Plan

## 1. Repository Mapping & Dependency Analysis

### Top-level Structure
- **README.md** – high-level project introduction
- **code/** – processing scripts, feature extraction, modeling utilities
- **data/** – cohort documentation and session mappings
- **metadata/** – per-rat sleep-stage thresholds
- **requirements.txt** – pinned Python dependencies

*(Directory listing omitted here for brevity)*

### External Libraries and Roles
- **mne** – EEG I/O, filtering, epoching, resampling  
- **yasa** – band-power computation for spectral/cordance features  
- **fooof** – spectral exponent/aperiodic fitting  
- **neurokit2, librosa, pactools** – signal-processing and cross-frequency metrics  
- **h2o** and **H2OAutoML** – automated model training with cross-validation  
- Other helpers: `numpy`, `pandas`, `matplotlib`, `scipy`, `networkx`, `pynwb`, etc.

### Module Dependency Flow
load_data_01_.py → preprocess_02_.py
(sleep/sleep_classification.py)
→ eeg_feature_extraction_03*.py → features.py
→ psd_analysis*.py
→ modelling_07_*.py (uses utils.py)


---

## 2. Pipeline Overview

### Data Ingestion  
**Files:** `load_data_01_biotrial.py`, `load_data_01_crl.py`  
- Parses raw `.mat` (Biotrial) or `.edf` (CRL) recordings  
- Maps session IDs, resamples to 500 Hz, saves baseline/manipulation FIF segments  

**Inputs:** Raw files, session mapping CSVs (`sessions_encoded.csv`)  
**Outputs:** `condition-{id}/rat_{id}_baseline.fif` and `rat_{id}_manipulation.fif` (+ optional misc channels)  

**Hardcoded:** 2 h baseline & 20 h manipulation windows, fixed sampling rate, per-cohort channel dictionaries

---

### Preprocessing & Sleep Scoring  
**Files:** `preprocess_02_*.py`  
- Filtering (high-pass, notch, low-pass), 4 s epochs  
- Auto sleep staging via thresholds  
- Artifact rejection (EEG/EMG magnitude or flat segments)  

**Calls:** `sleep.sleep_classification.sleep_scoring_from_file` with cohort-specific thresholds  

**Outputs:** `_preprocessed_raw.fif` (with annotations) and `_preprocessed_epochs.fif`  

**Hardcoded:** Thresholds and epoch duration per cohort

---

### Feature Extraction  
**Files:** `eeg_feature_extraction_03*.py`, `features.py`  
- Crops first 10 min, drops “BAD” annotations  
- Computes: spectral exponent, IAF, Hurst exponent, spectral centroid/bandwidth, Lempel-Ziv/Kolmogorov complexity, PAC, bandpower-based cordance/asymmetry  
- Saves per-rat CSVs with condition, rat ID, stage, session metadata  

**Inputs:** Preprocessed FIF files  
**Outputs:** `features/rat_{id}/condition-{id}/..._neural_features.csv`  

**Hardcoded:** Fixed crop, predefined frequency bands, metadata parsing

---

### PSD & Statistical Analysis  
**Files:** `psd_analysis*.py`  
- Welch PSD per recording, aggregated by rat/condition  
- Plots mean & SEM for baseline vs. manipulation  
- Interval analyses, FDR/cluster corrections, event-aligned spectra

---

### Model Training  
**Files:** `modelling_07_binary.py`, `modelling_07_multiclass.py`  
- H2O AutoML with cross-subject folds  
- Binary script: trains on all pairwise condition combinations, logs to Weights&Biases  

**Outputs:** Leaderboard, confusion matrices, feature importance plots  

**Hardcoded:** 4-fold split by rat ID, fixed AutoML params

---

### Utilities  
**Files:** `utils.py`, sleep helpers  
- PSD plotting: `plot_psd_and_save`, `plot_psd_and_save_rawarray`  
- AutoML helpers: `get_best_non_stacked_model`, `analyze_cv_models`  
- Sleep: signal loading, RMS computation, threshold-based scoring

---

## 3. Metadata & Configuration Handling

- **Thresholds:** `metadata/biotrial_sleep_thresholds.csv`, `metadata/crl_sleep_thresholds.csv` – EMG/EEG cut-offs per condition and rat  
- **Session maps:** `data/*_info/sessions_encoded.csv` – used by `map_sessions_to_id`  
- Loaded during preprocessing/sleep scoring to drive stage labeling  

**Improvements:**  
- Consolidate thresholds and maps into structured YAML/JSON  
- Add validation and config path management

---

## 4. File & Directory Conventions

- **Raw:** `data/<cohort>_data/raw`  
- **Processed FIF:** `data/<cohort>_data/mne_raw` and `/preprocessed`  
- **Naming:** `rat_{id}_baseline.fif` / `_manipulation.fif` → `_preprocessed_raw.fif` / `_preprocessed_epochs.fif`  
- **Features:** `data/<cohort>_results/features/rat_<id>/condition-{id}/rat_<id>_cond_<id>_stage_<stage>_neural_features.csv`

---

## 5. Model Training Interface

- H2O AutoML initialized once; folds assigned by rat to avoid leakage  
- Params: `nfolds=5`, `max_models=10`  
- Leaderboard + CV artifacts saved for explanation plots  
- Utility functions analyze per-fold importance & produce confusion matrices  

**Suggestion:** Expose via CLI/Notebook for param selection (conditions, features, metrics)

---

## 6. Refactor Recommendations

- **Data loading:** Merge biotrial/crl loaders into configurable class  
- **Preprocessing:** Factor shared logic; move thresholds to config  
- **Features:** Split into thematic modules (spectral, complexity, connectivity) + registry  
- **Model training:** Encapsulate H2O AutoML into sklearn-style class  
- **General:** Central configuration system to replace manual path handling

---

## 7. Suggested Improvements

- **CI/CD:** GitHub Actions – `black`, `flake8`, `mypy`, unit tests, CSV schema checks  
- **Docs:** Docstrings + type hints; Sphinx API docs  
- **Logging:** Replace prints with `logging`  
- **Performance:** Parallelization controls via CLI; cache intermediates

---

## 8. Future Extension Ideas

- Support EMG, video-based behavior  
- Real-time streaming (LSL) for live staging/artifact detection  
- Streamlit dashboard for hypnograms, spectra, predictions

---

## 9. Agent Collaboration – Master Task List

| #  | Goal                                           | Inputs                       | Outputs                              | Relevant Files                                      | Suggested Function Signature                                     |
|----|------------------------------------------------|------------------------------|--------------------------------------|-----------------------------------------------------|-------------------------------------------------------------------|
| 1  | Centralize configuration in YAML/JSON          | Existing CSVs, paths         | `config.py` + sample configs         | `load_data_01_*`, `preprocess_02_*`, `sleep/...`    | `load_config(path: Path) -> Dict[str, Any]`                       |
| 2  | Abstract unified data loader                   | Raw `.mat` & `.edf`          | DataLoader returning `mne.Raw`       | `load_data_01_biotrial.py`, `load_data_01_crl.py`   | `class DataLoader: def read(self, path: Path) -> mne.io.Raw`      |
| 3  | Shared preprocessing pipeline                  | Raw FIF                      | Preprocessed Raw & Epochs            | `preprocess_02_biotrial.py`, `preprocess_02_crl.py` | `def preprocess(raw: mne.io.Raw, cfg: Dict) -> Tuple[Raw, Epochs]`|
| 4  | Feature registry & modularization              | `features.py`                | Separated modules + registry         | `features.py`, `eeg_feature_extraction_03.py`       | `def register_feature(name: str, func: Callable)`                 |
| 5  | CLI entrypoints for pipeline stages            | Command-line args            | Executable scripts                   | All stage scripts                                   | `def main(args: argparse.Namespace) -> None`                      |
| 6  | Unit tests for sleep scoring and features      | Sample FIF/CSV               | tests/ suite                         | `sleep/sleep_classification.py`, `features.py`      | `def test_sleep_scoring(tmp_path)`                                |
| 7  | Refactor model training into sklearn-like class| Feature CSVs                 | ModelTrainer `.fit` / `.predict`     | `modelling_07_*`, `utils.py`                        | `class AutoMLTrainer(BaseEstimator)`                              |
| 8  | Streamlit dashboard prototype                  | Preprocessed data, CSVs      | Interactive web app                   | `sleep_vis.py`, feature outputs                     | `def launch_dashboard(data_dir: Path)`                           |
| 9  | Documentation build system                     | Docstrings, README           | docs/ website                        | Entire repo                                         | *make docs script*                                                |

---
