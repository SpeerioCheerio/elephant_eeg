
import pandas as pd
import numpy as np
import yasa
import librosa
import neurokit2 as nk
import transfreq
from fooof import FOOOF
from scipy import signal
import mne
from mne_features.feature_extraction import extract_features
from pactools.comodulogram import Comodulogram
from collections import namedtuple  # noqa: I100
import matplotlib.pyplot as plt
import mne  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import center_of_mass
from scipy.signal import argrelmin, savgol_filter
import warnings
import os
warnings.filterwarnings("ignore")

def calculate_hurst_exp(mne_file, iter_freqs, epochs_duration):
    """
    Instead of calculating the Hurst Exponent on the broadband, raw
    time series, we do it on the absolute Hilbert Transform and
    within each different band.
    TODO: Q-Shall we include the Hurst exponenet on generic bands
    (e.g: 4-7 for Theta, or shall we calculate that on individually
     extracted frequency bands?)
    In any case, in the following, I will calculate that for generic bands
    This function returns a dictionary per

    """
    # define the freq bands to calculate and extract the Hurst Exponent
    iter_freqs = iter_freqs
    # initialize hashmap
    collector = {}
    for freq in iter_freqs:
        # transform this into an epochs object, as this is exactly
        # what the MNE features toolbox expects
        mne_file.set_channel_types({'EEG': 'eeg'})
        epochs = mne.make_fixed_length_epochs(
            mne_file.copy(), duration=epochs_duration, verbose=False, preload=True
        )
        # filter in the band of choise
        filtered_epochs = epochs.copy().filter(freq[1], freq[2])
        # get the analytic signal
        hilbert_epochs = filtered_epochs.copy().apply_hilbert(envelope=True)
        # extract the epoched data
        data = hilbert_epochs.get_data()
        # Get the Hurst Exponent
        print(f"Extracting Hurst Exponent for {freq}")
        hurst_exp = extract_features(data, mne_file.info["sfreq"], ["hurst_exp"])
        # append to collector
        collector[f"{freq[0]}"] = np.mean(hurst_exp)

    # tranform the dictionary into a dataframe here index=Channel names
    # columns=Frequency bands
    data = [collector[i] for i in collector.keys()]
    hurst_df = pd.DataFrame(data=data).T
    hurst_df.columns = list(collector.keys())
    hurst_df.index = mne_file.info["ch_names"]
    # add prefix to the daataframe will be easier merged later
    hurst_df = hurst_df.add_prefix("hurst_exp_")

    return hurst_df


def calculate_spectral_centroid_and_bandwidth(mne_file):
    """
    Calculate the spectral centroid and bandwidth on the whole duration and per
    sensor and derive features from it.
    Spectral features
    -----------------
    spectral_centroid gradient (mean value)
    spectral_centroid variance
    spectral_bandwidth (mean value)
    """
    print("Extracting spectral centroid and bandwidth")

    data = mne_file.get_data()
    sr = mne_file.info["sfreq"]
    # extract the spectral centroid
    spectral_centroid = np.squeeze(librosa.feature.spectral_centroid(y=data, sr=sr))
    # extract the spectral bandwidth
    spectral_bandwidth = np.squeeze(
            librosa.feature.spectral_bandwidth(y=data, sr=sr)
        )

    centroid_gradient = np.gradient(spectral_centroid, axis=0)

    # Extract Spectral Centroid Features
    # Todo: Is it correct to use axis = 0 if there is only one channel?
    centroid_gradient_mean = np.mean(centroid_gradient, axis=0)
    centroid_variance = np.var(spectral_centroid, axis=0)
    # Extract Spectral Bandwidth Features
    mean_bandwidth = np.mean(spectral_bandwidth, axis=0)

    data = [centroid_gradient_mean, centroid_variance, mean_bandwidth]
    spectral_df = pd.DataFrame(data=data).T
    spectral_df.columns = [
            "spectral_centroid_gradient",
            "spectral_centroid_variance",
            "mean_bandwidth",
        ]
    spectral_df.index = mne_file.info["ch_names"]

    return spectral_df

def calculate_lempel_ziv_and_Kolmogorov_complexity(mne_file, intervals, window, subject, epochs_duration):
    """
    Calculate the Lempel Ziv complexity and the Kolmogotov
    complexity per channel, by taking the aggregate of
    small epochs (5s).
    """

    print(f"Calculating Lempel-Ziv {subject}")

    duration = epochs_duration  # in seconds
    if intervals:
        if window < duration:
            duration = window

    epochs = mne.make_fixed_length_epochs(
        mne_file, duration=duration, verbose=False, preload=True
    )

    epoched_data = epochs.get_data()

    # create channel collectors
    channel_lempel, channel_kolmogorov = [], []

    for ch, ch_name in enumerate(mne_file.info["ch_names"]):
        ch_data = epoched_data[:, ch, :]
        # Calculate the Lempel-Ziv complexity for each epoch
        lzc_list, kolmogorov_complexity_list = [], []
        for epoch_id in range(0, ch_data.shape[0]):
            signal = ch_data[epoch_id, :]
            lzc, info = nk.complexity_lempelziv(signal)
            # append to lists
            lzc_list.append(lzc)
            kolmogorov_complexity_list.append(info["Complexity_Kolmogorov"])

        # Convert the list of LZC values to a NumPy array
        lzc = np.array(lzc_list).mean()
        kolmogorov = np.array(kolmogorov_complexity_list).mean()
        # append to the channel collectors
        channel_lempel.append(lzc)
        channel_kolmogorov.append(kolmogorov)

    # convert the results into a dataframe
    results = pd.DataFrame(data=[channel_lempel, channel_kolmogorov]).T
    results.columns = ["Lempel_Ziv", "Kolmogorov"]
    results.index = mne_file.info["ch_names"]

    return results


def calculate_CFC(mne_file, intervals, window, epochs_duration):
    """
    Calculate the Cross-Frequency-Coupling (Canolty method)
    per channel, by taking the aggregate of small epochs (5s).
    """

    # create epochs
    duration = epochs_duration  # in seconds
    if intervals:
        if window < duration:
            duration = window

    epochs = mne.make_fixed_length_epochs(
        mne_file, duration=duration, verbose=False, preload=True
    )

    epoched_data = epochs.get_data()

    n_trials = epoched_data.shape[0]

    # Hyperparameters
    fs = mne_file.info["sfreq"]
    low_fq_range = np.linspace(1, 10, 40)
    low_fq_width = 1.0  # Hz
    method = "canolty"

    # define the estimator
    estimator = Comodulogram(
        fs=fs,
        low_fq_range=low_fq_range,
        low_fq_width=low_fq_width,
        method=method,
        progress_bar=True,
        n_jobs=-1,
    )

    channel_collector = []
    # first, we need to loop over channels
    for channel, ch_name in enumerate(mne_file.ch_names):
        trial_collector = []
        # then, over trials
        for trial in range(0, n_trials):
            signal = epoched_data[trial, channel, :]
            # fit the estimator on the trial data
            estimator.fit(signal)
            # now get the max PAC
            pac_values = estimator.get_maximum_pac()
            # append to collector
            trial_collector.append(pac_values)
        # collect all data
        df = pd.DataFrame(trial_collector)
        df.columns = ["low_freq_pac", "high_freq_pac", "max_pac"]
        channel_collector.append(df)

    pac_df = pd.DataFrame(
        [channel_collector[i].mean() for i in range(0, len(channel_collector))]
    )
    pac_df.index = mne_file.info["ch_names"]
    # to avoid zeroing issues
    pac_df["max_pac"] = pac_df["max_pac"] * 1e6

    return pac_df


def calculate_wmsi(raw, epochs_duration, bands):
    """
    Calculate the Weighted Spectral Mutual Information
    """
    n_channels = raw.info["nchan"]
    wmsi_matrix = np.zeros((n_channels, n_channels))

    epochs = mne.make_fixed_length_epochs(
        raw, duration=epochs_duration, verbose=False, preload=True
    )
    frequency_bands = {name: (low, high) for name, low, high in bands}

    for fmin, fmax in frequency_bands.values():
        # Compute the cross-spectral density matrix for the frequency band
        csd = mne.time_frequency.csd_morlet(
            epochs, frequencies=[(fmin + fmax) / 2], n_cycles=(fmax - fmin) / 2
        )

        # Compute the coherence
        csd_data = csd.get_data()
        coh = np.abs(csd_data) ** 2
        coh /= np.abs(np.diag(csd_data))[:, np.newaxis] * np.abs(np.diag(csd_data))

        # Update the wMSI matrix
        wmsi_matrix += coh

        # Normalize the wMSI matrix
    wmsi_matrix /= len(frequency_bands)

    # Get the indices of the lower-triangle, excluding the diagonal
    lower_triangle_indices = np.tril_indices(wmsi_matrix.shape[0], k=-1)

    # Extract the lower-triangle elements
    lower_triangle_elements = wmsi_matrix[lower_triangle_indices]

    # returns the lower-triangle elements

    return lower_triangle_elements

def spectral_exponent_IAF_TF_and_A3_A2_ratio(mne_file):
    """
    Extract the aperiodic part of the power spectrum
    author: Thomas
    This method performs the following tasks:
        1. Computes and return the spectral (aperiodic) exponent both for each
           sensor but also for the average power spectrum.
        2. Corrects the Power Spectrum by removing the aperiodic component.
        3. Calculates and returns the IAF from the corrected Power Spectrum
    Returns two dataframes with this info.
        1. The first contains information per sensor (sensor_df)
        2. The second info extracted from an averaged PSD (average_df)
    """

    # --------------------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------------------
    def __parametrize_psd(psd, freqs, freq_range, avg=True):
        """
        Parametrization of the neural power spectra using the FOOOF toolbox
        as described in Donoghue et al. (2020) Nat Neurosci
        (https://fooof-tools.github.io/)
        INPUTS
        - psd: Power spectrum (Nsensors x Nfreqs)
        - freqs: Frequencies (1 x Nfreqs)
        - avg: Averages across electrodes if 'True' not if 'False'
        - freq_range: Fitting frequency range ([lower_bound, upper_bound])
        OUTPUTS:
        - exp: Spectral exponent (Nsensors x Nfreqs)
        - rsq: R^2 value (indicating goodness-of-fit)
        - pks: Peaks of periodic components (see FOOOF toolbox for details)
        - fm: Full FOOOF object
        """

        def fill_nans(arr):
            df = pd.DataFrame(arr)
            df.interpolate(method="linear", axis=1, inplace=True)
            out = df.to_numpy()
            return out

        # # interpolate line noise
        ln_idx = (freqs > 49) & (freqs < 51)
        psd[:, ln_idx] = np.nan
        psd = fill_nans(psd)

        # Average PSD averaged across all electrodes
        if avg == True:
            fm = FOOOF(
                peak_width_limits=[1, 20], max_n_peaks=15, min_peak_height=0.1
            )

            psd_mean = np.mean(psd, axis=0)

            fm.fit(freqs, psd_mean, freq_range)

            exp = fm.get_params("aperiodic_params", "exponent")
            off = fm.get_params("aperiodic_params", "offset")

        # Fit PSD from each electrode separately
        else:
            num_of_elec = psd.shape[0]
            exp = np.zeros((num_of_elec, 1))
            off = np.zeros((num_of_elec, 1))

            for ielec in range(0, num_of_elec):
                fm = FOOOF(
                    peak_width_limits=[1, 20], max_n_peaks=15, min_peak_height=0.1
                )

                fm.fit(freqs, psd[ielec, :], freq_range)

                exp[ielec, 0] = fm.get_params("aperiodic_params", "exponent")
                off[ielec, 0] = fm.get_params("aperiodic_params", "offset")

        return exp, off, fm

    def __correct_PSD(psd, freqs, exp, off):
        """
        Removes aperiodic component from PSD.
        INPUTS
        - psd: Power spectrum (Nsensors x Nfreqs)
        - freqs: Frequencies (1 x Nfreqs)
        - exp: Spectral exponent
        - off: Offset
        OUTPUTS:
        - corr_psd: Corrected PSD
        """
        corr_psd = np.log10(psd) - (-exp * np.log10(freqs) + off)

        return corr_psd

    def __compute_IAF(psd, freqs, freq_range=[5, 15]):
        """
        Computes individual alpha frequency by fitting a Gaussian
        to the PSD in the defined frequency range. Peaks are detected via
        Scipy's find_peaks, which allows to select peaks via their prominence.
        INPUTS
        - psd: Power spectrum (Nsensors x Nfreqs)
        - freqs: Frequencies (1 x Nfreqs)
        - freq_range: Alpha range ([lower_bound, upper_bound])
        OUTPUTS:
        - iaf: Individual alpha frequency
        """

        num_of_elec = psd.shape[0]

        alpha_idx = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        alpha_freqs = freqs[alpha_idx]

        # Detrend signal to eliminate (remaining) 1/f component
        detr_psd = signal.detrend(psd[:, alpha_idx])

        iaf = np.zeros(num_of_elec) * np.nan
        nan_idx = np.array([])

        for ielec in range(0, num_of_elec):
            try:
                loc, param = signal.find_peaks(detr_psd[ielec, :], prominence=0.05)

                max_loc = loc[np.argmax(param["prominences"])]
                iaf[ielec] = alpha_freqs[max_loc]

            except:
                nan_idx = np.append(nan_idx, ielec)
                print("Could not identify alpha peak...")

        if nan_idx.size > 0:
            # Take max value in range (IAF - 2 Hz) - (IAF + 2 Hz)

            search_range = [
                max(np.nanmean(iaf) - 2, freq_range[0]),
                min(np.nanmean(iaf) + 2, freq_range[1]),
            ]

            lb = np.argmin(abs(alpha_freqs - search_range[0]))
            ub = np.argmin(abs(alpha_freqs - search_range[1]))

            print(alpha_freqs[lb:ub])

            for ielec in nan_idx:
                iaf[int(ielec)] = alpha_freqs[
                    np.argmax(detr_psd[int(ielec), lb:ub]) + lb
                ]

        return iaf

    def __compute_a3a2_ratio(psd, freqs, iaf, tf):
        """
        Compute individual alpha3-alpha2 ratio.
        Method described in Moretti et al. (2013) Front. Aging Neurosci.
        (https://www.frontiersin.org/articles/10.3389/fnagi.2013.00063)
        INPUTS:
        - psd: Power spectrum, corrected for 1/f component (Nsensors x Nfreqs)
        - freqs: Frequencies (1 x Nfreqs)
        - iaf: individual alpha frequency (Nsensors x 1)
        - tf: Transfer frequency (1 x 1)
        OUTPUTS:
        - a3a2: alpha3-alpha2 ratio
        """

        # Number of electrodes
        num_of_elec = psd.shape[0]

        # Range normaliziation of corrected PSD (between 0 and 1)
        min_psd = psd.min(axis=-1).reshape(-1, 1)
        max_psd = psd.max(axis=-1).reshape(-1, 1)
        norm_psd = (psd - min_psd) / (max_psd - min_psd)

        # Compute midpoint (MP) between TF and IAF
        mps = tf + (iaf - tf) / 2

        # Alpha2: From MP to IAF (including both endpoints)
        idx_alpha2 = np.array(
            [(mps[x] <= freqs) & (iaf[x] >= freqs) for x in range(0, num_of_elec)]
        )

        # Alpha3: From IAF to IAF+2 (including both endpoints)
        idx_alpha3 = np.array(
            [
                (iaf[x] <= freqs) & (iaf[x] + 2 >= freqs)
                for x in range(0, num_of_elec)
            ]
        )

        # Compute mean power for alpha2 frequency range
        pow_alpha2 = np.array(
            [
                np.mean(norm_psd[x, idx_alpha2[x]], axis=0, keepdims=True)
                for x in range(0, num_of_elec)
            ]
        )

        # Compute mean power fopr alpha3 frequency range
        pow_alpha3 = np.array(
            [
                np.mean(norm_psd[x, idx_alpha3[x]], axis=0, keepdims=True)
                for x in range(0, num_of_elec)
            ]
        )

        # Compute alpha3-alpha2 ratio
        a3a2 = pow_alpha3 / pow_alpha2

        return a3a2

    ####################################################
    # Exponent extraction
    ####################################################

    # Assume `mne_file` is your raw file
    raw = mne_file

    # Set the parameters
    fmin, fmax = 1., raw.info['sfreq'] / 2.  # The maximum frequency is Nyquist frequency
    n_fft = int(2 * raw.info["sfreq"])  # The length of FFT

    # Compute PSD
    psds, freqs = mne.time_frequency.psd_array_welch(
        raw.get_data(), fmin=fmin, fmax=fmax, sfreq=raw.info['sfreq'],
        n_fft=n_fft)

    psds_welch, freqs_welch = psds, freqs

    # per sensor
    try:
        exp_sensor, off_sensor, _ = __parametrize_psd(
            psds_welch, freqs_welch, freq_range=[1, fmax], avg=False
        )
    except:
        try:
            exp_sensor, off_sensor, _ = __parametrize_psd(
                psds_welch, freqs_welch, freq_range=[2, fmax], avg=False
            )
        except:
            exp_sensor, off_sensor, _ = __parametrize_psd(
                psds_welch, freqs_welch, freq_range=[4, fmax], avg=False
            )
    try:
        # averaged PSD
        exp_average, off_average, _ = __parametrize_psd(
            psds_welch, freqs_welch, freq_range=[1, fmax], avg=True
        )
    except:
        try:
            exp_average, off_average, _ = __parametrize_psd(
                psds_welch, freqs_welch, freq_range=[2, fmax], avg=True
            )
        except:
            exp_average, off_average, _ = __parametrize_psd(
                psds_welch, freqs_welch, freq_range=[4, fmax], avg=True
            )
    ####################################################
    # Correct Power Spectrum
    ####################################################
    # use the values reported at the paper (>2)

    # Assuming mne_file is your Raw instance
    raw = mne_file

    # Set the parameters
    fmin, fmax = 1., raw.info['sfreq'] / 2.  # Frequency range

    # Compute the PSD
    psds, freqs = mne.time_frequency.psd_array_multitaper(raw.get_data(), sfreq=raw.info['sfreq'], fmin=fmin, fmax=fmax)


    # per sensor
    corr_psd_sensor = __correct_PSD(psds, freqs, exp_sensor, off_sensor)

    ####################################################
    # EXTRACT THE IAF (from the corrected PSD)
    ####################################################
    try:
        iaf_sensor = __compute_IAF(corr_psd_sensor, freqs, freq_range=[4, 15])
    except ValueError:
        iaf_sensor = __compute_IAF(corr_psd_sensor, freqs, freq_range=[4, 15])


    ####################################################
    # EXTRACT THE TF (from the corrected PSD)
    ####################################################
    info = transfreq.functions.compute_transfreq(corr_psd_sensor, freqs, method=2)
    tf = info["tf"]

    ####################################################
    # EXTRACT THE A3/A2 ratio
    ####################################################
    # per sensor
    a3a2_ratio_sensor = __compute_a3a2_ratio(corr_psd_sensor, freqs, iaf_sensor, tf)

    ####################################################
    # Collect extracted metrics
    ####################################################

    sensor_df = pd.DataFrame(
        data=[
            exp_sensor.flatten(),
            iaf_sensor.flatten(),
            a3a2_ratio_sensor.flatten(),
            exp_sensor.shape[0] * [tf],
        ]
    ).T
    sensor_df.columns = ["spectral_exponent", "iaf", "a3a2_ratio", "tf"]
    sensor_df.index = mne_file.info["ch_names"]

    return sensor_df



def extract_cordance(absolute_bandpower, relative_bandpower, bands):
    """
    Calculate theta cordance - taken from: https://www.nature.com/articles/npp201223#Sec3
    Returns dict with results saved per bands.
    """

    # user feedback
    print("Extracting frontal cordance")

    # get the bands necessary to calculate the metric
    band_names = [bands[i][0] for i in range(0, len(bands))]
    #rois = list(rois.keys())

    cordance_dict = {}
    #for roi in rois:
    cordance_dict['EEG'] = {}
    for band in band_names:
        cordance_dict['EEG'][band] = {}
        # normalize the band power
        Anorm = absolute_bandpower[band] / absolute_bandpower["TotalAbsPow"]
        Rnorm = relative_bandpower[band] / relative_bandpower["TotalAbsPow"]
        # calculate the cordance for the two frontal sensors
        cordance = np.abs(Anorm['EEG'].mean()) + np.abs(
                Rnorm['EEG'].mean())


        cordance_dict['EEG'][band] = cordance

    return cordance_dict

def extract_asymmetry(feature_dataframe, bands, rois):
    """
    From the feature dataframe, get the bandpower features, and within
    that, calculate the frontal asymmetry per band and returns a dict.
    The asymmetry is calculated per Region of Interest and band.
    """
    # user feedback
    print("Extracting frontal asymmetry")
    # get the bands necessary to calculate the metric
    band_names = [bands[i][0] for i in range(0, len(bands))]

    # inplace func to calculate FAA
    def faa(x, y):
        return np.mean(np.abs(np.log(x) - np.log(y)))

    rois = list(rois.keys())

    asymmetry_dict = {}

    for roi in rois:
        asymmetry_dict[roi] = {}
        for band in band_names:
            asymmetry_dict[roi][band] = {}
            asymmetry_dict[roi][band] = faa(
                feature_dataframe[band][rois[roi]["left"]].mean(),
                feature_dataframe[band][rois[roi]["right"]].mean(),
            )

    return asymmetry_dict


def extract_low_level_features(mne_file, epochs_duration):
    """
    Start at the broadband signal, and extract low-level, signal-processing
    features.
    """


    # first, select the features to track (by default, the calculation) takes
    # place at the trial level - therefore, for the computation, we need to
    # create fixed-length epochs (in a similar manner to the envelope correlation
    # analysis)

    univariate_features = [
        "app_entropy",
        "decorr_time",
        "higuchi_fd",
        "hjorth_complexity",
        "hjorth_complexity_spect",
        "hjorth_mobility",
        "hjorth_mobility_spect",
        "katz_fd",
        "kurtosis",
        "line_length",
        "mean",
        "ptp_amp",
        "quantile",
        "rms",
        "samp_entropy",
        "skewness",
        #"spect_edge_freq",
        "spect_entropy",
        #'spect_slope',
        #'energy_freq_bands',
        "std",
        #"svd_entropy",
        #"svd_fisher_info",
        "variance",
        "zero_crossings",
    ]

    # start by creating the epochs for the feature-extraction
    try:
        epochs = mne.make_fixed_length_epochs(
            mne_file, duration=epochs_duration, verbose=False, preload=True
        )
    except ValueError as e:
        raise RuntimeError(f"Error with file {mne_file}: {e}") from e


    # extract the data
    data = epochs.get_data()
    # loop through the features and calculate per feature to construct a
    # human-readable dataframe
    # first, initialize a hashmap to hold the data
    univariate_extracted_features_dict = {}
    for uni_feature in univariate_features:
        print(f"Extracting {uni_feature, mne_file}")
        # extract features for all trials
        try:
            extracted_feature = extract_features(
                data, mne_file.info["sfreq"], {uni_feature}
            )
        except IndexError:
            nan_array = np.empty((data.shape[0], data.shape[1]))
            nan_array[:] = np.nan
            extracted_feature = nan_array
            print(f" Failed extracting {uni_feature, mne_file}")

        # the feature is calculated per TRIAL, therefore, the shape is
        # N_TRIALS X N_SENSORS
        # Therefore, we make a decision here -> Get the median of the feature
        # for all trials PER SENSOR --> DIM: 1 X SENSORS (vector)
        extracted_feature = np.median(extracted_feature, axis=0)
        # append to the collector
        univariate_extracted_features_dict[uni_feature] = extracted_feature

    # construct the dataframe
    feature_dataframe = pd.DataFrame.from_dict(univariate_extracted_features_dict)
    # set the channel names as index
    feature_dataframe.set_axis(
        mne_file.info["ch_names"], axis="index"
    )

    return feature_dataframe


def savgol_iaf(raw, band, picks=None,  # noqa: C901
               fmin=None, fmax=None,
               resolution=0.1,
               average=True,
               ax=None,
               window_length=11, polyorder=5,
               pink_max_r2=1):
    """Estimate individual alpha frequency (IAF).

    Parameters
    ----------
    raw : instance of Raw
        The raw data to do these estimations on.
    picks : array-like of int | None
        List of channels to use.
    fmin : int | None
        Lower bound of alpha frequency band. If None, it will be
        empirically estimated using a polynomial fitting method to
        determine the edges of the central parabolic peak density,
        with assumed center of 10 Hz.
    fmax : int | None
        Upper bound of alpha frequency band. If None, it will be
        empirically estimated using a polynomial fitting method to
        determine the edges of the central parabolic peak density,
        with assumed center of 10 Hz.
    resolution : float
        The resolution in the frequency domain for calculating the PSD.
    average : bool
        Whether to average the PSD estimates across channels or provide
        a separate estimate for each channel. Currently, only True is
        supported.
    ax : instance of matplotlib Axes | None | False
        Axes to plot PSD analysis into. If None, axes will be created
        (and plot not shown by default). If False, no plotting will be done.
    window_length : int
        Window length in samples to use for Savitzky-Golay smoothing of
        PSD when estimating IAF.
    polyorder : int
        Polynomial order to use for Savitzky-Golay smoothing of
        PSD when estimating IAF.
    pink_max_r2 : float
        Maximum R^2 allowed when comparing the PSD distribution to the
        pink noise 1/f distribution on the range 1 to 30 Hz.
        If this threshold is exceeded, then IAF is assumed unclear and
        None is returned for both PAF and CoG.

    Returns
    -------
    results_iaf_df : df from instance of ``collections.namedtuple`` called IAFEstimate

         df with fields for the peak alpha frequency (PAF),
         alpha center of gravity (CoG), and the bounds of the alpha band
         (as a tuple).

    Notes
    -----
    Based on method developed by
        Corcoran, A. W., Alday, P. M., Schlesewsky, M., &
        Bornkessel-Schlesewsky, I. (2018). Toward a reliable, automated method
        of individual alpha frequency (IAF) quantification. Psychophysiology,
        e13064. doi:10.1111/psyp.13064
    """

    IafEst = namedtuple('IAFEstimate',
                        [f'Peak{band}Frequency', f'{band}CenterOfGravity', f'{band}Band', f'{band}rsquared'])

    n_fft = int(raw.info['sfreq'] / resolution)

    # Calculate the length of your signal
    n_times = raw.n_times

    # Ensure n_fft does not exceed the length of your signal
    n_fft = min(n_fft, n_times)
    spectrum = raw.compute_psd(method="welch", picks=picks, n_fft=n_fft,
                               fmin=1., fmax=30.)
    psd = spectrum.get_data()
    freqs = spectrum.freqs

    if average:
        psd = np.mean(psd, axis=0)

    if fmin is None or fmax is None:
        if fmin is None:
            fmin_bound = 4
        else:
            fmin_bound = fmin

        if fmax is None:
            fmax_bound = 15
        else:
            fmax_bound = fmax

        alpha_search = np.logical_and(freqs >= fmin_bound,
                                      freqs <= fmax_bound)
        freqs_search = freqs[alpha_search]
        psd_search = savgol_filter(psd[alpha_search],
                                   window_length=psd[alpha_search].shape[0],
                                   polyorder=10)
        # argrel min returns a tuple, so we flatten that with [0]
        # then we get the last element of the resulting array with [-1]
        # which is the minimum closest to the 'median' alpha of 10 Hz
        if fmin is None:
            try:
                left_min = argrelmin(psd_search[freqs_search < 10])[0][-1]
                fmin = freqs_search[freqs_search < 10][left_min]
            except IndexError:
                raise ValueError("Unable to automatically determine lower end  of alpha band.")   # noqa: 501
        if fmax is None:
            # here we want the first element of the array which is closest to
            # the 'median' alpha of 10 Hz
            try:
                right_min = argrelmin(psd_search[freqs_search > 10])[0][0]
                fmax = freqs_search[freqs_search > 10][right_min]
            except IndexError:
                raise ValueError("Unable to automatically determine upper end of alpha band.")   # noqa: 501
    psd_smooth = savgol_filter(psd,
                               window_length=window_length,
                               polyorder=polyorder)
    alpha_band = np.logical_and(freqs >= fmin, freqs <= fmax)

    slope, intercept, r, p, se = stats.linregress(np.log(freqs),
                                                  np.log(psd_smooth))
    if r**2 > pink_max_r2:
        paf = None
        cog = None
        print(f'r squared {r**2} > than pink max {pink_max_r2}')

    paf_idx = np.argmax(psd_smooth[alpha_band])
    paf = freqs[alpha_band][paf_idx]

    cog_idx = center_of_mass(psd_smooth[alpha_band])
    try:
        cog_idx = int(np.round(cog_idx[0]))
        cog = freqs[alpha_band][cog_idx]
    except ValueError:
        cog = None
        # set PAF to None as well, because this is a pathological case
        paf = None

    if ax:
        plt_psd, = ax.plot(freqs, psd, label="Raw PSD")
        plt_smooth, = ax.plot(freqs, psd_smooth, label="Smoothed PSD")
        plt_pink, = ax.plot(freqs,
                            np.exp(slope * np.log(freqs) + intercept),
                            label='$1/f$ fit ($R^2={:0.2}$)'.format(r**2))
        plt_paf = ax.axvline(x=paf, color='r', linestyle='--', label='PAF')
        plt_cog = ax.axvline(x=cog, color='b', linestyle='--', label='COG')
        try:
            plt_search, = ax.plot(freqs_search, psd_search,
                                  label='Alpha-band Search Parabola')
            ax.legend(handles=[plt_psd, plt_smooth, plt_search, plt_pink, plt_paf, plt_cog])

        except UnboundLocalError:
            # this happens when the user fully specified an alpha band
            ax.legend(handles=[plt_psd, plt_smooth, plt_pink, plt_paf, plt_cog])


        ax.set_ylabel("PSD")
        ax.set_xlabel("Hz")

        filename = raw.info['description']
        plt.savefig(f'tmp/IAF_{filename}.png')

    results_iaf_df = pd.DataFrame(IafEst(paf, cog, (fmin, fmax), r**2)).T
    results_iaf_df.columns = [f'{band}BandPAF',f'{band}CenterofGravity', f'Peak{band}Frequency', f'{band}rsquared']
    results_iaf_df.index = raw.info["ch_names"]

    return results_iaf_df
