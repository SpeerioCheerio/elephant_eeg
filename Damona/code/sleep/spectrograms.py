import numpy as np
from pathlib import Path
import mne
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from lspopt import spectrogram_lspopt


def compute_spectrogram_lspopt(elec_data, sfreq,
                               parameters=None, transformation="log"):
    """
    Compute multi-taper spectrogram
    :param elec_data: 1D-numpy array
        Electrophysiological data
    :param sfreq: float or int
        Sampling frequency
    :param parameters: dict
        win_sec: int - multi taper window
        fmin: float - lower frequency
        fmax: float - upper frequency

    :return:
    Sxx: 2D-numpy array
        Spectrogram
    t: 1D-numpy array
        timestamps
    f: 1D-numpy array
        frequencies
    parameters: dict
        parameters
    """
    # parameters
    default_params = {"win_sec": 30, "fmin": 0.5,"fmax": 18}
    if parameters is None:
        parameters = default_params
    win_sec = parameters.get("win_sec", default_params["win_sec"])
    fmin = parameters.get("fmin", default_params["fmin"])
    fmax = parameters.get("fmax", default_params["fmax"])

    output_params = {"fmin": fmin, "fmax": fmax,"win_sec": win_sec}

    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sfreq)
    assert elec_data.size > 2 * nperseg, 'Data length must be at least 2 * win_sec.'
    f, t, Sxx = spectrogram_lspopt(elec_data, sfreq, nperseg=nperseg, noverlap=0)

    # Tranformations and normalisations
    if transformation == "log":
        if np.any(Sxx == 0):
            Sxx += 0.00000001
        Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]

    return Sxx, t, f, output_params


def plot_spectrogram(spectrogram_array,
                     times,
                     frequencies,
                     trimperc=2.5,
                     cmap='RdBu_r',
                     axe_plot=None,
                     rescale=3600.,
                     start_time=0,
                     title='Spectrogram',
                     colourbar=True):
    """
    Plot a spectrogram from output of compute_spectrogram
    Can save the png.

    :param spectrogram_array: 2D-numpy array
        Spectrogram
    :param times: 1D-numpy array
        timestamps
    :param frequencies: 1D-numpy array
        frequencies
    :param trimperc:
    :param cmap: str
        colormap name
    :param axe_plot: matplotlib axes
        where to plot the figure
    :param rescale: float
        rescale factor of the x-axis (3600 to convert seconds to hours)
    :param start_time: float
        start of the record, between -8*3600 and 16*3600 (0 is midnight)
    :param title: str
        title of the figure
    :param colourbar: bool
        1 to display the colorbar, 0 otherwise

    :return:
    ax: axes matplotlib
        axe of the spectrogram graph
    """
    # data
    t, f, Sxx = times, frequencies, spectrogram_array
    t = (t + start_time) / rescale

    # Normalization
    vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
    norm = Normalize(vmin=vmin, vmax=vmax)

    # axes
    if axe_plot is None:
        fig, axs = plt.subplots(1, 1, figsize=(12, 4))
        ax = np.ravel(axs)[0]
        plt.rcParams.update({'font.size': 20})
    else:
        ax = axe_plot
    ax.set_title(title)

    im = ax.pcolormesh(t, f, Sxx, norm=norm, cmap=cmap, antialiased=True, shading='auto')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_ylim(min(f), max(f))

    # Add colourbar
    if colourbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.95, fraction=0.1, aspect=25)
        cbar.ax.set_ylabel('Log Power (dB / Hz)', rotation=270, labelpad=20)

    if axe_plot is None:
        fig.show()

    return ax


base_path = Path(Path.home(), 'Sama', 'Damona')
data_dir = base_path / 'data'

if __name__ == '__main__':
    # filename
    biotrial_dir = data_dir / 'biotrial_data' / 'mne_raw'
    condition = 6
    rat_id = 7
    session = "baseline"
    filename = biotrial_dir / f'condition-{condition}' / f'rat_{rat_id}_{session}.fif'

    # load a signal
    raw = mne.io.read_raw_fif(filename)
    eeg_data = raw.get_data()[0]
    emg_data = raw.get_data()[1]
    sfreq = raw.info['sfreq']
    # spectrograms
    params = {"win_sec": 30, "fmin": 0.5, "fmax": 35}
    specg_eeg, t_eeg, freq_eeg, _ = compute_spectrogram_lspopt(
        eeg_data, sfreq, transformation="log", parameters=params)

    params = {"win_sec": 30, "fmin": 1, "fmax": 250}
    specg_emg, t_emg, freq_emg, _ = compute_spectrogram_lspopt(
        emg_data, sfreq, transformation="log", parameters=params)

    # plot
    fig, axs = plt.subplots(2, 1, figsize=(15, 8))
    axs = np.ravel(axs)
    rescale = 3600
    plt.rcParams.update({'font.size': 12})

    # spectrogram EEG
    img1 = plot_spectrogram(specg_eeg, t_eeg, freq_eeg, axe_plot=axs[0], rescale=rescale,
                            start_time=0, title='EEG')
    img2 = plot_spectrogram(specg_emg, t_emg, freq_emg, axe_plot=axs[1], rescale=rescale,
                            start_time=0, title='EMG')

    axs[1].set_xlabel('Time (h)')

    plt.show()

