# DATA PREPROCCCESSING MODULE


import tempfile
from pydub import AudioSegment, effects
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

# UNDERSTANDING PREPROCESS PARAMETERS FOR CORRECT DIMENSIONS
# Number of frames fixed at 1024, as this is what model expects.
# sr fixed at 256000, as this is the highest expected sampling rate.
# Hyperparameter is segment_length (t), which impacts required hop_length
#
# required hoplength for spectrogram:
# t             sr    hop_length
# 0.1s	1024	256000	25
# 0.5s	1024	256000	125
# 1.0s	1024	256000	250
#
# segment_length = hop_length * (256000 / 1024)


# Preprocess audio file
# This function preprocesses an audio file, normalizes it, computes the mel spectrogram, and returns segments of the spectrogram.
def preprocess(
    filepath:str,
    fmin:int=17000,
    fmax:int=80000,
    n_fft:int=4096,
    hop_length:int=125,  # if you change this, change segment_length accordingly
    segment_length:int=0.5,  # in s, if you change this, change hop_length accordingly
    allowed_length:int=60,  # in s
    target_sr:int=256000,  # resample to highest expected sampling rate
    to_db:bool=True,  # needs to be true, otherwise the spectrogram is ugly
    mean_std:bool=True,  # if True, normalize mean, std
):
    """
    Preprocess an audio file to compute slices of its mel spectrogram of size (n_mels, t) = (512, 1024).

    This function normalizes the audio, 
    computes and normalizes the mel spectrogram using mean and std, 
    returns segments of the spectrogram.

    Args:
        filepath (str): Path to the audio file.
        fmin (int): Minimum frequency for mel spectrogram. (based on general bat frequency range)
        fmax (int): Maximum frequency for mel spectrogram. (based on general bat frequency range)
        n_fft (int): Number of FFT components.
        hop_length (int): Number of samples between successive frames.
        segment_length (float): Length of each segment in seconds.
        allowed_length (float): Maximum allowed length of the audio file in seconds.
        target_sr (int): Highest expected sampling rate for the audio files.
        to_db (bool): Whether to convert the spectrogram to dB scale. (recommended)
        mean_std (bool): Whether to normalize the spectrogram by mean and standard deviation. (recommended)
    Returns:
        list: List of mel spectrogram slices with dimension (n_mels, t) = (512, 1024)
        int: Sampling rate of the audio file. (used for plotting, not required for model)
    
    """
    # filecheck
    if not filepath.endswith(".wav"):
        raise ValueError("Input file must be a .wav file.")
    # length check
    raw = AudioSegment.from_file(filepath)
    if raw.duration_seconds > allowed_length:
        raise ValueError(
            f"Audio file is too long ({raw.duration_seconds:.2f}s), "
            f"maximum allowed length is {allowed_length}s."
        )

    # normalize volume
    normalized = effects.normalize(raw)

    # convert to wav
    tempfile_name = tempfile.mktemp(suffix=".wav")
    normalized.export(tempfile_name, format="wav")

    # Load with librosa, set highest expected sr
    y, sr = librosa.load(tempfile_name, sr=target_sr, mono=True)

    # Set mel bands, because model expects images of size (512, 1024)
    n_mels = 512

    # Compute Mel spectrogram, returns ndarray of shape (n_mels, t)
    # dont set nfft, it is set automatically by librosa...
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,  # columns in S
        win_length=int(sr / 200),  # window length for FFT, 5ms per bin
        n_mels=n_mels,  # rows in S
        fmin=fmin,
        fmax=fmax,
    )

    # Uncomment for debugging/checks
    # print(f"Mel spectrogram shape: {S.shape}, sr: {sr}")
    # plot_spectrogram(S)

    # Normalize by mean and std, greatly improves spectrogram quality
    if mean_std:
        S = (S - np.mean(S)) / np.std(S)

    # Convert to dB scale, greatly improves spectrogram quality
    if to_db:
        S = librosa.power_to_db(S, ref=np.max)

    # Slice spectrogram into equal segments of segment_length
    slices = []
    frames_per_segment = int(
        segment_length * sr / hop_length
    )  # frames per segment, segment length in s
    if S.shape[1] < frames_per_segment:
        raise ValueError(
            f"Audio file is too short ({S.shape[1] * hop_length / sr:.2f}s), "
            f"minimum required length is {segment_length}s."
        )
    for i in range(0, S.shape[1], frames_per_segment):
        slice = S[:, i : i + frames_per_segment]
        # last slice overlaps with previous one
        if slice.shape[1] < frames_per_segment:
            break
        # overlap
        # if slice.shape[1] < frames_per_segment:
        #      slice = S[:, -frames_per_segment:]
        slices.append(slice)

    # Generate time and mel frequency axes
    # times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    # mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

    # return segments
    # sr is retruned to draw the spectrogram plot, this is not required for the model
    return slices, sr


# def filter_slices(slices, id):


# Plot the spectrogram
def plot_spectrogram(
    S,
    path: None | str = None,
    sr:int=256000,
    fmin:int=10000,
    fmax:int=80000,
    hop_length:int=512,
    title:str="Spectrogram",
    hline: None | tuple = None,
):
    """
    Plots mel spectrogram. Not required for model, but useful for debugging and visualization.
    Args:
        S (ndarray): Mel spectrogram to plot, shape (n_mels, t).
        path (str, optional): Path to save the plot. If None, the plot is shown.
        sr (int): Sampling rate of the audio file.
        fmin (int): Minimum frequency for the y-axis of the spectrogram.
        fmax (int): Maximum frequency for the y-axis of the spectrogram.
        hop_length (int): Hop length used for the spectrogram.
        title (str): Title of the plot.
        hline (tuple, optional): Horizontal lines to draw on the plot, 
            e.g. for visualising frequency bands of interest 
            This parameter is used for debugging the data filtering section

    Returns:
        None: Displays or saves the plot.
    """
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(
        S,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{title}")
    plt.tight_layout()
    if hline:
        plt.axhline(y=hline[0])
        plt.axhline(y=hline[1])
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()


# Important! Filtering does not work optimally yet for each bat type.
# hyperparameter -30db
def filter_nonempty_max(
    slices: list,
    bat_min_freq: int,
    bat_max_freq: int,
    threshold_db: int = -30,
    # parameters for mel_freqs must be equal to preprocess parameters
    n_mels: int = 512,
    fmin: int = 17000,
    fmax: int = 80000,
    save_dir: str | None = None,
)-> tuple[list, list]:
    """
    This function filters slices based on the maximum energy in a specific frequency band.
    It keeps slices where the maximum energy in the bat frequency band is above a certain threshold.
    The goal is to filter out empty slices, such that those can be labeled as 'empty'.

    Args:
        slices (list): List of mel spectrogram slices, each with shape (n_mels, t).
        bat_min_freq (int): Minimum frequency for the bat frequency band.
        bat_max_freq (int): Maximum frequency for the bat frequency band.
        threshold_db (int): Threshold in dB for filtering slices. Default is -30 dB.
        n_mels (int): Number of mel bands. Default is 512.
        fmin (int): Minimum frequency for the mel spectrogram. Default is 17000 Hz.
        fmax (int): Maximum frequency for the mel spectrogram. Default is 80000 Hz.
        save_dir (str | None): Directory to save plots of filtered slices. If None, plots are not saved.

    Returns:
        tuple: A tuple containing two lists:
            - filtered: List of slices that are kept (might contain bat calls).
            - empty: List of slices that will be labeled empty (no bat calls).
    """

    # automate frequency bands based on label
    # bat_min_freq = frequency_bands[en_name][0]
    # bat_max_freq = frequency_bands[en_name][1]

    # Convert expected bat frequency range to Mel bin indices
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    bat_bins = np.where((mel_freqs >= bat_min_freq) & (mel_freqs <= bat_max_freq))[0]

    # print("bat_bins:", bat_bins)
    # print(f"bat_min_freq: {bat_min_freq}, bat_max_freq: {bat_max_freq}")

    filtered = []
    empty = []
    k = 0
    # Determine for each slice whether a bat call might be present or not
    for S in slices:
        if save_dir:
            savepath = os.path.join(save_dir, f"plot_{k}.png")
        else:
            savepath = None

        # Max energy in bat bins (more negative means quieter)
        max_energy = np.max(S[bat_bins, :])

        if max_energy > threshold_db:
            filtered.append(S)
            print(f"slice kept ({k})\nmax energy: {max_energy:.2f} dB")
            plot_spectrogram(
                S,
                path=savepath,
                title=f"kept, max, db: {max_energy:.2f}",
                hline=(bat_min_freq, bat_max_freq),
            )
        else:
            empty.append(S)
            print(f"slice deleted ({k})\nmax energy: {max_energy:.2f} dB")
            plot_spectrogram(
                S,
                path=savepath,
                title=f"deleted, max, db: {max_energy:.2f}",
                hline=(bat_min_freq, bat_max_freq),
            )
        k += 1
    return filtered, empty


# filtering seems to work for sample of myotis, pipistrellus,
# something wrong for brown long eared bat..?




# Functions below are not part of model pipeline

def plot_mean_energy_histogram(
    slices: list,
    # parameters for mel_freqs must be equal to parameters used for preprocess function
    fmin: int = 17000,
    fmax: int = 80000,
    n_mels: int = 512,
    bat_min_freq: int = 41000,
    bat_max_freq: int = 48000,
    title: str = "Histogram of Mean Energy in Bat Frequency Band",
):
    """
    This function was used to inform the choice of threshold for filtering slices.
    It plots a histogram of the mean energy in a specific frequency band for each slice.
    This is not part of the model pipeline.
    """
    # Get mel frequency bins
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    bat_bins = np.where((mel_freqs >= bat_min_freq) & (mel_freqs <= bat_max_freq))[0]

    # Compute mean energy in the bat frequency bins for each slice
    mean_energies = [np.mean(S[bat_bins, :]) for S in slices]

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(mean_energies, bins=100, color="purple", edgecolor="black")
    plt.title(f"{title}")
    plt.xlabel("Mean Energy (dB)")
    plt.ylabel("Number of Slices")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mean_energies


def plot_max_energy_histogram(
    slices: list,
    bat_min_freq: int,
    bat_max_freq: int,
    title="Histogram of Mean Energy in Bat Frequency Band",
    # parameters for mel_freqs must be equal to parameters used for preprocess function
    fmin: int = 17000,
    fmax: int = 80000,
    n_mels: int = 512,
):
    """
    This function was used to inform the choice of threshold for filtering slices.
    It plots a histogram of the mean energy in a specific frequency band for each slice.
    This is not part of the model pipeline.
    """
    # Get mel frequency bins
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    bat_bins = np.where((mel_freqs >= bat_min_freq) & (mel_freqs <= bat_max_freq))[0]

    # compute max energies
    energies = [np.max(S[bat_bins, :]) for S in slices]
    # Compute mean energy in the bat frequency bins for each slice
    # mean_energies = [np.mean(S[bat_bins, :]) for S in slices]

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(energies, bins=50, color="purple", edgecolor="black")
    plt.title(f"{title}")
    plt.xlabel("Mean Energy (dB)")
    plt.ylabel("Number of Slices")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
