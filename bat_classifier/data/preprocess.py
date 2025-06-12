# DATA PREPROCCCESSING MODULE


import tempfile
from pydub import AudioSegment, effects
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bat_classifier.data.download_data import raw_path, data_path, label_path
import os

# UNDERSTANDING PREPROCESS PARAMETERS FOR CORRECT DIMENSIONS
# Number of frames fixed at 1024, as this is what model expects.
# sr fixed at 256000, as this is the highest expected sampling rate.
# Hyperparameter is segment_length (t), which impacts required hop_length
#
# required hoplength for spectrogram:
# t     #frames  sr      hop_length
# 0.1s	1024	 256000	 25
# 0.5s	1024	 256000	 125
# 1.0s	1024	 256000	 250
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
    segment_length:float=0.5,  # in s, if you change this, change hop_length accordingly
    allowed_length:float=60.0,  # in s
    target_sr:int=256000,  # resample to highest expected sampling rate
    to_db:bool=True,  # True recommended, model performs better with dB scale
    mean_std:bool=True,  # True recommended, model performs better with normalized spectrogram
):
    """
    This function preprocesses a single audio file by:
    normalizing its volume, computing the mel spectrogram,
    normalizing mean and standard deviation, converting to dB scale,
    and slicing the spectrogram into segments of a specified length.

    IMPORTANT: Change segment_length / hop_length in accordance with the formula:
        segment_length = hop_length * (256000 / 1024)
    This ensures that the mel spectrogram has the correct dimensions (512, 1024) for each time slice.

    Args:
        filepath (str): Path to the audio file.
        fmin (int): Minimum frequency for the mel spectrogram in Hz.
        fmax (int): Maximum frequency for the mel spectrogram in Hz.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames, determines time scale in x-axis.
        segment_length (float): Desired length of each audio segment in seconds.
        allowed_length (float): Maximum allowed length of the audio file in seconds.
        target_sr (int): Target sample rate for librosa, set to highest expected sampling rate.
        to_db (bool): Recommended. Whether to convert the spectrogram to dB scale.
        mean_std (bool): Recommended. Whether to normalize by mean and standard deviation.

    Returns:
        tuple: A tuple containing:
            - list of numpy arrays, representing mel spectrograms of size (512, 1024) for each time slice.
            - int representing the sample rate of the audio file.
    """
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

    return slices, sr


def plot_spectrogram(S, sr, fmin=10000, fmax=80000, hop_length=512, hline = None):
    """
    This function plots a mel spectrogram using librosa's display module.

    Args:
        S (numpy.ndarray): Mel spectrogram to plot, shape (512, 1024).
        sr (int): Sample rate of the audio.
        fmin (int): Minimum frequency for the spectrogram in Hz.
        fmax (int): Maximum frequency for the spectrogram in Hz.
        hop_length (int): Number of samples between successive frames, determines time scale in x-axis.
        hline (tuple, optional): Tuple of two frequencies to draw horizontal lines on the plot. 
            This can be used to visualize frequency bands of interest, for example when debugging. 

    Returns:
        Figure showing the mel spectrogram.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(
        S,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Mel Spectrogram")
    fig.tight_layout()
    if hline:
        plt.axhline(y=hline[0])
        plt.axhline(y=hline[1])
    return fig


def preprocess_all_data(df: pd.DataFrame, species_selection:list[str]):
    """
    This function preprocesses all audio files in the given DataFrame,
    with a cutoff point for the number of segments per species to prevent class imbalance.
    It saves the preprocessed segments as CSV files and a coresponding labels file.
    Returns nothing.

    Args:
        df (pd.DataFrame): DataFrame containing audio file metadata with columns "id" and "species".
        species_selection (list[str]): List of species names to filter the metadata DataFrame on.
    """
    labels = []

    # Prevent class imbalance
    max = 280
    counts = [0, 0, 0, 0]

    for index, row in df.iterrows():
        file_id = row["id"]
        species_id = species_selection.index(row["species"])

        if counts[species_id] >= max:
            continue

        try:
            slices, sr = preprocess(raw_path(file_id))
            for idx, slice in enumerate(slices):
                if counts[species_id] >= max:
                    continue

                if slice.shape != (512, 1024):
                    print(f"wrong shape {slice.shape} for file: {file_id}")
                    continue

                out_path = data_path(file_id, idx)
                arr = np.asarray(slice)
                with open(out_path, "wb") as csv_file:
                    np.savetxt(csv_file, arr, delimiter=",")
                    # append successful labels
                    labels.append([f"{file_id}_{idx}", f"{species_id}"])
                counts[species_id] += 1

        except Exception as e:
            print(f"Could not process file {file_id}")
            print(e)

    labels = sorted(labels)

    all_labels = np.array(labels)[:, 1]
    unique, counts = np.unique(all_labels, return_counts=True)
    print("Label count:", unique, counts)

    with open(label_path(), "wb") as csv_file:
        np.savetxt(csv_file, labels, delimiter=",", fmt="%s")



