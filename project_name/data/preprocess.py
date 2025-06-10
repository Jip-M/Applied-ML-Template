# DATA PREPROCCCESSING MODULE


import tempfile
from pydub import AudioSegment, effects
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project_name.data.download_data import raw_path, data_path, label_path
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
    filepath,
    fmin=17000,
    fmax=80000,
    n_fft=4096,
    hop_length=125,  # if you change this, change segment_length accordingly
    segment_length=0.5,  # in s, if you change this, change hop_length accordingly
    allowed_length=60,  # in s
    target_sr=256000,  # resample to highest expected sampling rate
    to_db=True,  # needs to be true, otherwise the spectrogram is ugly
    mean_std=True,  # if True, normalize mean, std
):
    # filecheck
    # if not filepath.endswith(".wav"):
    #     raise ValueError("Input file must be a .wav file.")
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


def preprocess_all_data(df: pd.DataFrame, species_selection):
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


# Important! Filtering does not work optimally yet for each bat type.
# hyperparameter -30db
def filter_nonempty_max(
    slices,
    bat_min_freq,
    bat_max_freq,
    threshold_db = -30,
    # parameters for mel_freqs must be equal to preprocess parameters
    n_mels = 512,
    fmin = 17000,
    fmax = 80000,
    save_dir = None,
)-> tuple[list, list]:


    # Convert expected bat frequency range to Mel bin indices
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    bat_bins = np.where((mel_freqs >= bat_min_freq) & (mel_freqs <= bat_max_freq))[0]

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
    slices,
    # parameters for mel_freqs must be equal to parameters used for preprocess function
    fmin = 17000,
    fmax = 80000,
    n_mels = 512,
    bat_min_freq = 41000,
    bat_max_freq = 48000,
    title = "Histogram of Mean Energy in Bat Frequency Band",
):
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
    slices,
    bat_min_freq,
    bat_max_freq,
    title="Histogram of Mean Energy in Bat Frequency Band",
    # parameters for mel_freqs must be equal to parameters used for preprocess function
    fmin = 17000,
    fmax = 80000,
    n_mels = 512,
):
    # Get mel frequency bins
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    bat_bins = np.where((mel_freqs >= bat_min_freq) & (mel_freqs <= bat_max_freq))[0]

    # compute max energies
    energies = [np.max(S[bat_bins, :]) for S in slices]
    # Compute mean energy in the bat frequency bins for each slice

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(energies, bins=50, color="purple", edgecolor="black")
    plt.title(f"{title}")
    plt.xlabel("Mean Energy (dB)")
    plt.ylabel("Number of Slices")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

