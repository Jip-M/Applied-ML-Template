# DATA PREPROCCCESSING MODULE

# Download data
# filepath = "Applied-ML-Template\data\sample\XC912090 - Gewone dwergvleermuis - Pipistrellus pipistrellus.wav"


import tempfile
from pydub import AudioSegment, effects
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from project_name.data.download_data import raw_path, data_path, label_path

# Notes: got rid of minmax possibility, because it doesnt change anything in the end

# required hoplength for spectrogram:
# t     sr      n_fft    hop_length
# 0.1s	1024	256000	25
# 0.5s	1024	256000	125
# 1.0s	1024	256000	250


# Preprocess audio file
# This function preprocesses an audio file, normalizes it, computes the mel spectrogram, and returns segments of the spectrogram.
def preprocess(
    filepath,
    fmin=17000,
    fmax=80000,
    n_fft=4096,
    hop_length=250,  # if you change this, change segment_length accordingly
    segment_length=1,  # in s, if you change this, change hop_length accordingly
    allowed_length=30,  # in s
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

    # if sr < 2 * fmax:
    #     raise ValueError(
    #         f"Sampling rate ({sr}) is too low for the specified fmax ({fmax}). "
    #         f"It appears your audio file may not contain the ultrasonic frequency range. "
    #         f"Please upload a recording with a higher sampling rate."
    #     )

    # Calculate mel bands,, (fmax - fmin) // fbin #=
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
    # S = np.abs(librosa.stft(y, n_fft=4096, hop_length=25))

    # print(f"Mel spectrogram shape: {S.shape}, sr: {sr}")

    # doesnt change anything, so commented out
    # Normalize the spectrogram
    # if minmax:
    #     S = (S - np.min(S)) / (np.max(S) - np.min(S))
    #     print(f"Spectrogram normalized to [0,1]: {S}")

    if mean_std:
        S = (S - np.mean(S)) / np.std(S)

    # Convert to dB scale if needed
    if to_db:
        S = librosa.power_to_db(S, ref=np.max)

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
        # if slice.shape[1] < frames_per_segment:
        #     slice = S[:, -frames_per_segment:]
        slices.append(slice)
    # print(f"slice size: {slices[0].shape}")

    # Generate time and mel frequency axes
    # times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    # mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)

    # return segments
    # sr is returned to draw the spectrogram plot, this is not required for the model
    return slices, sr


# Plot the spectrogram
def plot_spectrogram(S, sr, fmin=10000, fmax=80000, hop_length=512):
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
    return fig

def preprocess_all_data(df: pd.DataFrame, species_selection: list[str]):
    labels = []

    # Prevent class imbalance
    max = 100
    counts = [0,0,0,0]

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
        np.savetxt(csv_file, labels, delimiter=",", fmt='%s')

# Sample testing
# slices, sr = preprocess(filepath)
# k = 0
# for slice in slices:
#     if k == 5:  # bat is only in the middle of the audio file
#         plot_spectrogram(slice, sr)
#         break
#     k += 1
