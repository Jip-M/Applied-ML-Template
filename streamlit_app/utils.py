import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bat_classifier.data.preprocess import preprocess, plot_spectrogram
import numpy as np
import matplotlib.pyplot as plt


def preprocess_audio(filepath: str) -> tuple[list[np.ndarray], int]:
    """
    Preprocess an audio file for CNN prediction.
    Args:
        filepath (str): Path to the audio file.
    Returns:
        tuple: A tuple with the preprocessed audio slices and the sample rate.
    """
    slices, sr = preprocess(filepath)
    return slices, sr


def plot_audio_spectrogram(
    slice: np.ndarray,
    sr: int,
    fmin: int = 10000,
    fmax: int = 80000,
    hop_length: int = 512,
) -> plt.Figure:
    """
    Plot the spectrogram of the audio.
    Args:
        slice (numpy.ndarray): Audio slice to plot.
        sr (int): Sample rate of the audio.
        fmin (int): Minimum frequency for the spectrogram.
        fmax (int): Maximum frequency for the spectrogram.
        hop_length (int): Hop length for the spectrogram.
    Returns:
        matplotlib.figure.Figure: figure showing the spectrogram.
    """
    fig = plot_spectrogram(slice, sr, fmin, fmax, hop_length)
    return fig
