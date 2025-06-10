import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from project_name.data.preprocess import preprocess, plot_spectrogram

def load_model(model_path=None):
    pass

def preprocess_audio(filepath):
    """
    Preprocess an audio file for CNN prediction.
    """
    slices, sr = preprocess(filepath)
    return slices, sr

def plot_audio_spectrogram(slice, sr, fmin=10000, fmax=80000, hop_length=512):
    """
    Plot the spectrogram of the audio.
    """
    fig = plot_spectrogram(slice, sr, fmin, fmax, hop_length)
    return fig
