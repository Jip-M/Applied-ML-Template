import streamlit as st
import os
from utils import preprocess_audio, plot_audio_spectrogram

st.set_page_config(
    page_title="Data Exploration"
)
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Data Exploration")
st.write("""
Explore the dataset: listen to audio samples, view spectrograms, etc.
""")

sample_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/sample'))
uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

# give user option to select a sample.
sample_files = []
if os.path.exists(sample_folder):
    sample_files = [f for f in os.listdir(sample_folder) if f.endswith('.wav')]
    if sample_files:
        selected_sample = st.selectbox("Or select a sample from the dataset", sample_files)
    else:
        selected_sample = None
else:
    selected_sample = None

process = st.button("Show Spectrograms")
# if button is pressed, preprocess and show spectrograms.
if process:
    file_path = None
    if uploaded_file is not None:
        with open("temp_uploaded.wav", "wb") as f:
            f.write(uploaded_file.read())
        file_path = "temp_uploaded.wav"
    elif selected_sample:
        file_path = os.path.join(sample_folder, selected_sample)

    if file_path:
        slices, sr = preprocess_audio(file_path)
        st.success("Audio preprocessed!")
        audio_bytes = open(file_path, 'rb').read()
        st.audio(audio_bytes, format='audio/wav')
        for slice in slices:
            st.pyplot(plot_audio_spectrogram(slice, sr))
    else:
        st.warning("Please upload a file or select a sample before pressing the button.")
