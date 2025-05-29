import streamlit as st
import os
from utils import preprocess_audio

st.set_page_config(
    page_title="Model Evaluation"
)
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Model Evaluation")
st.write("""
Upload or choose an audio file to test the trained model and see predictions.
""")

sample_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/sample'))

uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

sample_files = []
if os.path.exists(sample_folder):
    sample_files = [f for f in os.listdir(sample_folder) if f.endswith('.wav')]
    if sample_files:
        selected_sample = st.selectbox("Or select a sample from the dataset", sample_files)
    else:
        selected_sample = None
else:
    selected_sample = None

process = st.button("Preprocess and predict")

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
        st.success("Prediction, yet to implement")
    else:
        st.warning("Please upload a file or select a sample")
