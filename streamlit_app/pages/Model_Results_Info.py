import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch
import joblib
from project_name.models.CNN import AudioCNN

st.set_page_config(page_title="Model Results & Info")
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Model Results & Information")

st.write("""
## Model Overview
This project uses a convolutional neural network and a logistic regression as a base model to classify bat species from audio recordings.
""")

st.subheader("CNN Model Results")
cnn_metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trained_model/cnn_metrics.csv'))
if os.path.exists(cnn_metrics_path):
    cnn_metrics = pd.read_csv(cnn_metrics_path)
    st.write("**CNN Evaluation Metrics:**")
    st.dataframe(cnn_metrics)
else:
    st.info("CNN metrics file not found.")

st.subheader("Logistic Regression Results")
lr_metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trained_model/logreg_metrics.csv'))
if os.path.exists(lr_metrics_path):
    lr_metrics = pd.read_csv(lr_metrics_path)
    st.write("**Logistic Regression Evaluation Metrics:**")
    st.dataframe(lr_metrics)
else:
    st.info("Logistic Regression metrics file not found.")

st.write("""
### Results Summary
- **CNN Model:**
    - Achieves high accuracy on spectrogram images of bat calls.
    - Handles complex patterns in audio data.
- **Logistic Regression:**
    - Used as a baseline model.
    - Simpler, interpretable, but less accurate on complex audio data.

### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions.
- **Confusion Matrix**: Shows true vs. predicted classes.
- **Confidence**: Model's certainty in its prediction.
""")
