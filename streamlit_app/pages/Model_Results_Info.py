import os
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Model Results & Info")
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Model Results & Information")

st.write("""
## Model Overview
This project uses a convolutional neural network and a logistic regression as a base model to classify bat species from audio recordings.
""")

def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    return fig

class_names = [
    "Pipistrellus pipistrellus",
    "Nyctalus noctula",
    "Plecotus auritus",
    "Myotis albescens"
]

st.subheader("CNN Model Results")
cnn_metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trained_model/cnn_metrics.csv'))
if os.path.exists(cnn_metrics_path):
    cnn_metrics = pd.read_csv(cnn_metrics_path)
    # Show accuracy and ROC AUC as separate metrics
    if 'Accuracy' in cnn_metrics.columns:
        st.metric(label="Accuracy", value=f"{cnn_metrics['Accuracy'][0]:.4f}")
    if 'ROC AUC' in cnn_metrics.columns:
        st.metric(label="ROC AUC", value=f"{cnn_metrics['ROC AUC'][0]:.4f}")
    # Show confusion matrix as heatmap
    if 'Confusion Matrix' in cnn_metrics.columns:
        cm = np.array(cnn_metrics['Confusion Matrix'][0].replace('[','').replace(']','').split(','), dtype=int).reshape(len(class_names), len(class_names))
        st.write("**Confusion Matrix:**")
        st.pyplot(plot_confusion_matrix(cm, class_names, "CNN Confusion Matrix"))
else:
    st.info("CNN metrics file not found.")

st.subheader("Logistic Regression Results")
lr_metrics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trained_model/lr_metrics.csv'))
if os.path.exists(lr_metrics_path):
    lr_metrics = pd.read_csv(lr_metrics_path)
    # Show accuracy and ROC AUC as separate metrics
    if 'Accuracy' in lr_metrics.columns:
        st.metric(label="Accuracy", value=f"{lr_metrics['Accuracy'][0]:.4f}")
    if 'ROC AUC' in lr_metrics.columns:
        st.metric(label="ROC AUC", value=f"{lr_metrics['ROC AUC'][0]:.4f}")
    # Show confusion matrix as heatmap
    if 'Confusion Matrix' in lr_metrics.columns:
        cm = np.array(lr_metrics['Confusion Matrix'][0].replace('[','').replace(']','').split(','), dtype=int).reshape(len(class_names), len(class_names))
        st.write("**Confusion Matrix:**")
        st.pyplot(plot_confusion_matrix(cm, class_names, "Logistic Regression Confusion Matrix"))
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

### Evaluation Metrics Explained
- **Accuracy**: The proportion of correct predictions out of all predictions made. Higher is better.
- **Confusion Matrix**: A table showing the number of correct and incorrect predictions for each class. The diagonal are the correct predictions, ideally these values are the highest.
- **ROC AUC**: Area Under the Receiver Operating Characteristic Curve. We use the One-vs-Rest (OvR) approach, since we have more than 2 classes. The ROC AUC is computed for each class against all others, and the average is reported. 1.0 is perfect, 0.5 is random guessing.
- **Why use these metrics?**
    - **Accuracy** is intuitive but can be misleading if classes are imbalanced.
    - **Confusion Matrix** gives a detailed breakdown of the predictions, imbalances can also be spotted here.
    - **ROC AUC (OvR)** is robust to class imbalance and gives another metric to evaluate model performance.
""")
