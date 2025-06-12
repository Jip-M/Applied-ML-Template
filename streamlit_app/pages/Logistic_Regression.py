import streamlit as st
import os
import joblib
import numpy as np
from utils import preprocess_audio
from bat_classifier.models.base_model import BaseModel

st.set_page_config(page_title="Logistic Regression Evaluation")
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Logistic Regression Model Evaluation")

st.write(
    """
## Logistic Regression Model Evaluation

This page allows you to upload an audio file or select a sample from the dataset to evaluate the trained logistic regression model. The model will predict the class of the audio based on the features extracted from it.

\nThe provided dataset contains a random selection of audio recordings of various bat species. 
         These have either been recorded in the Netherlands by colleagues at the Dutch ecological research company Gaia ('field') 
         or taken from the open-access database Xeno-Canto ('XC'). 
         The model was trained on the quality-A data of Xeno-Canto, whereas the XC samples are of quality B.

### How to Use
1. Upload a `.wav` file or select a sample from the dataset.
2. Click on "Preprocess and predict (Logistic Regression)".
3. View the predicted class for the audio.
         

"""
)

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

process = st.button("Preprocess and predict (Logistic Regression)")

# if button is pressed, preprocess and predict.
if process:
    file_path = None
    if uploaded_file is not None:
        with open("temp_uploaded_lr.wav", "wb") as f:
            f.write(uploaded_file.read())
        file_path = "temp_uploaded_lr.wav"
    elif selected_sample:
        file_path = os.path.join(sample_folder, selected_sample)

    if file_path:
        slices, sr = preprocess_audio(file_path)
        st.success("Audio preprocessed!")
        model_path_coef = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trained_model/coef.npy'))
        model_path_intercept = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trained_model/intercept.npy'))
        logreg_model = BaseModel()
        # names and respective images.
        class_names = [
            "Pipistrellus pipistrellus - Common Pipistrelle",
            "Nyctalus noctula - Common Noctule",
            "Plecotus auritus - Brown Long-eared Bat",
            "Myotis albescens - Silver-tipped Myotis"
        ]
        bat_img_map = {
            0: "about_bats_images/Pipistrellus pipistrellus.jpg",
            1: "about_bats_images/Nyctalus_noctula.jpg",
            2: "about_bats_images/Plecotus auritus.jpg",
            3: "about_bats_images/Myotis albescens.jpg"
        }
        if os.path.exists(model_path_coef):
            import numpy as np
            # since we do not train the model, we load the model, but we do not have the classes, so we initialize
            coef = np.load(model_path_coef)
            logreg_model.model.coef_ = coef
            intercept = np.load(model_path_intercept)
            logreg_model.model.intercept_ = intercept

            # Set the classes for the logistic regression model
            logreg_model.model.classes_ = np.array([0, 1, 2, 3])  # Assuming classes are 0, 1, 2, 3
            
            # check shape of slices and flatten them
            features = [s.flatten() for s in slices if s.shape == (512, 1024)]
            if features:
                features = np.mean(features, axis=0).reshape(1, -1)
                pred = int(logreg_model.model.predict(features))
                st.success(f"Predicted class: {class_names[pred]}")
                # Show bat image
                img_path = os.path.join(os.path.dirname(__file__), "about_bats_images", os.path.basename(bat_img_map[pred]))
                if os.path.exists(img_path):
                    st.image(img_path, width=300, caption=class_names[pred])
            else:
                st.warning("No valid slices for prediction.")
        else:
            st.warning("Trained logistic regression model not found.")
    else:
        st.warning("Please upload a file or select a sample")

