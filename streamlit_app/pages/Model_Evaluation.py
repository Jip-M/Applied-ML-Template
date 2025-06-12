import streamlit as st
import os
from utils import preprocess_audio

st.set_page_config(
    page_title="Model Evaluation"
)
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Main Model Evaluation")
st.write("""
Upload or choose an audio file to test the trained model and see predictions.
""")

st.write(
    """
## Convolutional Neural Network Model Evaluation

This page allows you to upload an audio file or select a sample from the dataset to evaluate the trained CNN. The model will predict the class of the audio based on the features extracted from it.

\nThe provided dataset contains a random selection of audio recordings of various bat species. 
         These have either been recorded in the Netherlands by colleagues at the Dutch ecological research company Gaia ('field') 
         or taken from the open-access database Xeno-Canto ('XC'). 
         The model was trained on the quality-A data of Xeno-Canto, whereas the XC samples are of quality B.

### How to Use
1. Upload a `.wav` file or select a sample from the dataset.
2. Click on "Preprocess and predict".
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

process = st.button("Preprocess and predict")
# if button is pressed, preprocess and predict.
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
        import torch
        import numpy as np
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
        from bat_classifier.models.CNN import AudioCNN
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trained_model/CNN.pt'))
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
        if os.path.exists(model_path):
            num_classes = len(class_names)
            # initialize CNN as the saved one.
            model = AudioCNN(num_classes=num_classes, learning_rate=0.001, number_of_epochs=1, patience = 6)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            all_probs = []
            for x in slices:
                if x.shape != (512, 1024):
                    continue
                x = (x - np.mean(x)) / (np.std(x) + 1e-8)
                x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0).float()
                # get logits to be able to predict and show confidence.
                with torch.no_grad():
                    logits = model(x_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    all_probs.append(probs)
            if all_probs:
                all_probs = np.array(all_probs)
                avg_probs = np.mean(all_probs, axis=0)
                pred_idx = int(np.argmax(avg_probs))
                confidence = float(avg_probs[pred_idx])
                st.success(f"Predicted class: {class_names[pred_idx]} (confidence: {confidence:.2f})")
                # Show bat image
                img_path = os.path.join(os.path.dirname(__file__), "about_bats_images", os.path.basename(bat_img_map[pred_idx]))
                if os.path.exists(img_path):
                    st.image(img_path, width=300, caption=class_names[pred_idx])
            else:
                st.warning("No valid slices with shape (512, 1024) found for prediction.")
        else:
            st.warning("Trained CNN model not found.")
    else:
        st.warning("Please upload a file or select a sample")
