import streamlit as st

st.set_page_config(
    page_title="Home"
)
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Bat Audio Classification Project")
st.write("""
Welcome to the Bat Audio Classification Streamlit App!

This app allows you to explore, preprocess, and classify bat audio recordings using a convolutional neural network, and a base model (logistic regression) to compare to.

**How to use this app:**
- **About:** General information about the project and its goals.
- **About Bats:** Learn about the bat species in this project and how their calls appear in spectrograms.
- **Data Exploration:** Visualize and listen to bat audio files and inspect their spectrograms.
- **Logistic Regression:** Evaluate the logistic regression model on your audio files for comparison with the CNN.
- **Model Evaluation:** Upload or select an audio file to get predictions from the trained CNN model.
- **Model Results & Info:** View model performance, evaluation metrics, and a summary of results.

Navigate using the sidebar to explore data and generate predictions. Each page is designed to help you understand and interact with the bat audio classification pipeline.
""")
