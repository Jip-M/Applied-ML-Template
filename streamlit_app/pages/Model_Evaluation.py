import streamlit as st

st.set_page_config(
    page_title="Model Evaluation"
)
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Model Evaluation")
st.write("""
Upload or choose an audio file to test the trained model and see predictions.
""")
