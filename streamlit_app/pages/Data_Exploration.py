import streamlit as st

st.set_page_config(
    page_title="Data Exploration"
)
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Data Exploration")
st.write("""
Explore the dataset: listen to audio samples, view spectrograms, etc.
""")
