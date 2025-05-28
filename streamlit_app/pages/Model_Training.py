import streamlit as st

st.set_page_config(
    page_title="Model Training"
)
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Model Training")
st.write("""
Train the CNN model on your audio dataset. View training loss and accuracy plots.
""")
