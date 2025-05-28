import streamlit as st

st.set_page_config(
    page_title="Home"
)
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("Bat Audio Classification Project")
st.write("""
Welcome to the Bat Audio Classification Streamlit App!

Navigate using the sidebar to explore data, train models, and evaluate predictions.
""")
