import streamlit as st

st.set_page_config(
    page_title="About"
)
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("About")
st.write("""
This project was developed for the Applied Machine Learning course (WBAI065-05) at Rijksuniversiteit Groningen.

**Team:**
- Tijn Krikke
- Sara Mijnheer
- Cristian Mihaiescu
- Helena Jager

**Project Supervisor:**
- Xenia Demetriou

**References:**
- [Project GitHub](https://github.com/Jip-M/Applied-ML-Template)
- [Data Source](https://xeno-canto.org/)
""")
