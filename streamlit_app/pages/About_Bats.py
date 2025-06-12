import streamlit as st
import os

st.set_page_config(page_title="About Bats")
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("About Bats")
st.write("""
### Why classify bat audio?
This is important for conservation, biodiversity monitoring, and ecological research.
""")

bat_info = [
    {
        "name": "Pipistrellus pipistrellus (Common Pipistrelle)",
        "img": "about_bats_images/Pipistrellus pipistrellus.jpg",
        "desc": "Small brown bat, widespread in Europe.",
        "calls": "Calls are short, typically between 45–55 kHz. In a spectrogram, these appear as short, downward going lines."
    },
    {
        "name": "Nyctalus noctula (Common Noctule)",
        "img": "about_bats_images/Nyctalus_noctula.jpg",
        "desc": "One of the largest European bats, often flies high and fast, often hunt in swarms.",
        "calls": "Calls are longer and around 20–25 kHz. In a spectrogram, these look like horizontal or slightly sloping lines."
    },
    {
        "name": "Plecotus auritus (Brown Long-eared Bat)",
        "img": "about_bats_images/Plecotus auritus.jpg",
        "desc": "Recognizable by its very large ears.",
        "calls": "Calls are very quiet, often between 30–50 kHz. In a spectrogram, these are faint, wide-bandwidth, short signals."
    },
    {
        "name": "Myotis albescens (Silver-tipped Myotis)",
        "img": "about_bats_images/Myotis albescens.jpg",
        "desc": "Found in the Americas, named for its silvery tips.",
        "calls": "Calls are often between 40–60 kHz, with a steep slope in the spectrogram."
    },
]

# loop over the bats information, print an image and the description.
for bat in bat_info:
    st.subheader(bat["name"])
    img_path = os.path.join(os.path.dirname(__file__), "about_bats_images", os.path.basename(bat["img"]))
    if os.path.exists(img_path):
        st.image(img_path, width=300)
    st.write(f"**Description:** {bat['desc']}")
    st.write(f"**Calls:** {bat['calls']}")
    st.markdown("---")

st.write("""
*Spectrograms are visual representations of sound, showing frequency (vertical axis) over time (horizontal axis), with intensity as color.*
""")
