import streamlit as st
import os

st.set_page_config(page_title="About Bats")
st.sidebar.markdown("### Navigation\nSelect a page above.")
st.title("About Bats")
st.write(
    """
### Why classify bat audio?
Acoustic monitoring is a key tool in ecological field research, 
which plays a vital role in conservation, biodiversity monitoring, and species protection. 
By recording and analyzing bat calls, researchers can identify different species and assess their presence in a given area.
\nBats often roost in gaps within buildings and roofs and use structures like tree lines or rows of buildings as flight paths. 
When these features are altered or removed, bats may be disturbed or lose essential habitats, 
potentially leading to population decline. 
On average, bats give birth to only one pup per year, 
which makes their populations particularly sensitive to environmental changes.
\nThis vulnerability is why Dutch and European law requires that bats be protected. 
As a result, ecological assessment is mandatory before any construction or renovation activities, 
helping to prevent the decline—or even extinction—of local bat species.
        
"""
)

bat_info = [
    {
        "name": "Pipistrellus pipistrellus (Common Pipistrelle)",
        "img": "about_bats_images/Pipistrellus pipistrellus.jpg",
        "desc": "The Small brown bat is widespread in Europe. It is the most common type of bat in the Netherlands, and also the smallest type that lives there..",
        "calls": "Calls are short, typically between 45–55 kHz. In a spectrogram, these appear as vertical lines, shaped like a right pointing hockeystick.",
    },
    {
        "name": "Nyctalus noctula (Common Noctule)",
        "img": "about_bats_images/Nyctalus_noctula.jpg",
        "desc": "The common noctule is one of the largest European bats, also found in the Netherlands. They often fly high and fast, and they often hunt in swarms.",
        "calls": "Calls are longer and around 20–25 kHz. In a spectrogram, these look like horizontal or slightly sloping lines.",
    },
    {
        "name": "Plecotus auritus (Brown Long-eared Bat)",
        "img": "about_bats_images/Plecotus auritus.jpg",
        "desc": "The brown long-eared bat is recognizable by its very large ears. Due to its large ears, it is able to hear insects fly, and therefore does not need its echolocation to locate them. "
        "Furthermore, its echolocation is so quiet that it is only detected by bat detectors when the bat is within a few meters distance, which make recordings of this species quite rare.",
        "calls": "Calls are very quiet, often between 30–50 kHz. In a spectrogram, these calls usually appear as two short vertical traces, one above the other, with a noticeable gap around 30 kHz separating them.",
    },
    {
        "name": "Myotis albescens (Silver-tipped Myotis)",
        "img": "about_bats_images/Myotis albescens.jpg",
        "desc": "The silver-tipped myotis is found in the Americas, and is named for its silvery tips.",
        "calls": "Calls are often between 40–60 kHz, with a steep slope in the spectrogram. Unlike the common pipistrelle, whose calls often have a characteristic “hockey stick” shape, Myotis calls appear almost purely vertical, indicating a rapid frequency drop with little horizontal spread.",
    },
]

# loop over the bats information, print an image and the description.
for bat in bat_info:
    st.subheader(bat["name"])
    img_path = os.path.join(
        os.path.dirname(__file__), "about_bats_images", os.path.basename(bat["img"])
    )
    if os.path.exists(img_path):
        st.image(img_path, width=300)
    st.write(f"**Description:** {bat['desc']}")
    st.write(f"**Calls:** {bat['calls']}")
    st.markdown("---")

st.write(
    """
*Spectrograms are visual representations of sound, showing frequency (vertical axis) over time (horizontal axis), with intensity as color. 
         When listening to bats using a bat-detector, the frequencies are lowered to human-hearing range using heterodyne, time-expansion, or frequency division.*
"""
)
