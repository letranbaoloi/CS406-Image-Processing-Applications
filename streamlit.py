import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

from retrieve_fn import *
import time
from itertools import cycle


st.set_page_config(layout="wide")


st.title("CONTENT-BASED IMAGE RETRIEVAL")
col1, col2 = st.columns(2)
with col1:
    st.header("QUERY")

    st.subheader("Upload image")
    img_file = st.file_uploader(label="", type=["png", "jpg"])

    if img_file:
        img = Image.open(img_file)
        st.subheader("Result size")
        k = st.slider("", 1, 50, 5)
        st.subheader("Option")
        option1 = st.selectbox(
            "",
            ("Original", "Crop"),
        )
        st.write("Preview")
        if option1 == "Original":
            st.image(img)
        elif option1 == "Crop":
            # Get a cropped image from the frontend
            img = st_cropper(img, realtime_update=True, box_color="#FF0004")

            # Manipulate cropped image at will
            st.write("Preview")
            _ = img.thumbnail((150, 150))
            st.image(img)


with col2:
    st.header("RESULT")
    if img_file:
        start = time.time()
        indices, _ = retrieve(img, k)
        # indices = retrieve_augmented(img, 4)
        end = time.time()

        st.markdown(f"**Finish in {(end - start):.2f} seconds**")
        caption = []
        filteredImages = []
        for idx in indices:
            img, cap = get_image_from_index(idx)
            filteredImages.append(img)
            caption.append(cap)

        cols = cycle(
            st.columns(3)
        )  # st.columns here since it is out of beta at the time I'm writing this
        for idx, filteredImage in enumerate(filteredImages):
            next(cols).image(filteredImage, width=150, caption=caption[idx])
