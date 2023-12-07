import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image, ImageFilter

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

    st.subheader("Select feature space")
    feature_space_opt = st.selectbox(
        "",
        (
            "ResNet50",
            "ResNet50 (Augmented)",
            "HOG",
        ),
    )

    if img_file:
        img = Image.open(img_file)
        st.subheader("Result size")
        k = st.slider("", 1, 100, 10)
        st.subheader("Option")
        img_processing_opt = st.selectbox(
            "",
            (
                "Original",
                "Crop",
                "Blur",
                "Sharpen",
                "Smooth",
                "Edge",
                "Horizontal Flip",
                "Vertical Flip",
            ),
        )
        st.write("Preview")

        match img_processing_opt:
            case "Original":
                pass
            case "Blur":
                img = img.filter(ImageFilter.BLUR)
            case "Sharpen":
                img = img.filter(ImageFilter.SHARPEN)
            case "Smooth":
                img = img.filter(ImageFilter.SMOOTH)
            case "Edge":
                img = img.filter(ImageFilter.FIND_EDGES)
            case "Horizontal Flip":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            case "Vertical Flip":
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            case "Crop":
                # Get a cropped image from the frontend
                img = st_cropper(img, realtime_update=True, box_color="#FF0004")
                st.write("Preview (Resized)")
                _ = img.thumbnail((250, 250))
                st.image(img)
            case _:
                pass
        if img_processing_opt != "Crop":
            st.image(img)


with col2:
    st.header("RESULT")
    if img_file:
        start = time.time()
        if feature_space_opt == "ResNet50":
            indices, scores = retrieve(img, k)
        elif feature_space_opt == "ResNet50 (Augmented)":
            indices, scores = retrieve_augmented(img, k)
        elif feature_space_opt == "HOG":
            indices, scores = retrieve(img, k, index=hog_l2_index)
        end = time.time()

        st.markdown(f"**Finish in {(end - start):.2f} seconds**")
        caption = []
        filteredImages = []
        indices = indices[:k]
        scores = scores[:k]
        for idx, img_idx in enumerate(indices):
            img, name = get_image_from_index(img_idx)
            filteredImages.append(img)
            caption.append(f"{name.split('_')[1]}({scores[idx]:.2f})")

        cols = cycle(
            st.columns(3)
        )  # st.columns here since it is out of beta at the time I'm writing this
        for idx, filteredImage in enumerate(filteredImages):
            next(cols).image(filteredImage, width=150, caption=caption[idx])
