"""
app.py
------
Streamlit web interface for the Constraint-Based Story Generator
Uses BLIP for captions and Phi-2 for local story generation.
"""

import streamlit as st
import torch
from PIL import Image
import os
import tempfile

# Local imports
from src.preprocess import load_images_from_folder
from src.captioning import load_blip_model, generate_captions
from src.storygen import load_local_model, generate_story, load_config


# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Constraint-Based Story Generator",
    page_icon="📖",
    layout="wide"
)

st.title("📸 Constraint-Based Story Generator")
st.write("Upload one or more images and let the AI generate a story for you — completely offline!")

# ---------------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------------
@st.cache_resource
def load_all_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.success(f"Using device: {device}")
    config = load_config()

    # Load BLIP (image captioning)
    processor_blip, model_blip = load_blip_model(device)

    # Load Phi-2 (local text generation)
    text_gen = load_local_model(model_name="microsoft/phi-2", device=device)

    return device, config, processor_blip, model_blip, text_gen


device, config, processor_blip, model_blip, text_gen = load_all_models()

# ---------------------------------------------------------------
# SIDEBAR SETTINGS
# ---------------------------------------------------------------
st.sidebar.header("🧠 Story Constraints")
tone = st.sidebar.selectbox("Tone", ["inspirational", "mysterious", "funny", "dramatic", "romantic", "sad", "neutral"])
max_words = st.sidebar.slider("Max Words", 100, 600, 300)
config["story_tone"] = tone
config["max_words"] = max_words


# ---------------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------------
uploaded_files = st.file_uploader("Upload your images (JPG, PNG, WEBP):", accept_multiple_files=True, type=["jpg", "jpeg", "png", "webp"])

if uploaded_files:
    st.subheader("📷 Uploaded Images")
    cols = st.columns(len(uploaded_files))
    temp_dir = tempfile.mkdtemp()
    image_paths = []

    for i, uploaded_file in enumerate(uploaded_files):
        image_path = os.path.join(temp_dir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_paths.append(image_path)
        with cols[i]:
            st.image(Image.open(uploaded_file), use_container_width=True)

    # ---------------------------------------------------------------
    # CAPTION GENERATION
    # ---------------------------------------------------------------
    if st.button("🖼️ Generate Captions"):
        st.subheader("🧾 Generated Captions")
        images = [Image.open(p).convert("RGB") for p in image_paths]
        captions = generate_captions(images, processor_blip, model_blip, device)
        for i, cap in enumerate(captions, 1):
            st.write(f"**{i}.** {cap}")

        st.session_state["captions"] = captions

    # ---------------------------------------------------------------
    # STORY GENERATION
    # ---------------------------------------------------------------
    if "captions" in st.session_state and st.button("📖 Generate Story"):
        st.subheader("📝 Generated Story")
        story = generate_story(st.session_state["captions"], config, text_gen)
        st.write(story)
