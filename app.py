import streamlit as st
import torch
from PIL import Image
import os
import tempfile

# Local imports
from src.preprocess import load_images_from_folder
from src.captioning import load_blip_model, generate_captions
from src.storygen import load_local_model, generate_story, load_config
from src.embeddings import load_clip_model, generate_embeddings
from src.theme_eval import cluster_themes, compute_story_metrics
from sentence_transformers import SentenceTransformer



# PAGE CONFIG
st.set_page_config(
    page_title="Constraint-Based Story Generator",
    page_icon="📖",
    layout="wide"
)

st.title("📸 Constraint-Based Story Generator")
st.write("Upload one or more images and let the AI generate a story for you — completely offline!")


# LOAD MODELS
@st.cache_resource
def load_all_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.success(f"Using device: {device}")
    config = load_config()

    processor_blip, model_blip = load_blip_model(device)
    processor_clip, model_clip = load_clip_model(device)
    text_gen = load_local_model("Qwen/Qwen2.5-1.5B-Instruct", device)

    return device, config, processor_blip, model_blip, processor_clip, model_clip, text_gen

device, config, processor_blip, model_blip, processor_clip, model_clip, text_gen = load_all_models()


# SIDEBAR SETTINGS
st.sidebar.header("🧠 Story Constraints")
tone = st.sidebar.selectbox("Tone", ["inspirational", "mysterious", "funny",
                                     "dramatic", "romantic", "sad", "neutral"])
max_words = st.sidebar.slider("Max Words", 100, 600, 300)
config["story_tone"] = tone
config["max_words"] = max_words


# IMAGE UPLOAD
uploaded_files = st.file_uploader("Upload your images:", accept_multiple_files=True,
                                  type=["jpg", "jpeg", "png", "webp"])

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

    
    # 1) CAPTION GENERATION
    if st.button("🖼️ Generate Captions"):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        st.session_state["captions"] = generate_captions(images, processor_blip, model_blip, device)
        st.subheader("🧾 Generated Captions")
        for i, cap in enumerate(st.session_state["captions"], 1):
            st.write(f"**{i}.** {cap}")

    
    # 2) START STORY PIPELINE (Generate Embeddings + Themes)
    if "captions" in st.session_state and st.button("📖 Start Story Generation"):
        images = [Image.open(p).convert("RGB") for p in image_paths]
        img_emb, txt_emb = generate_embeddings(images, st.session_state["captions"],
                                               processor_clip, model_clip, device)

        st.session_state["image_embs"] = img_emb
        st.session_state["text_embs"] = txt_emb

        labels, cluster_info = cluster_themes(img_emb, txt_emb, st.session_state["captions"], n_clusters=3)
        st.session_state["clusters"] = cluster_info

    
    # 3) THEME SELECTION (only shown when clusters exist)
    if "clusters" in st.session_state:
        cluster_info = st.session_state["clusters"]
        st.subheader("🎯 Detected Themes (clusters)")

        theme_list = ["None"] + [f"Theme {i+1} : {cluster_info[i]['top_captions']}" for i in range(len(cluster_info))]
        chosen = st.radio("Choose a theme:", theme_list, key="theme_choice")

        st.session_state["theme_hint"] = None

        if chosen != "None":
            idx = int(chosen.split()[1]) - 1
            cluster = cluster_info[idx]

            top_caps = cluster["top_captions"]

            if top_caps:
                theme_hint = " | ".join(top_caps)
            else:
                theme_hint = f"Theme {idx+1}: general scene cluster"

            st.session_state["theme_hint"] = theme_hint


        
        # 4) FINAL STORY GENERATION BUTTON
        if st.button("✨ Generate Final Story"):

            story = generate_story(
                st.session_state["captions"],
                st.session_state["image_embs"],
                st.session_state["text_embs"],
                config,
                text_gen,
                theme_hint=st.session_state["theme_hint"]
            )

            st.subheader("📝 Generated Story")
            st.write(story)

            with st.expander("📊 Story Evaluation"):
                sent_model = SentenceTransformer("all-MiniLM-L6-v2")
                metrics = compute_story_metrics(
                    story,
                    st.session_state["captions"],
                    st.session_state["text_embs"],
                    sentence_model=sent_model
                )
                st.write(metrics)
