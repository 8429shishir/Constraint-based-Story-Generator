"""
storygen.py
------------
Generates a story from image captions using a local language model.
Optimized for mid-range systems (24GB RAM, 4GB GPU, i7 Processor)
"""

import os
import yaml
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# ==============================================================
# CONFIGURATION LOADER
# ==============================================================
def load_config(path="config.yaml"):
    """Load settings from config.yaml"""
    if not os.path.exists(path):
        raise FileNotFoundError("❌ config.yaml not found!")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ==============================================================
# PROMPT BUILDER
# ==============================================================
def build_prompt(captions, config):
    """
    Build a descriptive, instruction-based prompt for story generation.
    """
    caption_text = "\n".join([f"- {cap}" for cap in captions])
    tone = config.get("story_tone", "neutral")
    word_limit = config.get("max_words", 300)

    prompt = f"""
You are a skilled storyteller. Combine these image captions into one coherent story.

Image captions:
{caption_text}

Guidelines:
- The story must logically connect the scenes.
- Avoid bullet points or repetition.
- Maintain a {tone} tone.
- Write within {word_limit} words.
- Output in paragraph form only.

Story:
"""
    return prompt.strip()


# ==============================================================
# LOAD LOCAL MODEL
# ==============================================================
def load_local_model(model_name="microsoft/phi-2", device="cpu"):
    """
    Loads a local text generation model from Hugging Face.
    microsoft/phi-2 -> 2.7GB model, very good for coherent storytelling.
    """
    print(f"🚀 Loading local model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with GPU acceleration if available
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )

    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )
    return text_gen


# ==============================================================
# GENERATE STORY
# ==============================================================
def generate_story(captions, config, text_gen):
    """
    Generate story text from a list of image captions and configuration.
    """
    prompt = build_prompt(captions, config)
    print("🧠 Generating story locally... (may take a few seconds)")

    output = text_gen(
        prompt,
        max_length=config.get("max_words", 300),
        temperature=0.8,
        do_sample=True,
        top_p=0.9,
        truncation=True
    )[0]["generated_text"]

    # Clean up output
    if "Story:" in output:
        output = output.split("Story:")[-1].strip()

    return output


# ==============================================================
# MAIN PIPELINE (TEST)
# ==============================================================
if __name__ == "__main__":
    from preprocess import load_images_from_folder
    from captioning import load_blip_model, generate_captions

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")

    # Load configuration
    config = load_config()

    # Step 1: Load and caption images
    folder = "data/images/"
    images = load_images_from_folder(folder)
    processor_blip, model_blip = load_blip_model(device)
    captions = generate_captions(images, processor_blip, model_blip, device)

    print("\n🖼️ Image Captions:")
    for i, cap in enumerate(captions, 1):
        print(f"{i}. {cap}")

    # Step 2: Load local model & generate story
    text_gen = load_local_model(model_name="microsoft/phi-2", device=device)
    story = generate_story(captions, config, text_gen)

    # Step 3: Print result
    print("\n📝 Generated Story:\n")
    print(story)
