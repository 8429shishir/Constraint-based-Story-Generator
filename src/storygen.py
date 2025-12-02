"""
storygen.py
------------
Generates story using captions + embeddings + Qwen2.5 model.
"""

import os
import yaml
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


# CONFIG LOADER
def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise FileNotFoundError(" config.yaml not found!")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# PROMPT BUILDER ( UPDATED TO INCLUDE EMBEDDINGS)
def build_prompt(captions, image_embeds, text_embeds, config):
    """
    Build a prompt using:
    - BLIP captions
    - CLIP embeddings (optional text description)
    """

    caption_text = "\n".join([f"- {cap}" for cap in captions])
    tone = config.get("story_tone", "neutral")
    word_limit = config.get("max_words", 300)

    # convert embeddings to readable form (short numbers)
    embed_info = f"""
        Embedding Summary:
        - Image embedding shape: {tuple(image_embeds.shape)}
        - Text embedding shape: {tuple(text_embeds.shape)}
        (Use embeddings to ensure the story matches visual + semantic meaning)
        """

    prompt = f"""
        You are a skilled storyteller. Use both the image captions and the CLIP embeddings
        to generate a coherent story that reflects both visual and semantic meaning.

        Image Captions:
        {caption_text}

        {embed_info}

        Guidelines:
        - The story must logically connect the scenes.
        - Maintain a {tone} tone.
        - Write within {word_limit} words.
        - Output in paragraph form only.

        Story:
        """
    return prompt.strip()


# LOAD QWEN MODEL
def load_local_model(model_name="Qwen/Qwen2.5-1.5B-Instruct", device="cpu"):
    print(f" Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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


# GENERATE STORY 
def generate_story(captions, image_embeds, text_embeds, config, text_gen, theme_hint=None):
    prompt = build_prompt(captions, image_embeds, text_embeds, config)

    # if theme_hint provided, prepend to prompt to bias generation
    if theme_hint:
        prompt = f"Theme hint (focus on this): {theme_hint}\n\n" + prompt

    print(" Generating Qwen story...")
    output = text_gen(
        prompt,
        max_length=config.get("max_words", 300),
        temperature=0.8,
        do_sample=True,
        top_p=0.9
    )[0]["generated_text"]

    if "Story:" in output:
        output = output.split("Story:")[-1].strip()
    return output
