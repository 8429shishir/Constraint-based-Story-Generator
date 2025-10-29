"""
embeddings.py
--------------
Generates vector embeddings for images and text using CLIP (OpenAI).
"""

import torch
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def load_clip_model(device="cpu"):
    """
    Load the CLIP model and processor.
    """
    print("⚡ Loading CLIP model (it may take a minute on first run)...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return processor, model


def generate_embeddings(images, captions, processor, model, device="cpu"):
    """
    Takes lists of images and captions, returns image and text embeddings.

    Args:
        images (List[PIL.Image.Image])
        captions (List[str])
    Returns:
        (torch.Tensor, torch.Tensor): image_embeds, text_embeds
    """
    print("🧮 Generating embeddings...")
    image_embeds, text_embeds = [], []

    for img, caption in tqdm(zip(images, captions), total=len(images), desc="🔢 Encoding pairs"):
        inputs = processor(text=[caption], images=img, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        img_emb = outputs.image_embeds.detach().cpu()
        txt_emb = outputs.text_embeds.detach().cpu()
        image_embeds.append(img_emb)
        text_embeds.append(txt_emb)

    image_embeds = torch.cat(image_embeds)
    text_embeds = torch.cat(text_embeds)

    print(f"✅ Generated {len(image_embeds)} embeddings.")
    return image_embeds, text_embeds


if __name__ == "__main__":
    # Quick self-test
    from preprocess import load_images_from_folder
    from captioning import load_blip_model, generate_captions

    folder = "data/images/"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 1 – load images
    images = load_images_from_folder(folder)

    # Step 2 – generate captions
    processor_blip, model_blip = load_blip_model(device)
    captions = generate_captions(images, processor_blip, model_blip, device)

    # Step 3 – generate embeddings
    processor_clip, model_clip = load_clip_model(device)
    img_embs, txt_embs = generate_embeddings(images, captions, processor_clip, model_clip, device)

    print("Image embedding shape:", img_embs.shape)
    print("Text embedding shape:", txt_embs.shape)