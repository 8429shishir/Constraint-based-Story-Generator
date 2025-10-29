import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# initialize BLIP model
def load_blip_model(device="cpu"):
    print("🚀 Loading BLIP model (it may take a minute the first time)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model


def generate_captions(images, processor, model, device="cpu"):
    """
    Takes a list of PIL images and returns generated captions.
    """
    captions = []
    for img in tqdm(images, desc="🧠 Generating captions"):
        inputs = processor(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        text = processor.decode(out[0], skip_special_tokens=True)
        captions.append(text)
    return captions


if __name__ == "__main__":
    # Quick test
    from preprocess import load_images_from_folder
    import os

    folder = "data/images/"
    if not os.path.exists(folder):
        raise FileNotFoundError("Please create 'data/images/' and add a few images first!")

    images = load_images_from_folder(folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor, model = load_blip_model(device)
    captions = generate_captions(images, processor, model, device)

    print("\n📝 Generated Captions:")
    for i, cap in enumerate(captions, 1):
        print(f"{i}. {cap}")