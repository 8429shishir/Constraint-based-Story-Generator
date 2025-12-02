import os
from PIL import Image
from tqdm import tqdm

def load_images_from_folder(folder_path, resize=(224, 224)):
    """
    Loads and resizes all images from a given folder.

    Args:
        folder_path (str): Path to the image directory.
        resize (tuple): Resize dimensions (width, height).

    Returns:
        List[Image.Image]: List of PIL image objects.
    """
    images = []
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f" Folder not found: {folder_path}")

    for filename in tqdm(os.listdir(folder_path), desc="Loading images"):
        if filename.lower().endswith(supported_formats):
            path = os.path.join(folder_path, filename)
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(resize)
                images.append(img)
            except Exception as e:
                print(f"⚠️ Could not load {filename}: {e}")
    print(f" Loaded {len(images)} images from {folder_path}")
    return images


if __name__ == "__main__":
    # For quick testing
    folder = "data/images/"
    imgs = load_images_from_folder(folder)
    print(f"Total images: {len(imgs)}")
