import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from colorizator import MangaColorizator 

# --- CONFIGURATION ---
INPUT_ROOT = "input_manga"         # Your main folder
OUTPUT_ROOT = "output_manga"       # Where results go
MODEL_PATH = "networks/generator.pth" # The file you moved
IMAGE_SIZE = 576             # Standard size (keep multiples of 32)
DENOISE_LEVEL = 25           # 0 to disable, 25 is standard
VIBRANCY_BOOST = 1         # 1.0 = Original, 1.4 = 40% more colorful

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps' # Mac M-Series GPU
    return 'cpu'

def boost_color(image_array, factor):
    """Converts numpy array to PIL, boosts color, returns PIL image"""
    # Convert from 0-1 Float (Matplotlib) to 0-255 Int (PIL)
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    
    img = Image.fromarray(image_array)
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def main():

    start_time = time.time()

    # 1. Setup
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Error: Folder '{INPUT_ROOT}' not found.")
        print("   Please create a folder named 'Manga' and put chapter folders inside it.")
        return

    device = get_device()
    print(f"🚀 Loading MangaColorizator on {device}...")
    
    # Initialize the Repo's internal colorizer
    try:
        colorizer = MangaColorizator(device, generator_path=MODEL_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Model not found at '{MODEL_PATH}'.")
        print("   Make sure you moved 'generator.pth' into the 'networks' folder!")
        return



    # 2. Find Chapters
    chapters = sorted([d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))])
    print(f"📂 Found {len(chapters)} chapters: {chapters}")

    # 3. Process Loop
    for chapter in chapters:
        input_chap_path = os.path.join(INPUT_ROOT, chapter)
        output_chap_path = os.path.join(OUTPUT_ROOT, chapter)
        
        # Create output folder
        os.makedirs(output_chap_path, exist_ok=True)
        
        # Get Images (Ignore .DS_Store)
        valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
        images = [f for f in os.listdir(input_chap_path) if f.lower().endswith(valid_exts) and not f.startswith('.')]
        images.sort()

        if not images:
            continue
            
        print(f"🎨 Processing Chapter {chapter} ({len(images)} images)...")

        for idx, img_name in enumerate(images):
            in_path = os.path.join(input_chap_path, img_name)
            out_path = os.path.join(output_chap_path, img_name)

            try:
                # A. Load Image (using Matplotlib to match inference.py)
                image = plt.imread(in_path)
                
                # B. Run Colorizer (Repo Logic)
                # Note: We pass args manually since we aren't using argparse
                colorizer.set_image(image, size=IMAGE_SIZE, denoise_sigma=DENOISE_LEVEL)
                result_array = colorizer.colorize()
                
                # C. Boost Color & Save (Custom Logic)
                final_img = boost_color(result_array, VIBRANCY_BOOST)
                final_img.save(out_path)

                # --- MEMORY HYGIENE BLOCK ---
                # This ensures page 1's memory is gone before page 2 starts
                del result_array 
                if device == 'mps':
                    torch.mps.empty_cache() # Clears Mac M4 memory
                elif device == 'cuda':
                    torch.cuda.empty_cache() # Clears NVIDIA GPU memory
                
                print(f"   [{idx+1}/{len(images)}] Saved: {img_name}")
                
            except Exception as e:
                print(f"   ⚠️ Failed {img_name}: {e}")

    end_time = time.time()
    total_seconds = end_time - start_time
    
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    print(f"✅ All Done! Check the '{OUTPUT_ROOT}' folder.")
    print(f"⏱️ Total Time Taken: {minutes} minutes and {seconds} seconds")

if __name__ == "__main__":
    main()