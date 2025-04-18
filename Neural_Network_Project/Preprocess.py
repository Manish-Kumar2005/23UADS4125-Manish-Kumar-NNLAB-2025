import os
import cv2
import numpy as np
from PIL import Image, ImageOps

input_root = 'Neural_Network_Project/Dataset_Project1'  # Input folder (use / or raw string)
output_root = 'processed_dataset'  # Output folder
os.makedirs(output_root, exist_ok=True)

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')

        # Crop to square using updated resampling method
        img = ImageOps.fit(img, (500, 500), method=Image.Resampling.LANCZOS)

        # Resize to 256x256
        img = img.resize((256, 256), resample=Image.Resampling.LANCZOS)

        # Convert to grayscale
        img = img.convert('L')

        # Thresholding
        img_np = np.array(img)
        _, thresh = cv2.threshold(img_np, 200, 255, cv2.THRESH_BINARY)

        return Image.fromarray(thresh)
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

for species in os.listdir(input_root):
    species_path = os.path.join(input_root, species)
    if not os.path.isdir(species_path):
        continue

    output_species_path = os.path.join(output_root, species)
    os.makedirs(output_species_path, exist_ok=True)

    for filename in os.listdir(species_path):
        img_path = os.path.join(species_path, filename)
        processed_img = preprocess_image(img_path)
        
        if processed_img is not None:
            output_path = os.path.join(output_species_path, filename)
            processed_img.save(output_path)
