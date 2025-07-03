#!/usr/bin/env python3
"""
Create test training preview images for development
"""
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_test_preview_image(epoch_num, output_dir):
    """Create a test training preview image"""
    # Create a 512x512 test image
    img = Image.new('RGB', (512, 512), color='white')
    draw = ImageDraw.Draw(img)
    
    # Create some fake training visualization
    # Draw background gradient
    for y in range(512):
        color_value = int(255 * (y / 512))
        draw.line([(0, y), (512, y)], fill=(color_value, color_value, 255))
    
    # Draw some fake segmentation masks
    import random
    random.seed(epoch_num)
    
    for _ in range(10):
        x = random.randint(50, 450)
        y = random.randint(50, 450)
        radius = random.randint(10, 30)
        color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline='black', width=2)
    
    # Add epoch info
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    text = f"Epoch {epoch_num:03d} - Training Preview"
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        text_width, text_height = 200, 20
    
    x = (512 - text_width) // 2
    y = 20
    draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill='white', outline='black')
    draw.text((x, y), text, fill='black', font=font)
    
    # Save the image
    filename = f"predictions_epoch_{epoch_num:03d}.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    print(f"Created test preview: {filepath}")
    return filepath

def main():
    # Create test data directory structure
    base_dir = Path("data")
    preview_dirs = [
        base_dir / "models" / "artifacts",
        base_dir / "models" / "predictions", 
        base_dir / "temp",
        base_dir / "mlflow" / "test_run" / "artifacts" / "predictions"
    ]
    
    for dir_path in preview_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # Create test preview images for epochs 1-5
    for epoch in range(1, 6):
        for dir_path in preview_dirs:
            create_test_preview_image(epoch, str(dir_path))
    
    print("\n✅ Test training preview images created successfully!")
    print("\nYou can now test the training preview functionality in the web interface.")
    print("The images will be served at URLs like:")
    print("http://localhost:8000/ml/model/{model_id}/training-preview/predictions_epoch_001.png")

if __name__ == "__main__":
    main()
