#!/usr/bin/env python3
"""
Test Tesseract with French language and different settings
"""

import pytesseract
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def create_french_test_image():
    """Create a test image with French text"""
    # French text with accents similar to your document
    french_text = """Québec MÉDECINE
Laval Hôpital
175, bld. René Lévesque
Laval H7M 3L5
Date de naissance: 14/05/2024
Adresse: 123 rue des Érables"""
    
    # Create image with better settings for OCR
    img = Image.new('RGB', (600, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a better font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # Draw text line by line
    lines = french_text.strip().split('\n')
    y_pos = 20
    for line in lines:
        draw.text((20, y_pos), line, fill='black', font=font)
        y_pos += 25
    
    return img

def test_tesseract_configs():
    """Test different Tesseract configurations"""
    print("Testing Tesseract with French text...")
    
    # Create test image
    img = create_french_test_image()
    img.save("test_french_document.png")
    print("Created test image: test_french_document.png")
    
    # Convert PIL to OpenCV format for preprocessing
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Test different configurations
    configs = [
        {
            'name': 'French only',
            'lang': 'fra',
            'config': '--oem 3 --psm 6'
        },
        {
            'name': 'English + French',
            'lang': 'eng+fra',
            'config': '--oem 3 --psm 6'
        },
        {
            'name': 'French with better PSM',
            'lang': 'fra',
            'config': '--oem 3 --psm 4'
        },
        {
            'name': 'French with auto orientation',
            'lang': 'fra',
            'config': '--oem 3 --psm 1'
        }
    ]
    
    print("\n" + "="*60)
    for config in configs:
        print(f"\n--- {config['name']} ---")
        print(f"Language: {config['lang']}")
        print(f"Config: {config['config']}")
        
        try:
            # Test with original image
            result = pytesseract.image_to_string(
                img, 
                lang=config['lang'], 
                config=config['config']
            )
            print("Original image result:")
            print(repr(result))
            
            # Test with preprocessed image
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply different preprocessing
            # 1. Denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # 2. Threshold
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL
            preprocessed_img = Image.fromarray(thresh)
            
            result_preprocessed = pytesseract.image_to_string(
                preprocessed_img, 
                lang=config['lang'], 
                config=config['config']
            )
            print("Preprocessed image result:")
            print(repr(result_preprocessed))
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 40)

def main():
    """Run the tests"""
    print("Tesseract French OCR Diagnostic")
    print("="*50)
    
    # Check Tesseract version
    try:
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
    except:
        print("Could not get Tesseract version")
    
    # Test configurations
    test_tesseract_configs()
    
    print("\n" + "="*50)
    print("RECOMMENDATIONS:")
    print("1. The issue might be with image preprocessing")
    print("2. Try different PSM (Page Segmentation Mode) values")
    print("3. Your original document might need better image quality")
    print("4. Consider using config: '--oem 3 --psm 6 -c preserve_interword_spaces=1'")

if __name__ == "__main__":
    main()