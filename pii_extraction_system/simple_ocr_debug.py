#!/usr/bin/env python3
"""
Simple OCR debugging script to test different approaches on the medical form.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

# Configure tesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def debug_form_ocr():
    """Debug OCR on the medical form with different approaches."""
    image_path = "/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data/b3965576-c316-467c-b2c6-df28e8581236.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔍 Debugging OCR on medical form")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return
    
    height, width = image.shape[:2]
    print(f"📐 Image: {width}x{height} pixels")
    
    # Test 1: Raw image with French language
    print(f"\n🔧 Test 1: Raw image with French (fra)")
    print("-" * 40)
    
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        text = pytesseract.image_to_string(
            pil_image,
            lang='fra',
            config='--oem 3 --psm 6'
        ).strip()
        
        print(f"✅ Text extracted: {len(text)} characters")
        if text:
            # Look for specific fields we expect
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if any(keyword in line.lower() for keyword in ['nom', 'prénom', 'adresse', 'naissance', 'téléphone']):
                    print(f"📋 Line {i+1}: {line}")
        else:
            print("❌ No text found")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Enhanced preprocessing for better field detection
    print(f"\n🔧 Test 2: Enhanced preprocessing")
    print("-" * 40)
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Gentle denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Use adaptive thresholding (better for forms with varying lighting)
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        
        # Convert back to PIL
        pil_enhanced = Image.fromarray(binary)
        
        text = pytesseract.image_to_string(
            pil_enhanced,
            lang='fra',
            config='--oem 3 --psm 6'
        ).strip()
        
        print(f"✅ Enhanced text: {len(text)} characters")
        if text:
            # Look for filled information
            lines = text.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.replace('_', '').replace('-', '').replace(' ', '').isspace():
                    if any(keyword in line.lower() for keyword in ['tremblay', 'steve', '1991', '438', '211', 'avenue']):
                        print(f"🎯 Found data line {i+1}: {line}")
                    elif len(line) > 5 and '___' not in line:
                        print(f"📋 Line {i+1}: {line}")
        else:
            print("❌ No enhanced text found")
            
    except Exception as e:
        print(f"❌ Enhanced preprocessing error: {e}")
    
    # Test 3: Focus on upper region where filled data is
    print(f"\n🔧 Test 3: Crop to upper form region")
    print("-" * 40)
    
    try:
        # Crop to upper portion where the filled data is (roughly top 60%)
        crop_height = int(height * 0.6)
        cropped = image[:crop_height, :]
        
        # Apply preprocessing
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        enhanced_crop = cv2.fastNlMeansDenoising(gray_crop, h=10)
        
        pil_cropped = Image.fromarray(cv2.cvtColor(enhanced_crop, cv2.COLOR_GRAY2RGB))
        
        text = pytesseract.image_to_string(
            pil_cropped,
            lang='fra',
            config='--oem 3 --psm 6'
        ).strip()
        
        print(f"✅ Cropped text: {len(text)} characters")
        if text:
            print("📄 Cropped region text:")
            print(text)
        else:
            print("❌ No text in cropped region")
            
    except Exception as e:
        print(f"❌ Cropping error: {e}")
    
    # Test 4: Different PSM modes for form handling
    print(f"\n🔧 Test 4: Different PSM modes")
    print("-" * 40)
    
    psm_configs = [
        (4, "Single column of text"),
        (6, "Uniform block of text"), 
        (8, "Single word"),
        (11, "Sparse text"),
        (13, "Raw line")
    ]
    
    for psm, description in psm_configs:
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            text = pytesseract.image_to_string(
                pil_image,
                lang='fra',
                config=f'--oem 3 --psm {psm}'
            ).strip()
            
            # Count meaningful content (not just underscores)
            meaningful_lines = [line for line in text.split('\n') 
                              if line.strip() and '___' not in line and len(line.strip()) > 2]
            
            print(f"PSM {psm} ({description}): {len(meaningful_lines)} meaningful lines")
            
            # Show lines with potential data
            for line in meaningful_lines[:3]:  # Show first 3 meaningful lines
                if any(keyword in line.lower() for keyword in ['tremblay', 'steve', '438', '211']):
                    print(f"  🎯 {line}")
                    
        except Exception as e:
            print(f"PSM {psm} error: {e}")

if __name__ == "__main__":
    debug_form_ocr()