#!/usr/bin/env python3
"""
Test JPG-specific OCR preprocessing
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import io

def create_test_image_with_compression():
    """Create test images with different compression levels"""
    # Create a clear test image with French text
    img = Image.new('RGB', (600, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    # French text similar to your document
    french_text = """Québec MÉDECINE
Laval Hôpital de la Santé
175, boul. René Lévesque
Date de naissance: 14/05/2024
Adresse: 123 rue des Érables"""
    
    # Draw text line by line
    lines = french_text.strip().split('\n')
    y_pos = 20
    for line in lines:
        draw.text((20, y_pos), line, fill='black', font=font)
        y_pos += 30
    
    # Save as PNG (lossless)
    img.save("test_clean.png", "PNG")
    
    # Save as JPG with different quality levels
    for quality in [95, 80, 60, 30]:
        img.save(f"test_jpg_q{quality}.jpg", "JPEG", quality=quality)
    
    return img

def test_jpg_preprocessing():
    """Test JPG preprocessing pipeline"""
    print("Testing JPG-specific OCR preprocessing...")
    
    # Create test images
    create_test_image_with_compression()
    
    # Test files
    test_files = [
        ("test_clean.png", "PNG (lossless)"),
        ("test_jpg_q95.jpg", "JPG Quality 95%"),
        ("test_jpg_q80.jpg", "JPG Quality 80%"),
        ("test_jpg_q60.jpg", "JPG Quality 60%"),
        ("test_jpg_q30.jpg", "JPG Quality 30%")
    ]
    
    print("\n" + "="*80)
    print("Testing OCR results with different image formats/qualities:")
    print("="*80)
    
    for filename, description in test_files:
        print(f"\n--- {description} ({filename}) ---")
        
        try:
            # Load image
            img = cv2.imread(filename)
            if img is None:
                print(f"Could not load {filename}")
                continue
            
            # Test basic OCR
            pil_img = Image.open(filename)
            basic_result = pytesseract.image_to_string(pil_img, lang='fra')
            
            # Apply JPG-specific preprocessing if needed
            if filename.endswith('.jpg'):
                # Simulate the JPG preprocessing pipeline
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # More aggressive denoising for JPG
                denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
                
                # Gaussian blur to smooth artifacts
                blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
                
                # Bilateral filter to preserve edges
                bilateral = cv2.bilateralFilter(blurred, 9, 75, 75)
                
                # Sharpening
                kernel_sharpen = np.array([[-1,-1,-1],
                                         [-1, 9,-1],
                                         [-1,-1,-1]])
                sharpened = cv2.filter2D(bilateral, -1, kernel_sharpen)
                
                # OTSU thresholding
                _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Opening to remove artifacts
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                
                # Convert back to PIL for OCR
                processed_pil = Image.fromarray(processed)
                
                # OCR with JPG-optimized config
                enhanced_result = pytesseract.image_to_string(
                    processed_pil, 
                    lang='fra',
                    config='--oem 3 --psm 6 -c textord_old_xheight=0 -c textord_min_xheight=2'
                )
                
                print("Basic OCR result:")
                print(repr(basic_result.strip()[:100]))
                print("\nJPG-Enhanced OCR result:")
                print(repr(enhanced_result.strip()[:100]))
                
                # Save processed image for inspection
                cv2.imwrite(f"processed_{filename}", processed)
                
            else:
                print("OCR result:")
                print(repr(basic_result.strip()[:100]))
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        
        print("-" * 60)
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print("1. If JPG results are poor, try converting to PNG first")
    print("2. For documents, always use PNG or TIFF if possible")
    print("3. If using JPG, use quality 90+ for text documents")
    print("4. The enhanced preprocessing should improve JPG results")

def main():
    """Run JPG OCR tests"""
    print("JPG OCR Quality Test")
    print("="*50)
    
    test_jpg_preprocessing()
    
    # Cleanup
    import os
    test_files = ["test_clean.png", "test_jpg_q95.jpg", "test_jpg_q80.jpg", 
                  "test_jpg_q60.jpg", "test_jpg_q30.jpg"]
    
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
    
    # Remove processed files
    for f in test_files:
        processed_f = f"processed_{f}"
        if os.path.exists(processed_f):
            os.remove(processed_f)

if __name__ == "__main__":
    main()