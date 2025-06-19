#!/usr/bin/env python3
"""
Debug script to test OCR on the specific medical form image.
This will help identify why the form fields are not being read correctly.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Add src to path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

from utils.document_processor import DocumentProcessor
from core.config import settings

# Configure tesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def debug_ocr_on_form():
    """Debug OCR extraction on the medical form."""
    image_path = "/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data/b3965576-c316-467c-b2c6-df28e8581236.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔍 Debugging OCR on: {image_path}")
    print("=" * 50)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return
    
    print(f"📐 Image dimensions: {image.shape}")
    
    # Test different OCR approaches
    approaches = [
        ("Raw image - PSM 4", "--oem 3 --psm 4"),
        ("Raw image - PSM 6", "--oem 3 --psm 6"), 
        ("Raw image - PSM 8", "--oem 3 --psm 8"),
        ("Raw image - PSM 11", "--oem 3 --psm 11"),
        ("Raw image - PSM 13", "--oem 3 --psm 13"),
    ]
    
    for approach_name, config in approaches:
        print(f"\n🔧 Testing: {approach_name}")
        print("-" * 30)
        
        try:
            # Convert to PIL
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Extract text
            text = pytesseract.image_to_string(
                pil_image,
                lang='fra',  # French language
                config=config
            ).strip()
            
            print(f"Text length: {len(text)} characters")
            if text:
                print("📄 Extracted text:")
                print(text[:500] + "..." if len(text) > 500 else text)
            else:
                print("❌ No text extracted")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 Testing with image preprocessing...")
    
    # Test with preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Gentle preprocessing for form documents
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Try adaptive thresholding (better for forms)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    pil_processed = Image.fromarray(binary)
    
    print(f"\n🔧 Testing: Preprocessed image - PSM 6")
    print("-" * 30)
    
    try:
        text = pytesseract.image_to_string(
            pil_processed,
            lang='fra',
            config='--oem 3 --psm 6'
        ).strip()
        
        print(f"Text length: {len(text)} characters")
        if text:
            print("📄 Extracted text:")
            print(text[:500] + "..." if len(text) > 500 else text)
        else:
            print("❌ No text extracted")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🔍 Testing with DocumentProcessor...")
    
    # Test with our actual document processor
    try:
        processor = DocumentProcessor()
        result = processor.process_document(Path(image_path))
        
        print(f"📄 DocumentProcessor result:")
        print(f"OCR text length: {len(result.get('ocr_text', ''))}")
        print(f"Raw text length: {len(result.get('raw_text', ''))}")
        
        ocr_text = result.get('ocr_text', '')
        if ocr_text:
            print("OCR Text:")
            print(ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text)
        
        print(f"\nBounding boxes found: {len(result.get('bounding_boxes', []))}")
        
    except Exception as e:
        print(f"❌ DocumentProcessor error: {e}")

if __name__ == "__main__":
    debug_ocr_on_form()