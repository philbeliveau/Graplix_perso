#!/usr/bin/env python3
"""
Debug the OCR issue to see what's happening
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_basic_ocr():
    """Test basic OCR functionality"""
    print("=== Testing Basic OCR ===")
    
    # Create a simple test image
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    test_text = "Hello World Test"
    draw.text((20, 30), test_text, fill='black', font=font)
    
    print(f"Original text: {test_text}")
    
    # Test with default settings
    print("\n--- Default Tesseract ---")
    try:
        result = pytesseract.image_to_string(img)
        print(f"Result: '{result.strip()}'")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with French language
    print("\n--- French Language ---")
    try:
        result = pytesseract.image_to_string(img, lang='fra')
        print(f"Result: '{result.strip()}'")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with optimized config
    print("\n--- Optimized Config ---")
    try:
        ocr_config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
        result = pytesseract.image_to_string(img, lang='fra', config=ocr_config)
        print(f"Result: '{result.strip()}'")
    except Exception as e:
        print(f"Error: {e}")

def test_document_processor():
    """Test the document processor with a simple image"""
    print("\n=== Testing Document Processor ===")
    
    try:
        from utils.document_processor import DocumentProcessor
        from core.config import settings
        
        print(f"Current OCR engine: {settings.processing.ocr_engine}")
        print(f"Current OCR languages: {settings.processing.ocr_languages}")
        
        # Create a simple test image file
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        test_text = "Québec Médecine Test"
        draw.text((20, 30), test_text, fill='black', font=font)
        
        # Save as test file
        test_file = "debug_test.png"
        img.save(test_file)
        
        print(f"Created test file: {test_file}")
        print(f"Test text: {test_text}")
        
        # Test document processor
        processor = DocumentProcessor()
        result = processor.process_document(test_file)
        
        print(f"Raw text: '{result.get('raw_text', 'NO TEXT')}' (length: {len(result.get('raw_text', ''))})")
        print(f"OCR text: '{result.get('ocr_text', 'NO OCR TEXT')}' (length: {len(result.get('ocr_text', ''))})")
        print(f"OCR engine: {result.get('ocr_engine', 'UNKNOWN')}")
        print(f"Bounding boxes: {len(result.get('bounding_boxes', []))}")
        
        # Clean up
        import os
        if os.path.exists(test_file):
            os.remove(test_file)
            
    except Exception as e:
        print(f"Document processor test failed: {e}")
        import traceback
        traceback.print_exc()

def test_image_preprocessing():
    """Test the image preprocessing function"""
    print("\n=== Testing Image Preprocessing ===")
    
    try:
        from utils.document_processor import DocumentProcessor
        
        # Create a test image
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        test_text = "Test Preprocessing"
        draw.text((20, 30), test_text, fill='black', font=font)
        
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Test preprocessing
        processor = DocumentProcessor()
        enhanced = processor._enhance_image_for_ocr(img_cv)
        
        print("Original image shape:", img_cv.shape)
        print("Enhanced image shape:", enhanced.shape)
        
        # Test OCR on both
        original_pil = img
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        # OCR on original
        original_result = pytesseract.image_to_string(original_pil)
        print(f"Original OCR: '{original_result.strip()}'")
        
        # OCR on enhanced
        enhanced_result = pytesseract.image_to_string(enhanced_pil, lang='fra', config='--oem 3 --psm 6 -c preserve_interword_spaces=1')
        print(f"Enhanced OCR: '{enhanced_result.strip()}'")
        
        # Save images for inspection
        enhanced_pil.save("debug_enhanced.png")
        print("Saved enhanced image as debug_enhanced.png")
        
    except Exception as e:
        print(f"Preprocessing test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all debug tests"""
    print("OCR Debug Tool")
    print("=" * 50)
    
    test_basic_ocr()
    test_document_processor()
    test_image_preprocessing()
    
    print("\n" + "=" * 50)
    print("If any test shows empty results, we've found the issue!")

if __name__ == "__main__":
    main()