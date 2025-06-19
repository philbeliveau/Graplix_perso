#!/usr/bin/env python3
"""
Test French OCR with both Tesseract and EasyOCR to diagnose issues
"""

import sys
import os
from pathlib import Path

def test_easyocr_installation():
    """Test if EasyOCR is properly installed"""
    print("=== Testing EasyOCR Installation ===")
    
    try:
        import easyocr
        print("✓ EasyOCR imported successfully")
        
        # Test initialization
        reader = easyocr.Reader(['en', 'fr'])
        print("✓ EasyOCR Reader initialized with English and French")
        
        return reader
    except ImportError as e:
        print(f"✗ EasyOCR import failed: {e}")
        print("Install with: pip install easyocr")
        return None
    except Exception as e:
        print(f"✗ EasyOCR initialization failed: {e}")
        return None

def test_tesseract_french():
    """Test Tesseract with French language"""
    print("\n=== Testing Tesseract French Support ===")
    
    try:
        import pytesseract
        
        # Check available languages
        languages = pytesseract.get_languages()
        print(f"Available languages: {languages}")
        
        if 'fra' in languages:
            print("✓ French language pack (fra) is available")
        else:
            print("✗ French language pack (fra) is NOT available")
            print("Install with: brew install tesseract-lang")
        
        return 'fra' in languages
    except Exception as e:
        print(f"✗ Tesseract test failed: {e}")
        return False

def test_sample_french_text():
    """Test OCR with sample French text"""
    print("\n=== Testing with Sample French Text ===")
    
    # Create a simple test image with French text
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create a simple image with French text
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # French text with accents
        french_text = "Québec Médecine Laval Hôpital"
        
        try:
            # Try to use a system font
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 30), french_text, fill='black', font=font)
        
        # Save test image
        test_image_path = "test_french.png"
        img.save(test_image_path)
        
        print(f"Created test image: {test_image_path}")
        print(f"Test text: {french_text}")
        
        # Test with Tesseract
        print("\n--- Tesseract Results ---")
        try:
            import pytesseract
            tesseract_result = pytesseract.image_to_string(img, lang='fra')
            print(f"Tesseract (fra): '{tesseract_result.strip()}'")
            
            tesseract_result_eng = pytesseract.image_to_string(img, lang='eng')
            print(f"Tesseract (eng): '{tesseract_result_eng.strip()}'")
        except Exception as e:
            print(f"Tesseract failed: {e}")
        
        # Test with EasyOCR
        print("\n--- EasyOCR Results ---")
        try:
            import easyocr
            reader = easyocr.Reader(['en', 'fr'])
            easyocr_results = reader.readtext(np.array(img))
            
            for detection in easyocr_results:
                bbox, text, confidence = detection
                print(f"EasyOCR: '{text}' (confidence: {confidence:.2f})")
        except Exception as e:
            print(f"EasyOCR failed: {e}")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            
    except Exception as e:
        print(f"Test image creation failed: {e}")

def test_document_processor():
    """Test the actual document processor"""
    print("\n=== Testing Document Processor ===")
    
    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))
        
        from utils.document_processor import DocumentProcessor, EASYOCR_AVAILABLE
        from core.config import settings
        
        print(f"EasyOCR available in document processor: {EASYOCR_AVAILABLE}")
        print(f"Current OCR engine setting: {settings.processing.ocr_engine}")
        
        # Test initialization
        processor = DocumentProcessor()
        print("✓ DocumentProcessor initialized")
        
        if hasattr(processor, 'easyocr_reader') and processor.easyocr_reader:
            print("✓ EasyOCR reader is initialized in DocumentProcessor")
        else:
            print("✗ EasyOCR reader is NOT initialized in DocumentProcessor")
        
    except Exception as e:
        print(f"Document processor test failed: {e}")

def main():
    """Run all tests"""
    print("French OCR Diagnosis Tool")
    print("=" * 50)
    
    # Test 1: EasyOCR installation
    easyocr_reader = test_easyocr_installation()
    
    # Test 2: Tesseract French support
    tesseract_french_ok = test_tesseract_french()
    
    # Test 3: Sample text OCR
    test_sample_french_text()
    
    # Test 4: Document processor
    test_document_processor()
    
    # Summary
    print("\n" + "=" * 50)
    print("DIAGNOSIS SUMMARY:")
    print("=" * 50)
    
    if easyocr_reader:
        print("✓ EasyOCR is available and should work for French text")
    else:
        print("✗ EasyOCR is not available - this could be why you're getting bad results")
    
    if tesseract_french_ok:
        print("✓ Tesseract French language pack is installed")
    else:
        print("✗ Tesseract French language pack is missing")
    
    print("\nRECOMMENDATIONS:")
    print("1. Try selecting 'EasyOCR' in the document processing interface")
    print("2. If still getting bad results, the issue might be image quality or preprocessing")
    print("3. Try the 'both' option to compare Tesseract vs EasyOCR results")
    print("4. Check if the document image is clear and high resolution")

if __name__ == "__main__":
    main()