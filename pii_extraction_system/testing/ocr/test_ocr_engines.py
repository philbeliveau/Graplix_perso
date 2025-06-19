#!/usr/bin/env python3
"""
Test script to verify both Tesseract and EasyOCR engines work correctly
"""

import sys
from pathlib import Path
import os

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.document_processor import DocumentProcessor
from core.config import settings

def test_ocr_engines():
    """Test both OCR engines"""
    print("Testing OCR Engines")
    print("=" * 50)
    
    # Check if EasyOCR is available
    try:
        import easyocr
        print("✓ EasyOCR is available")
        easyocr_available = True
    except ImportError:
        print("✗ EasyOCR is not available - install with: pip install easyocr")
        easyocr_available = False
    
    # Check if Tesseract is available
    try:
        import pytesseract
        print("✓ Tesseract is available")
        tesseract_available = True
    except ImportError:
        print("✗ Tesseract is not available")
        tesseract_available = False
    
    if not (easyocr_available or tesseract_available):
        print("No OCR engines available for testing")
        return
    
    print("\nTesting with different OCR engine configurations:")
    print("-" * 50)
    
    # Test configurations
    test_configs = []
    
    if tesseract_available:
        test_configs.append(('tesseract', "Tesseract OCR"))
    
    if easyocr_available:
        test_configs.append(('easyocr', "EasyOCR"))
        test_configs.append(('both', "Both Engines"))
    
    # Create a simple test image with text (you can replace this with an actual image path)
    print("Note: To fully test, provide an image file path with text content")
    print("For now, testing initialization only...")
    
    for engine, description in test_configs:
        print(f"\n{description} ({engine}):")
        try:
            # Temporarily set the OCR engine
            original_engine = settings.processing.ocr_engine
            settings.processing.ocr_engine = engine
            
            # Initialize document processor
            processor = DocumentProcessor()
            
            print(f"  ✓ Document processor initialized successfully")
            print(f"  ✓ OCR engine set to: {engine}")
            
            if engine in ['easyocr', 'both'] and processor.easyocr_reader:
                print(f"  ✓ EasyOCR reader initialized")
            elif engine in ['easyocr', 'both']:
                print(f"  ⚠ EasyOCR reader not initialized (may need GPU/dependencies)")
            
            # Restore original setting
            settings.processing.ocr_engine = original_engine
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nCurrent OCR configuration:")
    print(f"  - OCR Engine: {settings.processing.ocr_engine}")
    print(f"  - OCR Languages: {settings.processing.ocr_languages}")
    print(f"  - Use GPU: {settings.processing.easyocr_use_gpu}")
    
    print(f"\nTo test with actual documents:")
    print(f"  1. Install EasyOCR: pip install easyocr")
    print(f"  2. Use the Streamlit UI to select OCR engine in Configuration > Performance Settings")
    print(f"  3. Process image documents to compare results")

if __name__ == "__main__":
    test_ocr_engines()