#!/usr/bin/env python3
"""
Test the improved OCR configuration on the medical form.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Configure tesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def test_improved_preprocessing():
    """Test the improved preprocessing pipeline on the medical form."""
    image_path = "/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data/b3965576-c316-467c-b2c6-df28e8581236.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print("🔍 Testing Improved OCR Configuration")
    print("=" * 50)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return
    
    # Apply the new form-optimized preprocessing
    print("🔧 Applying form-optimized preprocessing...")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast (important for forms)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Gentle denoising to reduce JPG artifacts without losing text detail
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Use adaptive thresholding (better for forms with varying lighting)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
    )
    
    # Very minimal morphological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Test the new OCR configuration priority
    configs_to_try = [
        ('--oem 3 --psm 6', 'JPG forms - uniform text block'),
        ('--oem 3 --psm 4', 'JPG forms - single column'),
        ('--oem 3 --psm 11', 'JPG forms - sparse text'),
        ('--oem 3 --psm 13', 'JPG forms - raw line')
    ]
    
    best_result = None
    best_quality = 0
    
    for ocr_config, config_name in configs_to_try:
        print(f"\n🔧 Testing: {config_name}")
        print("-" * 30)
        
        try:
            # Convert processed image to PIL
            pil_image = Image.fromarray(processed)
            
            # Get OCR data for quality assessment
            ocr_data = pytesseract.image_to_data(
                pil_image,
                lang='fra',
                config=ocr_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process results and calculate quality score
            text_parts = []
            total_confidence = 0
            valid_detections = 0
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                confidence = int(ocr_data['conf'][i])
                
                if text and confidence > 30:
                    text_parts.append(text)
                    total_confidence += confidence
                    valid_detections += 1
            
            avg_confidence = total_confidence / valid_detections if valid_detections > 0 else 0
            full_text = ' '.join(text_parts)
            meaningful_content_score = 0
            
            # Quality scoring logic (same as in improved code)
            if '___' not in full_text:
                meaningful_content_score += 20
            
            form_indicators = ['nom', 'prénom', 'adresse', 'naissance', 'téléphone', 'date']
            for indicator in form_indicators:
                if indicator.lower() in full_text.lower():
                    meaningful_content_score += 10
            
            import re
            data_patterns = [
                r'\b[A-Z][a-z]+,?\s+[A-Z][a-z]+\b',  # Names
                r'\b\d{4}-\d{2}-\d{2}\b',            # Dates
                r'\b\d{3}[-\s]\d{3}[-\s]\d{4}\b',    # Phone numbers
                r'\b\d+\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Addresses
            ]
            
            for pattern in data_patterns:
                if re.search(pattern, full_text):
                    meaningful_content_score += 15
            
            quality_score = avg_confidence + meaningful_content_score
            
            print(f"Detections: {valid_detections}, Confidence: {avg_confidence:.1f}, Quality: {quality_score:.1f}")
            
            # Show key extracted data
            if quality_score > best_quality:
                best_quality = quality_score
                best_result = {
                    'text': full_text,
                    'config': config_name,
                    'quality': quality_score,
                    'confidence': avg_confidence
                }
            
            # Show lines with potential PII data
            lines = full_text.split()
            pii_indicators = ['TREMBLAY', 'Steve', '1991', '438', '211', 'Avenue']
            found_data = []
            for word in lines:
                if any(indicator in word for indicator in pii_indicators):
                    found_data.append(word)
            
            if found_data:
                print(f"🎯 Found PII data: {' '.join(found_data[:5])}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"🏆 BEST RESULT")
    print(f"Config: {best_result['config'] if best_result else 'None'}")
    print(f"Quality Score: {best_result['quality'] if best_result else 0}")
    print(f"Confidence: {best_result['confidence'] if best_result else 0}")
    
    if best_result:
        print(f"\n📄 Best extracted text:")
        text_lines = best_result['text'].split()
        for i in range(0, len(text_lines), 10):  # Show in chunks
            chunk = ' '.join(text_lines[i:i+10])
            if any(indicator in chunk for indicator in ['TREMBLAY', 'Steve', '1991', '438', '211']):
                print(f"🎯 {chunk}")
            elif chunk.strip():
                print(f"   {chunk}")

if __name__ == "__main__":
    test_improved_preprocessing()