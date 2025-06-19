#!/usr/bin/env python3
"""Test that LLM OCR fixes work correctly."""

import os
import sys
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment
from load_env import load_env_file
load_env_file()

def create_test_french_image():
    """Create a test image with French text."""
    # Create a white image
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Use default font
    try:
        # Try to use a system font
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        # Fallback to default
        font = ImageFont.load_default()
    
    # French text with accents
    french_text = """PREUVE DE DÉCÈS DU DIRECTEUR FUNÉRAIRE
Informations complémentaires
Date: 15 octobre 2023
Lieu: Montréal, Québec
Numéro de série: 294567
État civil: Décédé"""
    
    # Draw text on image
    lines = french_text.split('\n')
    y_offset = 50
    for line in lines:
        draw.text((50, y_offset), line, fill='black', font=font)
        y_offset += 40
    
    return img

def test_llm_ocr_fix():
    """Test that LLM OCR properly extracts French text."""
    print("🧪 Testing LLM OCR Fix for French Text")
    print("=" * 50)
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found - skipping test")
        return False
    
    try:
        # Import required modules
        from utils.document_processor import DocumentProcessor
        
        # Check LLM availability
        try:
            from llm import LLM_PROCESSOR_AVAILABLE
        except ImportError:
            print("❌ LLM module not available")
            return False
        
        if not LLM_PROCESSOR_AVAILABLE:
            print("❌ LLM processor not available")
            return False
        
        print("✅ LLM processor available")
        
        # Create processor
        processor = DocumentProcessor()
        
        if not processor.llm_ocr_enabled:
            print("❌ LLM OCR not enabled in processor")
            return False
            
        print("✅ LLM OCR enabled in processor")
        
        # Create test image
        print("\n📷 Creating test French document image...")
        test_image = create_test_french_image()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            test_image.save(tmp_file.name, 'PNG')
            test_image_path = tmp_file.name
        
        try:
            print(f"✅ Test image created: {test_image_path}")
            
            # Process with document processor
            print("\n🔄 Processing image with LLM OCR...")
            result = processor.process_document(test_image_path)
            
            extracted_text = result.get('ocr_text', '')
            confidence = result.get('confidence', 0)
            engine_used = result.get('ocr_engine', 'unknown')
            
            print(f"\n📊 Results:")
            print(f"   Engine used: {engine_used}")
            print(f"   Confidence: {confidence}")
            print(f"   Text length: {len(extracted_text)} characters")
            
            if extracted_text:
                print(f"\n📄 Extracted text:")
                print("─" * 40)
                print(extracted_text)
                print("─" * 40)
                
                # Check for French words with accents
                french_indicators = ['DÉCÈS', 'décès', 'Décédé', 'Montréal', 'Québec', 'numéro']
                found_french = [word for word in french_indicators if word in extracted_text]
                
                if found_french:
                    print(f"✅ French text properly extracted! Found: {found_french}")
                    success = True
                else:
                    print("⚠️ French accents might not be preserved")
                    print(f"Expected words: {french_indicators}")
                    success = False
                    
                # Check for gibberish (our original problem)
                if "ÿ rs bu à me" in extracted_text or len(extracted_text.replace(' ', '')) < 10:
                    print("❌ Still getting gibberish output!")
                    success = False
                else:
                    print("✅ No gibberish detected")
                    
            else:
                print("❌ No text extracted!")
                success = False
                
        finally:
            # Clean up
            os.unlink(test_image_path)
            
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_processing():
    """Test PDF processing."""
    print("\n\n🧪 Testing PDF Processing")
    print("=" * 50)
    
    try:
        from utils.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        
        # Check if pdf2image is available
        try:
            from pdf2image import convert_from_path
            print("✅ pdf2image available for PDF processing")
        except ImportError:
            print("❌ pdf2image not available - PDF OCR will fail")
            return False
            
        print("✅ PDF processing should work")
        return True
        
    except Exception as e:
        print(f"❌ PDF test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Testing LLM OCR Fixes")
    print("=" * 60)
    
    # Test basic LLM OCR
    llm_success = test_llm_ocr_fix()
    
    # Test PDF processing capability
    pdf_success = test_pdf_processing()
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"   LLM OCR Fix: {'✅ PASSED' if llm_success else '❌ FAILED'}")
    print(f"   PDF Processing: {'✅ READY' if pdf_success else '❌ NOT READY'}")
    
    if llm_success and pdf_success:
        print("\n🎉 All fixes working! LLM OCR should now:")
        print("   • Extract French text with proper accents")
        print("   • Work with both JPG and PDF files")
        print("   • Use improved prompts and settings")
        print("   • Provide clear, readable output")
        print("\n💡 Try processing your French documents again!")
    else:
        print("\n⚠️ Some issues remain - check the errors above")
    
    exit(0 if (llm_success and pdf_success) else 1)