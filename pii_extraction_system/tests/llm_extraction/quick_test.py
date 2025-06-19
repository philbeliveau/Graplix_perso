#!/usr/bin/env python3
"""
Quick test to see what's working with OpenAI
"""

import os
import base64
from pathlib import Path
from dotenv import load_dotenv
import openai
from PIL import Image
from io import BytesIO

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def test_single_image():
    """Test with a simple JPG image"""
    data_dir = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data")
    
    # Try with the JPG files (should be simpler)
    jpg_files = list(data_dir.glob("*.jpg"))
    
    if not jpg_files:
        print("❌ No JPG files found")
        return
    
    test_file = jpg_files[0]
    print(f"🔍 Testing with: {test_file.name}")
    
    try:
        # Convert to base64
        image = Image.open(test_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if max(image.size) > 1024:  # Smaller size
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        print("✅ Image encoded successfully")
        
        # Test with minimal prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        print(f"📄 OpenAI Response: {result}")
        
        if "sorry" in result.lower() or "can't" in result.lower():
            print("🚫 OpenAI is refusing to process this image")
        else:
            print("✅ OpenAI processed the image successfully!")
            
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_with_tesseract():
    """Test with Tesseract as backup"""
    try:
        import pytesseract
        from PIL import Image
        
        data_dir = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data")
        jpg_files = list(data_dir.glob("*.jpg"))
        
        if jpg_files:
            test_file = jpg_files[0]
            print(f"\n🔍 Testing Tesseract with: {test_file.name}")
            
            image = Image.open(test_file)
            text = pytesseract.image_to_string(image, lang='fra+eng')
            
            print(f"📄 Tesseract extracted {len(text)} characters")
            if text.strip():
                print(f"✅ Preview: {text[:200]}...")
                return text
            else:
                print("⚠️ Tesseract extracted no text")
                return ""
        
    except ImportError:
        print("⚠️ Tesseract not available (pip install pytesseract)")
        return None
    except Exception as e:
        print(f"❌ Tesseract error: {e}")
        return None

def main():
    print("🧪 Quick OCR Test")
    print("="*40)
    
    # Test OpenAI
    print("\n1️⃣ Testing OpenAI...")
    openai_result = test_single_image()
    
    # Test Tesseract
    print("\n2️⃣ Testing Tesseract...")
    tesseract_result = test_with_tesseract()
    
    # Summary
    print("\n📊 Summary:")
    print("-"*20)
    
    if openai_result and not ("sorry" in openai_result.lower() or "can't" in openai_result.lower()):
        print("✅ OpenAI: Working")
    else:
        print("🚫 OpenAI: Refusing or failing")
    
    if tesseract_result:
        print("✅ Tesseract: Working")
    else:
        print("❌ Tesseract: Not available or failing")
    
    # Recommendation
    print("\n💡 Recommendation:")
    if openai_result and not ("sorry" in openai_result.lower() or "can't" in openai_result.lower()):
        print("   Use OpenAI - it's working!")
    elif tesseract_result:
        print("   Use Tesseract as fallback - OpenAI is being restrictive")
    else:
        print("   Need to install and configure alternative OCR solution")

if __name__ == "__main__":
    main()