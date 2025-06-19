#!/usr/bin/env python3
"""
Quick test with one file to verify the new approach
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

def test_one_document():
    """Test the new PII extraction approach on one document"""
    data_dir = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data")
    
    # Test with the JPG that was working
    test_file = data_dir / "b3965576-c316-467c-b2c6-df28e8581236.jpg"
    
    if not test_file.exists():
        print(f"❌ File not found: {test_file}")
        return
    
    print(f"🔍 Testing new approach with: {test_file.name}")
    
    try:
        # Convert to base64
        image = Image.open(test_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        print("✅ Image encoded successfully")
        
        # Simple prompt that works
        prompt = "What text do you see in this image? Please include all names, dates, addresses, and phone numbers you can read."
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000
        )
        
        result = response.choices[0].message.content
        print(f"📄 OpenAI Response:")
        print("="*60)
        print(result)
        print("="*60)
        
        if "I'm sorry" in result or "can't assist" in result:
            print("🚫 OpenAI refused to process")
        else:
            print("✅ OpenAI processed successfully!")
            
            # Check if it used our format
            if "PERSONAL_INFO:" in result:
                print("🎯 OpenAI used our structured format!")
            else:
                print("⚠️ OpenAI didn't use structured format")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_one_document()