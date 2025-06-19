#!/usr/bin/env python3
"""
Test specifically with the problematic PDF
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

def test_pdf_document():
    """Test with the problematic PDF"""
    data_dir = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data")
    test_file = data_dir / "2023-07-07 -Formulaire Absence - Paternité.pdf"
    
    if not test_file.exists():
        print(f"❌ File not found: {test_file}")
        return
    
    print(f"🔍 Testing problematic PDF: {test_file.name}")
    
    try:
        from pdf2image import convert_from_path
        
        # Convert PDF to image with password
        pages = convert_from_path(
            str(test_file),
            dpi=150,  # Lower DPI for faster processing
            userpw="Hubert",
            first_page=1,
            last_page=1
        )
        
        if not pages:
            print("❌ Could not convert PDF")
            return
        
        print("✅ PDF converted to image")
        
        # Convert to base64
        image = pages[0]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Make smaller
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        print("✅ Image encoded")
        
        # Test different prompts on this specific document
        prompts = [
            "What text do you see?",
            "Please read the text in this form.",
            "This is a business document. Please transcribe the text.",
            "I need to digitize this form. What does it say?",
            "Please extract the visible text for data entry purposes."
        ]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}️⃣ Testing prompt: '{prompt}'")
            
            try:
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
                    max_tokens=1000,
                    temperature=0.0
                )
                
                result = response.choices[0].message.content
                print(f"📄 Response: {result[:150]}...")
                
                if "sorry" in result.lower() or "can't" in result.lower() or "unable" in result.lower():
                    print("🚫 REFUSED")
                else:
                    print("✅ SUCCESS!")
                    print(f"Full response: {result}")
                    break  # Found working prompt
                    
            except Exception as e:
                print(f"❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")

if __name__ == "__main__":
    test_pdf_document()