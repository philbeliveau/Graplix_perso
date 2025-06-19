#!/usr/bin/env python3
"""
Simple test to see what OpenAI will actually process
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

if not OPENAI_API_KEY:
    print("❌ Error: OPENAI_API_KEY not found in .env file")
    exit(1)

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def encode_image_to_base64(image_input):
    """Convert image (path or PIL) to base64."""
    try:
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        else:
            image = image_input
            
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        if max(image.size) > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

def test_different_prompts(image_data, file_name):
    """Test different prompt approaches"""
    
    prompts = {
        "Simple": "What text do you see in this image?",
        
        "Academic": "I am a researcher studying document digitization. Please transcribe the text visible in this image for academic purposes.",
        
        "Technical": "Please perform optical character recognition on this document image and provide the extracted text.",
        
        "Business": "As part of a document management system, please extract all visible text from this business document.",
        
        "Data Processing": "This is for data inventory purposes. Please transcribe all text visible in this document image."
    }
    
    results = {}
    
    for prompt_name, prompt_text in prompts.items():
        print(f"\n🔍 Testing '{prompt_name}' prompt...")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
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
            
            extracted_text = response.choices[0].message.content
            results[prompt_name] = {
                "success": True,
                "text": extracted_text,
                "length": len(extracted_text),
                "tokens": response.usage.total_tokens
            }
            
            print(f"✅ Success: {len(extracted_text)} chars")
            if "sorry" in extracted_text.lower() or "can't" in extracted_text.lower():
                print(f"⚠️  Refusal detected: {extracted_text[:100]}...")
            else:
                print(f"📄 Preview: {extracted_text[:100]}...")
            
        except Exception as e:
            results[prompt_name] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ Failed: {e}")
    
    return results

def test_different_models(image_data):
    """Test different OpenAI models"""
    
    models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo"
    ]
    
    simple_prompt = "Please transcribe the text in this image."
    results = {}
    
    for model in models:
        print(f"\n🤖 Testing model: {model}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": simple_prompt},
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
            
            extracted_text = response.choices[0].message.content
            results[model] = {
                "success": True,
                "text": extracted_text,
                "length": len(extracted_text),
                "refused": "sorry" in extracted_text.lower() or "can't" in extracted_text.lower()
            }
            
            print(f"✅ Success: {len(extracted_text)} chars")
            if results[model]["refused"]:
                print(f"⚠️  Refusal: {extracted_text[:100]}...")
            else:
                print(f"📄 Success: {extracted_text[:100]}...")
            
        except Exception as e:
            results[model] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ Failed: {e}")
    
    return results

def main():
    # Test on the first document we know OpenAI is refusing
    data_dir = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data")
    
    # Try the form document that had issues
    test_file = data_dir / "2023-07-07 -Formulaire Absence - Paternité.pdf"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print(f"🔍 Testing OpenAI OCR capabilities")
    print(f"📄 File: {test_file.name}")
    
    # Convert PDF to image first (using pdf2image)
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(str(test_file), dpi=200, first_page=1, last_page=1)
        if not pages:
            print("❌ Could not convert PDF to image")
            return
        
        # Encode first page
        image_data = encode_image_to_base64(pages[0])
        if not image_data:
            print("❌ Could not encode image")
            return
        
        print("✅ PDF converted to image successfully")
        
        # Test different prompts
        print("\n" + "="*50)
        print("TESTING DIFFERENT PROMPTS")
        print("="*50)
        
        prompt_results = test_different_prompts(image_data, test_file.name)
        
        # Test different models
        print("\n" + "="*50)
        print("TESTING DIFFERENT MODELS")
        print("="*50)
        
        model_results = test_different_models(image_data)
        
        # Summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        
        print("\n📊 Prompt Results:")
        for prompt_name, result in prompt_results.items():
            if result["success"]:
                status = "🚫 REFUSED" if ("sorry" in result["text"].lower() or "can't" in result["text"].lower()) else "✅ SUCCESS"
                print(f"  {prompt_name}: {status} ({result['length']} chars)")
            else:
                print(f"  {prompt_name}: ❌ ERROR")
        
        print("\n🤖 Model Results:")
        for model, result in model_results.items():
            if result["success"]:
                status = "🚫 REFUSED" if result["refused"] else "✅ SUCCESS"
                print(f"  {model}: {status} ({result['length']} chars)")
            else:
                print(f"  {model}: ❌ ERROR")
        
        # Find any successful extractions
        successful_prompts = [name for name, result in prompt_results.items() 
                            if result["success"] and not ("sorry" in result["text"].lower() or "can't" in result["text"].lower())]
        
        successful_models = [model for model, result in model_results.items() 
                           if result["success"] and not result["refused"]]
        
        if successful_prompts or successful_models:
            print(f"\n🎉 FOUND WORKING APPROACHES!")
            if successful_prompts:
                print(f"✅ Working prompts: {successful_prompts}")
            if successful_models:
                print(f"✅ Working models: {successful_models}")
        else:
            print(f"\n⚠️  ALL APPROACHES FAILED - OpenAI refuses this document type")
            print(f"💡 Recommendation: Use alternative OCR (Tesseract, EasyOCR, etc.)")
        
    except ImportError:
        print("❌ pdf2image not available. Install: pip install pdf2image")
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")

if __name__ == "__main__":
    main()