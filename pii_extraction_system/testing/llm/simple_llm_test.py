#!/usr/bin/env python3
"""
Simple LLM test focused on the medical form to verify GPT can extract PII.
"""

import os
import sys
import json
import base64
from io import BytesIO
from pathlib import Path

# Prompt for API key if not set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        import getpass
        OPENAI_API_KEY = getpass.getpass("Enter your OpenAI API key: ")
        if not OPENAI_API_KEY:
            print("❌ No API key provided")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Cancelled")
        sys.exit(1)

try:
    import openai
    from PIL import Image
    print("✅ Libraries loaded")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install: pip install openai pillow")
    sys.exit(1)

# Initialize client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def test_medical_form():
    """Test LLM OCR on the specific medical form."""
    image_path = "/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data/b3965576-c316-467c-b2c6-df28e8581236.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    print("🔍 Testing LLM OCR on medical form")
    print("=" * 40)
    
    # Load and encode image
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if needed
        if max(image.size) > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        base64_image = base64.b64encode(buffer.getvalue()).decode()
        
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return
    
    # Test prompt focused on French medical forms
    prompt = """Extract ALL text from this French medical form. Focus on:

NAMES: Look for "Nom, Prénom" fields
DATES: Look for "Date de naissance" and other dates  
ADDRESSES: Look for "Adresse" fields
PHONE: Look for "Téléphone" numbers
MEDICAL: Look for doctor names and medical information

Extract EXACTLY as written, preserving all French accents (é, è, à, ç, etc.).
Return the complete text without any interpretation."""

    models = ["gpt-4o-mini", "gpt-4o"]
    results = {}
    
    for model in models:
        print(f"\n🤖 Testing {model}...")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=3000,
                temperature=0.0
            )
            
            extracted_text = response.choices[0].message.content
            
            # Look for key PII data
            pii_found = []
            keywords = ["TREMBLAY", "Steve", "1991", "438", "211", "Avenue", "Téléphone", "Nom", "Adresse"]
            
            for keyword in keywords:
                if keyword.lower() in extracted_text.lower():
                    pii_found.append(keyword)
            
            results[model] = {
                "success": True,
                "text_length": len(extracted_text),
                "extracted_text": extracted_text,
                "pii_keywords_found": pii_found,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            print(f"✅ Success: {len(extracted_text)} chars, {len(pii_found)} PII keywords")
            if pii_found:
                print(f"   Found: {', '.join(pii_found[:5])}")
            
        except Exception as e:
            results[model] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ Error: {e}")
    
    # Save and display results
    output_file = "medical_form_llm_test.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 SUMMARY")
    print("=" * 20)
    
    for model, result in results.items():
        if result.get("success"):
            print(f"✅ {model}: Found {len(result['pii_keywords_found'])} PII indicators")
            print(f"   Text extracted: {result['text_length']} characters")
        else:
            print(f"❌ {model}: Failed - {result.get('error', 'Unknown error')}")
    
    print(f"\n📝 Full results saved to: {output_file}")
    
    # Show sample extraction from best result
    best_result = None
    for result in results.values():
        if result.get("success") and (not best_result or len(result['pii_keywords_found']) > len(best_result['pii_keywords_found'])):
            best_result = result
    
    if best_result:
        print(f"\n📄 Best extraction preview:")
        print("-" * 30)
        preview = best_result['extracted_text'][:500]
        print(preview + "..." if len(best_result['extracted_text']) > 500 else preview)
        
        # Look for specific lines with PII
        lines = best_result['extracted_text'].split('\n')
        pii_lines = []
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in ["nom", "tremblay", "steve", "1991", "adresse", "téléphone"]):
                pii_lines.append(line.strip())
        
        if pii_lines:
            print(f"\n🎯 Lines containing PII:")
            for line in pii_lines[:5]:  # Show first 5
                print(f"   {line}")

if __name__ == "__main__":
    test_medical_form()