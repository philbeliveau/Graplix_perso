#!/usr/bin/env python3
"""
LLM test for medical form - requires API key as command line argument.
Usage: python llm_test_with_key.py YOUR_OPENAI_API_KEY
"""

import os
import sys
import json
import base64
from io import BytesIO
from pathlib import Path

# Get API key from command line argument
if len(sys.argv) < 2:
    print("❌ Usage: python llm_test_with_key.py YOUR_OPENAI_API_KEY")
    print("   Or set OPENAI_API_KEY environment variable")
    sys.exit(1)

OPENAI_API_KEY = sys.argv[1] if len(sys.argv) > 1 else os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY or OPENAI_API_KEY.startswith('-'):
    print("❌ Please provide valid OpenAI API key")
    sys.exit(1)

try:
    import openai
    from PIL import Image
    print("✅ Libraries loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install openai pillow")
    sys.exit(1)

# Initialize client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def test_medical_form():
    """Test LLM OCR on the medical form that was having issues."""
    image_path = "/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data/b3965576-c316-467c-b2c6-df28e8581236.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Medical form not found: {image_path}")
        return False
    
    print("🔍 Testing LLM OCR on medical form")
    print("File:", image_path)
    print("=" * 50)
    
    # Load and prepare image
    try:
        image = Image.open(image_path)
        print(f"📐 Image size: {image.size}")
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large for API
        if max(image.size) > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            print(f"📐 Resized to: {image.size}")
        
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        base64_image = base64.b64encode(buffer.getvalue()).decode()
        print(f"✅ Image encoded successfully")
        
    except Exception as e:
        print(f"❌ Image processing error: {e}")
        return False
    
    # Specialized prompt for French medical forms
    prompt = """You are an expert French document reader. Extract ALL text from this medical form with perfect accuracy.

CRITICAL INSTRUCTIONS:
- This is a French medical form (Clinique de Médecine Familiale)
- Extract EXACTLY what is written, preserving all French accents (é, è, à, ç, ô, etc.)
- Pay special attention to filled-in fields:
  * Nom, Prénom (Name fields)
  * Date de naissance (Birth date)
  * Adresse (Address)
  * Téléphone (Phone number)
  * Nom du médecin (Doctor's name)
- Do NOT skip any text, even if it looks like form templates
- Preserve original line breaks and spacing
- Extract numbers, dates, and names EXACTLY as shown

Return ONLY the extracted text, no explanations or formatting."""

    # Test both models
    models = ["gpt-4o-mini", "gpt-4o"]
    results = {}
    
    for model in models:
        print(f"\n🤖 Testing with {model}...")
        
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
                max_tokens=4000,
                temperature=0.0
            )
            
            extracted_text = response.choices[0].message.content
            
            # Analyze extracted content for PII
            expected_data = {
                "name": ["TREMBLAY", "Steve"],
                "birth_date": ["1991"],
                "phone": ["438", "462"],
                "address": ["211", "Avenue"],
                "medical": ["Georges", "Barbara"]
            }
            
            found_data = {}
            for category, keywords in expected_data.items():
                found_data[category] = []
                for keyword in keywords:
                    if keyword.lower() in extracted_text.lower():
                        found_data[category].append(keyword)
            
            # Count total PII indicators found
            total_found = sum(len(items) for items in found_data.values())
            
            results[model] = {
                "success": True,
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "pii_analysis": found_data,
                "total_pii_indicators": total_found,
                "contains_template_lines": "____" in extracted_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "estimated_cost": (response.usage.prompt_tokens * 0.00015 + response.usage.completion_tokens * 0.0006) / 1000
                }
            }
            
            print(f"✅ Success: {len(extracted_text)} characters extracted")
            print(f"🎯 PII indicators found: {total_found}")
            print(f"💰 Estimated cost: ${results[model]['usage']['estimated_cost']:.4f}")
            
            if found_data:
                for category, items in found_data.items():
                    if items:
                        print(f"   {category}: {items}")
            
        except Exception as e:
            results[model] = {
                "success": False,
                "error": str(e)
            }
            print(f"❌ Failed: {e}")
    
    # Save results
    output_file = "medical_form_llm_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 FINAL RESULTS")
    print("=" * 30)
    
    success_count = sum(1 for r in results.values() if r.get("success"))
    
    if success_count == 0:
        print("❌ All LLM tests failed!")
        print("This suggests an issue with the API key or network connectivity.")
        return False
    
    print(f"✅ {success_count}/{len(models)} models succeeded")
    
    # Find best result
    best_model = None
    best_pii_count = 0
    
    for model, result in results.items():
        if result.get("success"):
            pii_count = result.get("total_pii_indicators", 0)
            print(f"\n🤖 {model}:")
            print(f"   PII indicators: {pii_count}")
            print(f"   Text length: {result.get('text_length', 0)}")
            print(f"   Contains templates: {result.get('contains_template_lines', False)}")
            
            if pii_count > best_pii_count:
                best_pii_count = pii_count
                best_model = model
    
    print(f"\n🏆 Best performer: {best_model} ({best_pii_count} PII indicators)")
    
    if best_model and results[best_model].get("success"):
        best_text = results[best_model]["extracted_text"]
        print(f"\n📄 Sample extraction from {best_model}:")
        print("-" * 40)
        
        # Show relevant lines only
        lines = best_text.split('\n')
        relevant_lines = []
        
        for line in lines:
            line = line.strip()
            if line and any(keyword.lower() in line.lower() for keyword in 
                           ["nom", "prénom", "adresse", "naissance", "téléphone", "tremblay", "steve", "1991", "438"]):
                relevant_lines.append(line)
        
        if relevant_lines:
            print("🎯 Lines containing PII:")
            for line in relevant_lines[:8]:  # Show first 8 relevant lines
                print(f"   {line}")
        else:
            # Fallback: show first few lines
            print("📝 First few lines of extracted text:")
            for line in lines[:10]:
                if line.strip():
                    print(f"   {line.strip()}")
    
    print(f"\n📝 Complete results saved to: {output_file}")
    
    # Conclusion
    if best_pii_count >= 4:
        print(f"\n🎉 SUCCESS: LLM OCR can extract PII from your documents!")
        print("   The issue is likely in the Streamlit dashboard integration, not the LLM models.")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: LLM found some data but may need prompt optimization.")
    
    return True

if __name__ == "__main__":
    success = test_medical_form()
    if success:
        print("\n💡 Next steps:")
        print("   1. Compare these results with your Streamlit dashboard output")
        print("   2. Check if the dashboard is using the same prompts and settings")
        print("   3. Verify image preprocessing in the dashboard pipeline")
    else:
        print("\n🔧 Troubleshooting needed - check API key and connectivity")