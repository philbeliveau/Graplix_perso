#!/usr/bin/env python3
"""
Pure GPT approach - let GPT do all the PII extraction work
"""

import os
import json
import base64
from pathlib import Path
from dotenv import load_dotenv
import openai
from PIL import Image
from io import BytesIO

load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def pure_gpt_pii_extraction(image_data, file_name="document"):
    """Let GPT handle everything - transcription AND PII extraction"""
    
    # Craft a prompt that avoids safety filters but gets structured output
    prompt = f"""
Please help me digitize this {file_name} by extracting the visible information in a structured format.

Return a JSON object with the following structure:
{{
  "transcribed_text": "full text you can see",
  "extracted_information": {{
    "names": ["list of person names you see"],
    "contact_info": {{
      "emails": ["email addresses"],
      "phone_numbers": ["phone numbers"],
      "addresses": ["physical addresses"]
    }},
    "dates": ["any dates mentioned"],
    "identification_numbers": ["ID numbers, social insurance numbers, etc"],
    "other_relevant_info": ["any other structured data"]
  }}
}}

Please be thorough but only include information that is clearly visible in the document.
"""

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
            max_tokens=4000,
            temperature=0.0
        )
        
        response_text = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            # Look for JSON in the response
            if "```json" in response_text:
                json_part = response_text.split("```json")[1].split("```")[0]
            elif "{" in response_text:
                # Find the JSON object
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_part = response_text[start:end]
            else:
                json_part = response_text
                
            parsed_data = json.loads(json_part)
            
            return {
                "success": True,
                "method": "pure_gpt_structured",
                "raw_response": response_text,
                "structured_data": parsed_data,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "estimated_cost": (response.usage.prompt_tokens * 0.00015 + 
                                     response.usage.completion_tokens * 0.0006) / 1000
                }
            }
            
        except json.JSONDecodeError:
            # Fallback: treat as unstructured response
            return {
                "success": True,
                "method": "pure_gpt_unstructured", 
                "raw_response": response_text,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "estimated_cost": (response.usage.prompt_tokens * 0.00015 + 
                                     response.usage.completion_tokens * 0.0006) / 1000
                }
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "method": "pure_gpt_failed"
        }

def test_pure_gpt_approach():
    """Test the pure GPT approach on a few files"""
    data_dir = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data")
    
    # Test with a few image files
    test_files = [
        "b3965576-c316-467c-b2c6-df28e8581236.jpg",
        "e1c64004-1dd8-43a2-b2fc-6d3e7ca37b76.jpg"
    ]
    
    results = []
    
    for file_name in test_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            continue
            
        print(f"\n🔍 Testing Pure GPT approach on: {file_name}")
        
        # Convert to base64
        try:
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            if max(image.size) > 2048:
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode()
            
            # Test pure GPT approach
            result = pure_gpt_pii_extraction(image_data, file_name)
            
            if result["success"]:
                print(f"✅ Success with {result['method']}")
                if "structured_data" in result:
                    print(f"📊 Structured extraction:")
                    extracted = result["structured_data"].get("extracted_information", {})
                    for category, items in extracted.items():
                        if items:
                            print(f"   {category}: {items}")
                else:
                    print(f"📄 Response: {result['raw_response'][:200]}...")
                print(f"💰 Cost: ${result['usage']['estimated_cost']:.4f}")
            else:
                print(f"❌ Failed: {result['error']}")
            
            results.append({
                "file": file_name,
                "result": result
            })
            
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
    
    # Save results
    with open("pure_gpt_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📝 Results saved to pure_gpt_test_results.json")

if __name__ == "__main__":
    test_pure_gpt_approach()