#!/usr/bin/env python3
"""
Standalone LLM OCR test to verify GPT-3.5 and GPT-4o can extract PII from documents.
This bypasses the Streamlit dashboard to test the LLM integration directly.
"""

import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any

# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        import getpass
        OPENAI_API_KEY = getpass.getpass("Please enter your OpenAI API key: ")
        if not OPENAI_API_KEY:
            print("❌ No API key provided")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
        sys.exit(1)

# Import required libraries
try:
    import openai
    import cv2
    import numpy as np
    from PIL import Image
    import base64
    from io import BytesIO
    print("✅ Libraries imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install openai opencv-python pillow")
    sys.exit(1)

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def encode_image(image_path):
    """Encode image to base64."""
    try:
        # Load and convert image
        if str(image_path).lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
            # For image files
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            # For other formats, try PIL directly
            pil_image = Image.open(image_path)
        
        # Convert to RGB if not already
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Resize if too large (GPT-4 has size limits)
        max_size = 2048
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
        
    except Exception as e:
        print(f"❌ Error encoding image {image_path}: {e}")
        return None

def process_with_llm(image_path, model="gpt-4o-mini"):
    """Process image with LLM OCR."""
    print(f"\n🔍 Processing {image_path.name} with {model}")
    
    # Encode image
    base64_image = encode_image(image_path)
    if not base64_image:
        return None
    
    # Prepare prompt for French/English PII extraction
    prompt = """You are an expert OCR system specialized in extracting PII (Personally Identifiable Information) from documents.

CRITICAL REQUIREMENTS:
- Extract ALL text from this image with perfect accuracy
- Pay special attention to French text and preserve ALL accents (é, è, à, ç, ô, etc.)
- Focus on identifying PII data such as:
  * Names (Nom, Prénom)
  * Birth dates (Date de naissance)
  * Addresses (Adresse)
  * Phone numbers (Téléphone)
  * Medical information
  * Employee information
  * Any personal data

EXTRACT EXACTLY as written, including:
- All French accents and special characters
- Dates in any format (DD/MM/YYYY, YYYY-MM-DD, etc.)
- Phone numbers with area codes
- Complete addresses
- Full names with proper capitalization

Return ONLY the extracted text, preserving original formatting and line breaks. Do NOT interpret or translate - extract literally.

Text to extract:"""

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
        
        # Extract PII patterns from the text
        pii_found = extract_pii_from_text(extracted_text)
        
        return {
            "model": model,
            "file": str(image_path),
            "success": True,
            "extracted_text": extracted_text,
            "pii_entities": pii_found,
            "text_length": len(extracted_text),
            "usage": {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
    except Exception as e:
        return {
            "model": model,
            "file": str(image_path),
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def extract_pii_from_text(text):
    """Extract PII patterns from extracted text."""
    import re
    
    pii_patterns = {
        "names": [
            r"(?:Nom|Prénom|Name)(?:\s*[:\-,]?\s*)([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+)*)",
            r"\b([A-ZÀ-Ÿ][a-zà-ÿ]+),\s*([A-ZÀ-Ÿ][a-zà-ÿ]+)\b"
        ],
        "birth_dates": [
            r"(?:Date de naissance|Birth date|Born)(?:\s*[:\-]?\s*)(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})",
            r"(?:naissance|birth)(?:\s*[:\-]?\s*)(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})",
            r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})\b",
            r"\b(\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b"
        ],
        "phone_numbers": [
            r"(?:Téléphone|Phone|Tel)(?:\s*[:\-]?\s*)(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})",
            r"\b(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})\b"
        ],
        "addresses": [
            r"(?:Adresse|Address)(?:\s*[:\-]?\s*)([\d\w\s,.-]+(?:Street|St|Avenue|Ave|Road|Rd|rue|avenue|boulevard))",
            r"\b(\d+\s+[A-Za-zÀ-ÿ\s]+(?:Street|St|Avenue|Ave|Road|Rd|rue|avenue|boulevard|place))\b"
        ],
        "medical_info": [
            r"(?:Médecin|Doctor|Dr)(?:\s*[:\-]?\s*)([A-ZÀ-Ÿ][a-zà-ÿ\s]+)",
            r"(?:Patient|Nom du patient)(?:\s*[:\-]?\s*)([A-ZÀ-Ÿ][a-zà-ÿ\s]+)"
        ]
    }
    
    found_pii = {}
    
    for category, patterns in pii_patterns.items():
        found_pii[category] = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(1 if match.groups() else 0):
                    entity = match.group(1 if match.groups() else 0).strip()
                    if entity and len(entity) > 1:
                        found_pii[category].append({
                            "text": entity,
                            "start": match.start(),
                            "end": match.end(),
                            "pattern": pattern
                        })
    
    # Remove empty categories
    return {k: v for k, v in found_pii.items() if v}

def process_pdf_with_llm(pdf_path, password="Hubert"):
    """Process PDF by converting to images first."""
    try:
        from pdf2image import convert_from_path
        
        print(f"\n🔍 Converting PDF {pdf_path.name} to images...")
        
        # Convert PDF to images
        pages = convert_from_path(
            str(pdf_path),
            dpi=200,
            fmt='RGB',
            userpw=password if password else None
        )
        
        results = []
        for i, page in enumerate(pages[:3]):  # Process first 3 pages max
            print(f"   Processing page {i+1}/{len(pages)}")
            
            # Save page as temporary image
            temp_path = Path(f"/tmp/page_{i+1}.png")
            page.save(temp_path, format='PNG')
            
            # Process with LLM
            result = process_with_llm(temp_path, "gpt-4o-mini")
            if result:
                result["page_number"] = i + 1
                results.append(result)
            
            # Clean up
            temp_path.unlink(missing_ok=True)
        
        return results
        
    except ImportError:
        print("❌ pdf2image not available. Install with: pip install pdf2image")
        return None
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return None

def main():
    """Main test function."""
    print("🚀 Standalone LLM OCR Test")
    print("=" * 50)
    
    data_dir = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Test models
    models_to_test = ["gpt-4o-mini", "gpt-4o"]
    
    # Find processable files
    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.png"))
    pdf_files = list(data_dir.glob("*.pdf"))
    
    all_results = []
    
    # Test image files
    for image_file in image_files[:2]:  # Test first 2 images
        print(f"\n📷 Testing image file: {image_file.name}")
        for model in models_to_test:
            result = process_with_llm(image_file, model)
            if result:
                all_results.append(result)
    
    # Test PDF files (first page only)
    for pdf_file in pdf_files[:2]:  # Test first 2 PDFs
        print(f"\n📄 Testing PDF file: {pdf_file.name}")
        pdf_results = process_pdf_with_llm(pdf_file)
        if pdf_results:
            all_results.extend(pdf_results)
    
    # Save results to JSON
    output_file = Path("llm_ocr_test_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 30)
    
    successful_extractions = [r for r in all_results if r.get("success")]
    failed_extractions = [r for r in all_results if not r.get("success")]
    
    print(f"✅ Successful extractions: {len(successful_extractions)}")
    print(f"❌ Failed extractions: {len(failed_extractions)}")
    
    # Show PII findings summary
    total_pii_found = 0
    for result in successful_extractions:
        pii_count = sum(len(entities) for entities in result.get("pii_entities", {}).values())
        total_pii_found += pii_count
        
        if pii_count > 0:
            print(f"\n🎯 {result['file']} ({result['model']}):")
            print(f"   PII entities found: {pii_count}")
            print(f"   Text length: {result.get('text_length', 0)} characters")
            
            # Show found PII
            for category, entities in result.get("pii_entities", {}).items():
                if entities:
                    print(f"   {category}: {[e['text'] for e in entities[:3]]}")  # Show first 3
    
    print(f"\n🎯 Total PII entities found: {total_pii_found}")
    print(f"📝 Full results saved to: {output_file}")
    
    # Show sample extraction
    if successful_extractions:
        sample = successful_extractions[0]
        print(f"\n📄 Sample extraction from {Path(sample['file']).name}:")
        print("-" * 40)
        text_preview = sample.get('extracted_text', '')[:500]
        print(text_preview + "..." if len(sample.get('extracted_text', '')) > 500 else text_preview)

if __name__ == "__main__":
    main()