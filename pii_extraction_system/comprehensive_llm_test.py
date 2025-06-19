#!/usr/bin/env python3
"""
Comprehensive LLM test that processes all files in the data folder.
Usage: python comprehensive_llm_test.py YOUR_OPENAI_API_KEY
"""

import os
import sys
import json
import base64
import traceback
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any
import time

# Get API key from command line
if len(sys.argv) < 2:
    print("❌ Usage: python comprehensive_llm_test.py YOUR_OPENAI_API_KEY")
    sys.exit(1)

OPENAI_API_KEY = sys.argv[1]
PASSWORD = "Hubert"  # For password-protected documents

try:
    import openai
    from PIL import Image
    print("✅ Libraries loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Install: pip install openai pillow pdf2image")
    sys.exit(1)

# Initialize client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def encode_image(image_path):
    """Encode image to base64."""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large
        if max(image.size) > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        print(f"❌ Error encoding {image_path}: {e}")
        return None

def process_pdf_to_images(pdf_path):
    """Convert PDF to images for LLM processing."""
    try:
        from pdf2image import convert_from_path
        
        # Try with password first
        try:
            pages = convert_from_path(
                str(pdf_path),
                dpi=200,
                fmt='RGB',
                userpw=PASSWORD,
                first_page=1,
                last_page=3  # Limit to first 3 pages
            )
        except:
            # Try without password
            pages = convert_from_path(
                str(pdf_path),
                dpi=200,
                fmt='RGB',
                first_page=1,
                last_page=3
            )
        
        return pages
    except ImportError:
        print("   ⚠️ pdf2image not available - skipping PDF")
        return None
    except Exception as e:
        print(f"   ❌ PDF conversion error: {e}")
        return None

def process_with_llm(file_path, content_type="image", image_data=None, model="gpt-4o-mini"):
    """Process file with LLM OCR."""
    
    # Generate appropriate prompt based on file type
    if "formulaire" in str(file_path).lower() or "form" in str(file_path).lower():
        prompt = """Extract ALL text from this form document. Focus on:
- Names and personal information
- Dates and numbers
- Form fields and their values
- Preserve all French accents (é, è, à, ç, etc.)
Extract exactly as written, maintaining original formatting."""
        
    elif "cv" in str(file_path).lower() or "curriculum" in str(file_path).lower():
        prompt = """Extract ALL text from this CV/resume. Focus on:
- Personal information (name, contact details)
- Work experience and dates
- Education information
- Skills and qualifications
- Preserve all French accents and formatting
Extract exactly as written."""
        
    elif "cheque" in str(file_path).lower():
        prompt = """Extract ALL text from this cheque/check. Focus on:
- Names and addresses
- Account numbers and routing numbers
- Amounts and dates
- Bank information
- Any handwritten or printed text
Extract exactly as written, preserving all details."""
        
    else:
        prompt = """Extract ALL text from this document with perfect accuracy.
- Preserve all French accents (é, è, à, ç, ô, etc.)
- Extract names, dates, addresses, phone numbers
- Maintain original formatting and line breaks
- Include all visible text, even form templates
Extract exactly as written, no interpretation."""

    try:
        if content_type == "image" and image_data:
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
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.0
            )
            
            extracted_text = response.choices[0].message.content
            
            # Analyze for PII
            pii_keywords = [
                "nom", "prénom", "name", "adresse", "address", "téléphone", "phone",
                "naissance", "birth", "email", "@", "TREMBLAY", "Steve", "Souad", "Touati"
            ]
            
            found_pii = []
            for keyword in pii_keywords:
                if keyword.lower() in extracted_text.lower():
                    found_pii.append(keyword)
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "text_length": len(extracted_text),
                "pii_keywords_found": found_pii,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def test_all_files():
    """Test LLM OCR on all files in data folder."""
    data_dir = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    print("🔍 Comprehensive LLM OCR Test")
    print("=" * 50)
    print(f"📁 Scanning: {data_dir}")
    
    # Get all files
    all_files = list(data_dir.iterdir())
    processable_files = [f for f in all_files if not f.name.startswith('~')]
    
    print(f"📊 Found {len(processable_files)} processable files")
    
    results = []
    total_cost = 0
    
    for i, file_path in enumerate(processable_files, 1):
        print(f"\n📄 [{i}/{len(processable_files)}] Processing: {file_path.name}")
        
        file_results = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix.lower(),
            "file_size_mb": round(file_path.stat().st_size / (1024*1024), 2),
            "results": {}
        }
        
        try:
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Process image files
                print("   🖼️ Processing as image...")
                image_data = encode_image(file_path)
                if image_data:
                    result = process_with_llm(file_path, "image", image_data, "gpt-4o-mini")
                    file_results["results"]["gpt-4o-mini"] = result
                    
                    if result.get("success"):
                        cost = (result["usage"]["prompt_tokens"] * 0.00015 + 
                               result["usage"]["completion_tokens"] * 0.0006) / 1000
                        total_cost += cost
                        print(f"   ✅ Success: {len(result['extracted_text'])} chars, "
                              f"{len(result['pii_keywords_found'])} PII indicators")
                    else:
                        print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
            elif file_path.suffix.lower() == '.pdf':
                # Process PDF files
                print("   📄 Converting PDF to images...")
                pdf_pages = process_pdf_to_images(file_path)
                if pdf_pages:
                    pdf_results = []
                    for page_num, page in enumerate(pdf_pages, 1):
                        print(f"      Page {page_num}/{len(pdf_pages)}")
                        
                        # Convert PIL page to base64
                        buffer = BytesIO()
                        page.save(buffer, format='PNG')
                        page_data = base64.b64encode(buffer.getvalue()).decode()
                        
                        result = process_with_llm(file_path, "image", page_data, "gpt-4o-mini")
                        if result.get("success"):
                            result["page_number"] = page_num
                            pdf_results.append(result)
                            
                            cost = (result["usage"]["prompt_tokens"] * 0.00015 + 
                                   result["usage"]["completion_tokens"] * 0.0006) / 1000
                            total_cost += cost
                    
                    file_results["results"]["pdf_pages"] = pdf_results
                    total_pii = sum(len(r.get('pii_keywords_found', [])) for r in pdf_results)
                    print(f"   ✅ Processed {len(pdf_results)} pages, {total_pii} total PII indicators")
                else:
                    file_results["results"]["error"] = "PDF conversion failed"
                    print("   ❌ PDF processing failed")
            
            elif file_path.suffix.lower() in ['.txt']:
                # For text files, read directly (no LLM needed for testing)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    file_results["results"]["direct_text"] = {
                        "success": True,
                        "text_length": len(text_content),
                        "preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
                    }
                    print(f"   ✅ Text file: {len(text_content)} characters")
                except Exception as e:
                    file_results["results"]["error"] = str(e)
                    print(f"   ❌ Text file error: {e}")
            
            else:
                # Skip unsupported formats
                file_results["results"]["skipped"] = f"Unsupported format: {file_path.suffix}"
                print(f"   ⏭️ Skipped: {file_path.suffix} format not supported")
        
        except Exception as e:
            file_results["results"]["error"] = str(e)
            print(f"   ❌ Processing error: {e}")
        
        results.append(file_results)
        
        # Add small delay to avoid rate limits
        time.sleep(1)
    
    # Save comprehensive results
    output_file = "comprehensive_llm_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_summary": {
                "total_files": len(processable_files),
                "total_cost_usd": round(total_cost, 4),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "file_results": results
        }, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 40)
    
    successful_files = 0
    total_pii_found = 0
    
    for file_result in results:
        file_name = file_result["file_name"]
        results_data = file_result["results"]
        
        if "gpt-4o-mini" in results_data and results_data["gpt-4o-mini"].get("success"):
            successful_files += 1
            pii_count = len(results_data["gpt-4o-mini"].get("pii_keywords_found", []))
            total_pii_found += pii_count
            print(f"✅ {file_name}: {pii_count} PII indicators")
            
        elif "pdf_pages" in results_data:
            pdf_pages = results_data["pdf_pages"]
            if pdf_pages:
                successful_files += 1
                pii_count = sum(len(page.get("pii_keywords_found", [])) for page in pdf_pages)
                total_pii_found += pii_count
                print(f"✅ {file_name}: {len(pdf_pages)} pages, {pii_count} PII indicators")
            
        elif "direct_text" in results_data:
            successful_files += 1
            print(f"✅ {file_name}: Text file processed")
            
        elif "skipped" in results_data:
            print(f"⏭️ {file_name}: {results_data['skipped']}")
            
        else:
            print(f"❌ {file_name}: Failed")
    
    print(f"\n🎯 Final Results:")
    print(f"   Successfully processed: {successful_files}/{len(processable_files)} files")
    print(f"   Total PII indicators found: {total_pii_found}")
    print(f"   Total API cost: ${total_cost:.4f}")
    print(f"   Results saved to: {output_file}")
    
    if total_pii_found > 10:
        print(f"\n🎉 EXCELLENT: LLMs are successfully extracting PII from your documents!")
        print("   The issue is likely in the Streamlit dashboard integration, not the LLM models.")
    elif total_pii_found > 0:
        print(f"\n✅ PARTIAL SUCCESS: LLMs found some PII but may need optimization.")
    else:
        print(f"\n⚠️ LOW SUCCESS: Check API setup, image quality, or prompt optimization needed.")

if __name__ == "__main__":
    test_all_files()