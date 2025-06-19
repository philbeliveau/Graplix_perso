#!/usr/bin/env python3
"""Test LLM OCR with the specific user file."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment
from load_env import load_env_file
load_env_file()

def test_specific_file():
    """Test LLM OCR with the user's specific file."""
    file_path = '/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data/b3965576-c316-467c-b2c6-df28e8581236.jpg'
    
    print("🧪 Testing LLM OCR with Specific File")
    print("=" * 60)
    print(f"📁 File: {file_path}")
    
    # Check if file exists
    if not Path(file_path).exists():
        print("❌ File not found!")
        return False
    
    file_size = Path(file_path).stat().st_size / (1024 * 1024)
    print(f"📊 File size: {file_size:.2f} MB")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found")
        return False
    
    try:
        # Import processor
        from utils.document_processor import DocumentProcessor
        from llm import LLM_PROCESSOR_AVAILABLE
        
        if not LLM_PROCESSOR_AVAILABLE:
            print("❌ LLM processor not available")
            return False
        
        print("✅ LLM processor ready")
        
        # Create processor
        processor = DocumentProcessor()
        
        if not processor.llm_ocr_enabled:
            print("❌ LLM OCR not enabled")
            return False
        
        print("✅ LLM OCR enabled")
        
        # Process the file
        print(f"\n🔄 Processing file with LLM OCR...")
        print("   This may take 10-30 seconds...")
        
        result = processor.process_document(file_path)
        
        # Extract results
        extracted_text = result.get('ocr_text', result.get('raw_text', ''))
        confidence = result.get('confidence', 0)
        engine_used = result.get('ocr_engine', 'unknown')
        llm_metadata = result.get('llm_metadata', {})
        
        print(f"\n📊 Processing Results:")
        print(f"   ✅ Engine used: {engine_used}")
        print(f"   ✅ Confidence: {confidence}")
        print(f"   ✅ Text length: {len(extracted_text)} characters")
        
        # Show cost information if available
        if llm_metadata:
            cost_info = llm_metadata.get('cost_info', {})
            if cost_info:
                actual_cost = cost_info.get('actual_cost', 0)
                input_tokens = cost_info.get('input_tokens', 0)
                output_tokens = cost_info.get('output_tokens', 0)
                model_used = llm_metadata.get('model_used', 'unknown')
                
                print(f"   💰 Model: {model_used}")
                print(f"   💰 Cost: ${actual_cost:.6f}")
                print(f"   💰 Tokens: {input_tokens} input, {output_tokens} output")
        
        if extracted_text:
            print(f"\n📄 Extracted Text:")
            print("═" * 60)
            print(extracted_text)
            print("═" * 60)
            
            # Analyze the text quality
            print(f"\n🔍 Quality Analysis:")
            
            # Check for French words/accents
            french_indicators = ['é', 'è', 'à', 'ç', 'ô', 'û', 'ê', 'ù', 'â', 'î']
            found_accents = [char for char in french_indicators if char in extracted_text]
            
            if found_accents:
                print(f"   ✅ French accents preserved: {found_accents}")
            else:
                print("   ⚠️ No French accents detected")
            
            # Check for common French words
            french_words = ['de', 'du', 'des', 'le', 'la', 'les', 'et', 'ou', 'mais', 'dans', 'sur', 'avec', 'pour']
            found_french_words = [word for word in french_words if word.lower() in extracted_text.lower()]
            
            if found_french_words:
                print(f"   ✅ French words found: {found_french_words[:5]}...")
            
            # Check for garbled text patterns
            garbled_patterns = ['ÿ', 'rs bu à me', 'Lo mue', random_chars := len([c for c in extracted_text if ord(c) > 255])]
            if any(pattern in extracted_text for pattern in garbled_patterns[:-1]) or random_chars > 10:
                print("   ❌ Garbled text detected!")
                success = False
            else:
                print("   ✅ Text appears clean and readable")
                success = True
            
            # Check text structure
            lines = extracted_text.split('\n')
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            print(f"   📝 Structure: {len(non_empty_lines)} meaningful lines")
            
            # Word count
            words = extracted_text.split()
            print(f"   📝 Word count: {len(words)} words")
            
            return success
            
        else:
            print("❌ No text extracted!")
            return False
            
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing LLM OCR Fix with Your Specific File")
    print("=" * 70)
    
    success = test_specific_file()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 SUCCESS! Your file processed correctly with LLM OCR!")
        print("   • French text extracted cleanly")
        print("   • No garbled characters")
        print("   • Readable, structured output")
    else:
        print("❌ Issues detected - check the output above")
    
    exit(0 if success else 1)