#!/usr/bin/env python3
"""
Debug script to test LLM service functionality
Run this to diagnose issues with the multimodal LLM service
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
from load_env import load_env_file
load_env_file()

def test_environment():
    """Test environment variables"""
    print("🔍 Testing Environment Variables:")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')
    google_key = os.getenv('GOOGLE_API_KEY')
    
    print(f"  OPENAI_API_KEY: {'✅ Set' if openai_key else '❌ Not set'}")
    print(f"  ANTHROPIC_API_KEY: {'✅ Set' if anthropic_key else '❌ Not set'}")
    print(f"  GOOGLE_API_KEY: {'✅ Set' if google_key else '❌ Not set'}")
    
    if not any([openai_key, anthropic_key, google_key]):
        print("❌ No API keys found! Please set at least one API key.")
        return False
    
    return True

def test_imports():
    """Test required imports"""
    print("\n📦 Testing Required Imports:")
    
    try:
        import openai
        print("  openai: ✅")
    except ImportError:
        print("  openai: ❌ (pip install openai)")
    
    try:
        import anthropic
        print("  anthropic: ✅")
    except ImportError:
        print("  anthropic: ❌ (pip install anthropic)")
    
    try:
        import google.generativeai
        print("  google.generativeai: ✅")
    except ImportError:
        print("  google.generativeai: ❌ (pip install google-generativeai)")
    
    try:
        from PIL import Image
        print("  PIL: ✅")
    except ImportError:
        print("  PIL: ❌ (pip install Pillow)")
    
    try:
        from pdf2image import convert_from_bytes
        print("  pdf2image: ✅")
    except ImportError:
        print("  pdf2image: ❌ (pip install pdf2image)")

def test_llm_service():
    """Test LLM service initialization"""
    print("\n🤖 Testing LLM Service:")
    
    try:
        from llm.multimodal_llm_service import llm_service
        print("  LLM service import: ✅")
        
        available_models = llm_service.get_available_models()
        print(f"  Available models: {len(available_models)}")
        
        for model in available_models:
            model_info = llm_service.get_model_info(model)
            print(f"    - {model}: {'✅' if model_info.get('available') else '❌'}")
        
        if not available_models:
            print("  ❌ No models available!")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ LLM service failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_extraction():
    """Test simple PII extraction"""
    print("\n🧪 Testing Simple PII Extraction:")
    
    try:
        from llm.multimodal_llm_service import llm_service
        
        # Create a simple test image (white background with text)
        from PIL import Image, ImageDraw, ImageFont
        import base64
        import io
        
        # Create test image
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add test text
        test_text = "John Doe\nEmail: john.doe@example.com\nPhone: (555) 123-4567"
        draw.text((20, 50), test_text, fill='black')
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        print("  Created test image ✅")
        
        # Get first available model
        available_models = llm_service.get_available_models()
        if not available_models:
            print("  ❌ No models available for testing")
            return False
        
        test_model = available_models[0]
        print(f"  Testing with model: {test_model}")
        
        # Test extraction
        result = llm_service.extract_pii_from_image(
            image_data=image_data,
            model_key=test_model,
            document_type="test"
        )
        
        print(f"  Extraction result: {'✅ Success' if result.get('success') else '❌ Failed'}")
        
        if result.get('success'):
            entities = result.get('pii_entities', [])
            print(f"  Entities found: {len(entities)}")
            for entity in entities[:3]:  # Show first 3
                print(f"    - {entity.get('type')}: {entity.get('text')}")
        else:
            print(f"  Error: {result.get('error')}")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"  ❌ Extraction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔧 LLM Service Debug Script")
    print("=" * 50)
    
    tests = [
        test_environment,
        test_imports,
        test_llm_service,
        test_simple_extraction
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"  Environment: {'✅' if results[0] else '❌'}")
    print(f"  Imports: {'✅' if len(results) > 1 else '❌'}")
    print(f"  LLM Service: {'✅' if len(results) > 2 and results[2] else '❌'}")
    print(f"  Extraction: {'✅' if len(results) > 3 and results[3] else '❌'}")
    
    if all(results):
        print("\n🎉 All tests passed! LLM service should be working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("\n💡 Common fixes:")
        print("  - Install missing packages: pip install openai anthropic google-generativeai Pillow pdf2image")
        print("  - Set API keys in your .env file")
        print("  - Check that your API keys are valid")

if __name__ == "__main__":
    main()