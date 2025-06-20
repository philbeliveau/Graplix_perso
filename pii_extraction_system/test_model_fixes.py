#!/usr/bin/env python3
"""
Test script to verify the LLM model availability fixes
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

def test_model_availability():
    """Test model availability with the new fixes"""
    print("🔧 Testing Model Availability Fixes")
    print("=" * 60)
    
    try:
        from llm.multimodal_llm_service import llm_service
        
        # Test debug functionality
        print("\n📊 Debug Information:")
        debug_info = llm_service.debug_model_availability()
        
        print(f"Total providers: {debug_info['total_providers_initialized']}")
        print("Available models by provider:")
        for provider, models in debug_info["provider_breakdown"].items():
            print(f"  {provider}: {models}")
        
        print("\nAPI Key Status:")
        for key, status in debug_info["api_key_status"].items():
            print(f"  {key}: {'✅ Set' if status else '❌ Not set'}")
        
        # Test model key normalization
        print("\n🔍 Testing Model Key Normalization:")
        test_models = [
            "gpt-4o-mini",           # Should map to openai/gpt-4o-mini
            "openai/gpt-4o",         # Should work as-is
            "deepseek-chat",         # Should map to deepseek/deepseek-chat
            "claude-3-5-sonnet-20241022",  # Should map to anthropic/...
            "invalid-model"          # Should fail with suggestion
        ]
        
        for model in test_models:
            print(f"\nTesting model: '{model}'")
            result = llm_service.test_model_access(model)
            print(f"  Normalized: {result['normalized_key']}")
            print(f"  Available: {'✅' if result['available'] else '❌'}")
            if result['error']:
                print(f"  Error: {result['error']}")
            if result['suggestions'] and result['suggestions'][0]:
                print(f"  Suggestion: {result['suggestions'][0]}")
        
        # Test actual extraction with different model key formats
        print("\n🧪 Testing PII Extraction with Different Model Keys:")
        
        # Create simple test text
        test_text = "John Doe lives at 123 Main St and his email is john@example.com"
        
        test_cases = [
            "gpt-4o-mini",           # Without provider prefix
            "openai/gpt-4o-mini",    # With provider prefix
        ]
        
        for model_key in test_cases:
            print(f"\nTesting extraction with model key: '{model_key}'")
            try:
                result = llm_service.extract_pii_from_text(
                    text=test_text,
                    model_key=model_key,
                    document_type="test"
                )
                
                if result.get('success'):
                    print("  ✅ Extraction successful")
                    entities = result.get('pii_entities', [])
                    print(f"  Found {len(entities)} entities")
                else:
                    print(f"  ❌ Extraction failed: {result.get('error')}")
                    if result.get('suggestion'):
                        print(f"  💡 Suggestion: {result.get('suggestion')}")
                    
            except Exception as e:
                print(f"  ❌ Exception: {e}")
        
        # Test with DeepSeek if available
        print("\n🤖 Testing DeepSeek Integration:")
        deepseek_result = llm_service.test_model_access("deepseek-chat")
        if deepseek_result['available']:
            print("  ✅ DeepSeek models available")
            # Test text extraction (DeepSeek doesn't support vision)
            try:
                result = llm_service.extract_pii_from_text(
                    text=test_text,
                    model_key="deepseek-chat",
                    document_type="test"
                )
                if result.get('success'):
                    print("  ✅ DeepSeek text extraction successful")
                else:
                    print(f"  ⚠️ DeepSeek extraction issue: {result.get('error')}")
            except Exception as e:
                print(f"  ❌ DeepSeek exception: {e}")
        else:
            print("  ❌ DeepSeek models not available (API key not configured)")
            print(f"  Error: {deepseek_result.get('error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_key_configuration():
    """Test API key configuration consistency"""
    print("\n🔑 API Key Configuration Test:")
    
    # Check for common API key environment variables
    api_keys_to_check = [
        ('OPENAI_API_KEY', 'OpenAI'),
        ('ANTHROPIC_API_KEY', 'Anthropic'),
        ('GOOGLE_API_KEY', 'Google'),
        ('MISTRAL_API_KEY', 'Mistral'),
        ('DEEPSEEK_API_KEY', 'DeepSeek (preferred)'),
        ('DEEPSEEK_API', 'DeepSeek (alternative)'),
    ]
    
    for env_var, description in api_keys_to_check:
        value = os.getenv(env_var)
        if value:
            print(f"  ✅ {description}: Set (length: {len(value)})")
        else:
            print(f"  ❌ {description}: Not set")
    
    # Provide configuration guidance
    print("\n💡 Configuration Notes:")
    print("  - For DeepSeek: Set either DEEPSEEK_API_KEY or DEEPSEEK_API")
    print("  - DeepSeek uses OpenAI-compatible API at https://api.deepseek.com/v1")
    print("  - Models without API keys will be skipped during initialization")

def main():
    """Run all tests"""
    print("🚀 LLM Model Fixes Validation")
    print("=" * 60)
    
    success = True
    
    # Test API key configuration
    test_api_key_configuration()
    
    # Test model availability
    if not test_model_availability():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests completed! Check output above for any issues.")
    else:
        print("⚠️  Some tests had issues. Review the output above.")
    
    print("\n📋 Summary of Fixes:")
    print("✅ Added DeepSeek provider to MultimodalLLMService")
    print("✅ Added model key normalization (gpt-4o-mini → openai/gpt-4o-mini)")
    print("✅ Added better error messages with suggestions")
    print("✅ Added comprehensive debugging methods")
    print("✅ Added fallback handling for unavailable models")

if __name__ == "__main__":
    main()