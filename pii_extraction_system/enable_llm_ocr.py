#!/usr/bin/env python3
"""Enable LLM OCR functionality by updating configuration."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment
from load_env import load_env_file
load_env_file()

def enable_llm_ocr():
    """Enable LLM OCR in the system configuration."""
    try:
        from core.config import settings
        
        print("🔧 Enabling LLM OCR functionality...")
        
        # Enable LLM OCR
        settings.processing.enable_llm_ocr = True
        settings.processing.llm_ocr_model = "gpt-4o-mini"  # Default to best balance
        settings.processing.max_llm_cost_per_document = 0.10
        settings.processing.llm_confidence_threshold = 0.8
        
        print("✅ LLM OCR enabled in configuration")
        print(f"   Primary model: {settings.processing.llm_ocr_model}")
        print(f"   Max cost per document: ${settings.processing.max_llm_cost_per_document}")
        print(f"   Confidence threshold: {settings.processing.llm_confidence_threshold}")
        
        # Check available providers
        print("\n🔑 Available providers:")
        providers = []
        
        if os.getenv("OPENAI_API_KEY"):
            providers.append("OpenAI")
            print("   ✅ OpenAI (GPT models)")
            
        if os.getenv("DEEPSEEK_API"):
            providers.append("DeepSeek")
            print("   ✅ DeepSeek (cost-effective)")
            
        if os.getenv("ANTHROPIC_API_KEY"):
            providers.append("Anthropic")
            print("   ✅ Anthropic (Claude models)")
            
        if os.getenv("NVIDIA_KEY"):
            providers.append("NVIDIA")
            print("   ✅ NVIDIA (specialized models)")
            
        if not providers:
            print("   ❌ No API keys found")
            print("   Add API keys to .env file to use LLM OCR")
            return False
            
        print(f"\n🎯 Ready to use LLM OCR with {len(providers)} provider(s)!")
        return True
        
    except Exception as e:
        print(f"❌ Error enabling LLM OCR: {e}")
        return False


if __name__ == "__main__":
    success = enable_llm_ocr()
    
    if success:
        print("\n🚀 LLM OCR is now enabled!")
        print("\nNext steps:")
        print("1. Run the dashboard: streamlit run src/dashboard/main.py")
        print("2. Go to 'Document Processing' or 'Batch Analysis' pages")
        print("3. Select 'llm' or 'llm+traditional' as OCR engine")
        print("4. Choose your preferred LLM model")
        print("5. Upload documents and see enhanced OCR results!")
    else:
        print("\n❌ Failed to enable LLM OCR")
        print("Check that API keys are configured in .env file")
        
    exit(0 if success else 1)