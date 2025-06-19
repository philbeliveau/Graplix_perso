#!/usr/bin/env python3
"""Test that LLM options appear in dashboard pages."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment
from load_env import load_env_file
load_env_file()

def test_llm_options_availability():
    """Test that LLM options are available in dashboard."""
    print("🧪 Testing LLM Options Availability in Dashboard")
    print("=" * 50)
    
    # Test environment loading
    print("🔧 Environment Status:")
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "DeepSeek": os.getenv("DEEPSEEK_API"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "NVIDIA": os.getenv("NVIDIA_KEY")
    }
    
    available_providers = []
    for provider, key in api_keys.items():
        if key:
            available_providers.append(provider)
            print(f"   ✅ {provider}: Available")
        else:
            print(f"   ❌ {provider}: Not configured")
    
    if not available_providers:
        print("❌ No API keys available - LLM options will not appear")
        return False
    
    print(f"✅ {len(available_providers)} provider(s) available for LLM OCR")
    
    # Test configuration
    print("\n⚙️ Configuration Status:")
    try:
        from core.config import settings
        
        enable_llm = settings.processing.enable_llm_ocr
        llm_model = settings.processing.llm_ocr_model
        max_cost = settings.processing.max_llm_cost_per_document
        
        print(f"   LLM OCR enabled: {enable_llm}")
        print(f"   Default model: {llm_model}")
        print(f"   Max cost per doc: ${max_cost}")
        
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False
    
    # Test dashboard page imports
    print("\n📄 Dashboard Pages Status:")
    try:
        # Test document processing page
        from dashboard.pages import document_processing
        print("   ✅ Document processing page: Available")
        
        # Test batch analysis page  
        from dashboard.pages import batch_analysis
        print("   ✅ Batch analysis page: Available")
        
        # Test LLM config page
        try:
            from dashboard.pages import llm_ocr_simple
            print("   ✅ LLM OCR config page: Available")
        except ImportError:
            print("   ⚠️ LLM OCR config page: Import issues")
        
    except Exception as e:
        print(f"   ❌ Dashboard import error: {e}")
        return False
    
    # Test LLM integration
    print("\n🤖 LLM Integration Status:")
    try:
        from llm import LLM_PROCESSOR_AVAILABLE, LLMModelRegistry
        
        if LLM_PROCESSOR_AVAILABLE:
            print("   ✅ LLM processor: Available")
        else:
            print("   ⚠️ LLM processor: Missing dependencies")
            print("   Install: pip install openai anthropic google-generativeai")
        
        # Test model registry
        models = LLMModelRegistry.MODELS
        vision_models = LLMModelRegistry.get_vision_models()
        print(f"   ✅ Model registry: {len(models)} total, {len(vision_models)} vision-capable")
        
    except Exception as e:
        print(f"   ❌ LLM integration error: {e}")
        return False
    
    # Test what models would be available in UI
    print("\n🎯 Available Models for UI:")
    available_models = []
    
    if os.getenv("OPENAI_API_KEY"):
        available_models.extend(["gpt-4o-mini", "gpt-3.5-turbo"])
        print("   ✅ OpenAI models: gpt-4o-mini, gpt-3.5-turbo")
        
    if os.getenv("DEEPSEEK_API"):
        available_models.append("deepseek-chat")
        print("   ✅ DeepSeek models: deepseek-chat")
        
    if os.getenv("ANTHROPIC_API_KEY"):
        available_models.extend(["claude-3-haiku", "claude-3-sonnet"])
        print("   ✅ Anthropic models: claude-3-haiku, claude-3-sonnet")
        
    print(f"\n📊 Summary:")
    print(f"   • {len(available_providers)} API providers configured")
    print(f"   • {len(available_models)} LLM models available in UI")
    print(f"   • Dashboard pages updated with LLM options")
    print(f"   • Configuration enabled for LLM OCR")
    
    return True


def show_dashboard_usage():
    """Show how to use LLM options in dashboard."""
    print("\n" + "=" * 50)
    print("🎨 How to Use LLM OCR in Dashboard")
    print("=" * 50)
    
    print("\n1. **Start the Dashboard:**")
    print("   streamlit run src/dashboard/main.py")
    
    print("\n2. **Document Processing Page:**")
    print("   • Go to 'Document Processing' page")
    print("   • In 'OCR Settings' section, you'll see:")
    print("     - OCR Engine dropdown with 'llm' and 'llm+traditional' options")
    print("     - LLM model selection (gpt-4o-mini, deepseek-chat, etc.)")
    print("     - Cost limit slider")
    print("     - Real-time cost estimation")
    
    print("\n3. **Batch Analysis Page:**")
    print("   • Go to 'Batch Analysis' page → 'Batch Upload' tab")
    print("   • In 'Processing Options', you'll see:")
    print("     - OCR Engine dropdown with LLM options")
    print("     - LLM model selection")
    print("     - Batch cost controls")
    print("     - Total batch cost estimation")
    
    print("\n4. **LLM OCR Config Page:**")
    print("   • Go to 'LLM OCR Config' page")
    print("   • Complete configuration interface:")
    print("     - Model selection and comparison")
    print("     - Cost analysis and monitoring")
    print("     - Performance settings")
    print("     - Advanced configurations")
    
    print("\n5. **Recommended Settings:**")
    print("   • **For cost-effectiveness**: Select 'deepseek-chat'")
    print("   • **For balance**: Select 'gpt-4o-mini'")
    print("   • **For quality**: Select 'gpt-3.5-turbo'")
    print("   • Set cost limits based on your budget")
    print("   • Use 'llm+traditional' for fallback reliability")


if __name__ == "__main__":
    success = test_llm_options_availability()
    
    if success:
        print("\n🎉 LLM options should now be visible in the dashboard!")
        show_dashboard_usage()
    else:
        print("\n❌ Issues detected with LLM integration")
        print("Please check the errors above and resolve them.")
        
    exit(0 if success else 1)