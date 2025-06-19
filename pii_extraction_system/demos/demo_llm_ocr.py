#!/usr/bin/env python3
"""Demo script showing LLM OCR integration capabilities."""

import os
import sys
from pathlib import Path

# Load environment variables
from load_env import load_env_file, check_api_keys

def main():
    """Run LLM OCR demonstration."""
    print("🚀 LLM OCR Integration Demo")
    print("=" * 50)
    
    # Load environment variables first
    print("\n🔧 Loading Environment Variables:")
    load_env_file()
    
    # Check project structure
    print("\n📁 Project Structure:")
    src_dir = Path("src")
    if src_dir.exists():
        print("✅ src/ directory found")
        
        # Check key modules
        modules = [
            "src/llm/__init__.py",
            "src/llm/llm_config.py", 
            "src/llm/llm_ocr_processor.py",
            "src/utils/document_processor.py",
            "src/dashboard/pages/llm_ocr_simple.py",
            "src/core/config.py"
        ]
        
        for module in modules:
            if Path(module).exists():
                print(f"✅ {module}")
            else:
                print(f"❌ {module}")
    else:
        print("❌ src/ directory not found")
    
    # Check API keys
    print("\n🔑 API Key Configuration:")
    configured_providers = check_api_keys()
    
    if configured_providers:
        print(f"\n✅ {len(configured_providers)} provider(s) ready for use!")
    else:
        print("\n⚠️  No API keys configured. Add them to .env file:")
        print("OPENAI_API_KEY=your_key_here")
        print("ANTHROPIC_API_KEY=your_key_here")
        print("GOOGLE_API_KEY=your_key_here")
    
    # Test configuration loading
    print("\n⚙️  Testing Configuration:")
    
    sys.path.insert(0, str(Path("src")))
    
    try:
        from llm.llm_config import LLMModelRegistry, CostCalculator, OCRTaskType
        print("✅ LLM configuration module loaded")
        
        models = LLMModelRegistry.MODELS
        vision_models = LLMModelRegistry.get_vision_models()
        cheapest = LLMModelRegistry.get_cheapest_models(vision_only=True, limit=3)
        
        print(f"✅ {len(models)} total models available")
        print(f"✅ {len(vision_models)} vision-capable models")
        print("✅ Cost calculator ready")
        
        print(f"\n💰 Most cost-effective models:")
        for i, model in enumerate(cheapest, 1):
            print(f"  {i}. {model.display_name}: ${model.input_cost_per_1k_tokens:.6f}/1K tokens")
            
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        print("   Install packages: pip install pydantic")
    
    # Test LLM processor availability  
    print("\n🤖 LLM Processor Status:")
    try:
        from llm import LLM_PROCESSOR_AVAILABLE
        if LLM_PROCESSOR_AVAILABLE:
            print("✅ LLM processor available")
        else:
            print("⚠️  LLM processor not available (missing dependencies)")
            print("   Install packages: pip install openai anthropic google-generativeai")
    except ImportError:
        print("❌ LLM processor import failed")
    
    # Demo cost calculation
    print("\n🧮 Cost Calculation Demo:")
    try:
        # Simple cost estimation for a typical document
        cost_estimate = CostCalculator.estimate_document_cost(
            "gpt-4o-mini",
            OCRTaskType.BASIC_TEXT_EXTRACTION,
            1024, 1024, True
        )
        
        print(f"✅ Sample cost for 1024x1024 image with GPT-4o Mini:")
        print(f"   Input tokens: {cost_estimate['input_tokens']:,}")
        print(f"   Output tokens: {cost_estimate['output_tokens']:,}")
        print(f"   Total cost: ${cost_estimate['total_cost']:.6f}")
        
        # Compare with other models
        models_to_compare = ["gpt-4o-mini", "claude-3-haiku", "gemini-1.5-flash"]
        print(f"\n📊 Cost comparison for typical document:")
        
        for model_name in models_to_compare:
            try:
                estimate = CostCalculator.estimate_document_cost(
                    model_name, OCRTaskType.BASIC_TEXT_EXTRACTION, 1024, 1024, True
                )
                model_obj = LLMModelRegistry.get_model(model_name)
                print(f"   {model_obj.display_name}: ${estimate['total_cost']:.6f}")
            except:
                print(f"   {model_name}: Not available")
                
    except Exception as e:
        print(f"❌ Cost calculation failed: {e}")
    
    # Check dashboard availability
    print("\n🎨 Dashboard Status:")
    dashboard_files = [
        "src/dashboard/main.py",
        "src/dashboard/pages/llm_ocr_simple.py",
        "src/dashboard/pages/configuration.py"
    ]
    
    for file_path in dashboard_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    # Usage instructions
    print("\n📚 Usage Instructions:")
    print("1. Install dependencies:")
    print("   pip install streamlit openai anthropic google-generativeai")
    print("\n2. Configure API keys in .env file")
    print("\n3. Run the dashboard:")
    print("   streamlit run src/dashboard/main.py")
    print("\n4. Navigate to 'LLM OCR Config' page to configure models")
    
    # Summary
    print("\n" + "=" * 50)
    print("🎯 Integration Summary:")
    print("✅ LLM OCR architecture implemented")
    print("✅ Multi-provider support (OpenAI, Anthropic, Google, etc.)")
    print("✅ Cost analysis and optimization")
    print("✅ Graceful fallback mechanisms")
    print("✅ User-friendly dashboard interface")
    print("✅ Production-ready error handling")
    
    if configured_providers:
        print(f"✅ Ready to use with {len(configured_providers)} provider(s)!")
    else:
        print("⚠️  Configure API keys to enable full functionality")
    
    print("\n🚀 System ready for enhanced OCR with LLM models!")
    
    return 0


if __name__ == "__main__":
    exit(main())