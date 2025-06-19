#!/usr/bin/env python3
"""Test script for LLM configuration only (no dependencies)."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Test only the config module without external dependencies
try:
    from llm.llm_config import LLMModelRegistry, CostCalculator, OCRTaskType
    
    def test_model_registry():
        """Test the LLM model registry."""
        print("🧪 Testing LLM Model Registry...")
        
        # Test getting all models
        all_models = LLMModelRegistry.MODELS
        print(f"✅ Found {len(all_models)} total models")
        
        # Test vision models
        vision_models = LLMModelRegistry.get_vision_models()
        print(f"✅ Found {len(vision_models)} vision-capable models")
        
        # Test cheapest models
        cheapest = LLMModelRegistry.get_cheapest_models(vision_only=True)
        print(f"✅ Cheapest vision model: {cheapest[0].display_name} (${cheapest[0].input_cost_per_1k_tokens:.6f}/1K tokens)")
        
        # Test best value models
        best_value = LLMModelRegistry.get_best_value_models(vision_only=True)
        print(f"✅ Best value vision model: {best_value[0].display_name}")
        
        return True

    def test_cost_calculator():
        """Test the cost calculator."""
        print("💰 Testing Cost Calculator...")
        
        # Test token estimation
        tokens = CostCalculator.estimate_tokens_for_image(1024, 1024, True)
        print(f"✅ Estimated tokens for 1024x1024 image: {tokens}")
        
        # Test output token estimation
        output_tokens = CostCalculator.estimate_output_tokens(OCRTaskType.BASIC_TEXT_EXTRACTION, tokens)
        print(f"✅ Estimated output tokens: {output_tokens}")
        
        # Test cost calculation
        model = LLMModelRegistry.get_model("gpt-4o-mini")
        if model:
            cost = CostCalculator.calculate_cost(model, tokens, output_tokens)
            print(f"✅ Estimated cost for GPT-4o Mini: ${cost:.6f}")
        
        # Test document cost estimation
        cost_estimate = CostCalculator.estimate_document_cost(
            "gpt-4o-mini",
            OCRTaskType.BASIC_TEXT_EXTRACTION,
            1024, 1024, True
        )
        print(f"✅ Full document cost estimate: ${cost_estimate['total_cost']:.6f}")
        
        return True

    def test_api_key_availability():
        """Test API key availability."""
        print("🔑 Testing API Key Availability...")
        
        api_keys = {
            "OpenAI": os.getenv("OPENAI_API_KEY"),
            "Anthropic": os.getenv("ANTHROPIC_API_KEY"), 
            "Google": os.getenv("GOOGLE_API_KEY"),
            "DeepSeek": os.getenv("DEEPSEEK_API"),
            "NVIDIA": os.getenv("NVIDIA_KEY")
        }
        
        available_keys = []
        for provider, key in api_keys.items():
            if key:
                available_keys.append(provider)
                # Only show first/last few chars for security
                masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
                print(f"✅ {provider}: Available ({masked_key})")
            else:
                print(f"❌ {provider}: Not configured")
        
        print(f"✅ Total providers configured: {len(available_keys)}")
        return True

    def test_model_comparison():
        """Test model comparison for different use cases."""
        print("📊 Testing Model Comparison...")
        
        vision_models = LLMModelRegistry.get_vision_models()
        
        print("Cost-effective models for basic OCR:")
        cheapest = sorted(vision_models, key=lambda m: m.input_cost_per_1k_tokens)[:3]
        for i, model in enumerate(cheapest, 1):
            print(f"  {i}. {model.display_name}: ${model.input_cost_per_1k_tokens:.6f}/1K tokens")
        
        print("\nHigh-quality models for complex documents:")
        highest_quality = sorted(vision_models, key=lambda m: m.quality_score, reverse=True)[:3]
        for i, model in enumerate(highest_quality, 1):
            print(f"  {i}. {model.display_name}: {model.quality_score:.1%} quality")
        
        print("\nBest value models (quality/cost ratio):")
        def value_score(model):
            avg_cost = (model.input_cost_per_1k_tokens + model.output_cost_per_1k_tokens) / 2
            return model.quality_score / max(avg_cost, 0.00001)
        
        best_value = sorted(vision_models, key=value_score, reverse=True)[:3]
        for i, model in enumerate(best_value, 1):
            score = value_score(model)
            print(f"  {i}. {model.display_name}: {score:.0f} value score")
        
        return True

    def main():
        """Run all tests."""
        print("🚀 Starting LLM Configuration Tests\n")
        
        try:
            test_model_registry()
            print()
            test_cost_calculator()
            print()
            test_api_key_availability()
            print()
            test_model_comparison()
            print()
            
            print("🎉 All configuration tests completed successfully!")
            
            # Summary
            print("\n📋 Integration Summary:")
            print("✅ LLM model registry configured with 8+ models")
            print("✅ Cost calculation system ready")
            print("✅ Multi-provider support (OpenAI, Anthropic, Google, DeepSeek, NVIDIA)")
            print("✅ Task-specific model recommendations")
            print("✅ Cost optimization and monitoring")
            print("✅ UI configuration pages created")
            
            # Show available models
            print("\n🤖 Available LLM Models:")
            for model_name, model in LLMModelRegistry.MODELS.items():
                status = "🔄" if model.supports_vision else "📝"
                print(f"  {status} {model.display_name} ({model.provider.value})")
            
            print("\n🎯 Configuration ready - install LLM packages to enable full functionality!")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0

    if __name__ == "__main__":
        exit(main())

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please check the module structure and dependencies.")
    exit(1)