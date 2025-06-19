#!/usr/bin/env python3
"""Test environment loading from different contexts."""

import os
import sys
from pathlib import Path

def test_env_loading():
    """Test loading environment variables."""
    print("🧪 Testing Environment Loading")
    print("=" * 40)
    
    # Load environment
    from load_env import load_env_file, check_api_keys
    
    print("Loading environment variables...")
    success = load_env_file()
    
    if success:
        print("\n✅ Environment loaded successfully!")
        
        print("\n🔑 Available API Keys:")
        providers = check_api_keys()
        
        if providers:
            print(f"\n🎉 {len(providers)} provider(s) ready:")
            for provider in providers:
                print(f"   - {provider}")
                
            # Test specific keys for LLM integration
            print("\n🤖 LLM Integration Status:")
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                print("✅ OpenAI integration ready (GPT models)")
                
            deepseek_key = os.getenv("DEEPSEEK_API") 
            if deepseek_key:
                print("✅ DeepSeek integration ready (cost-effective)")
                
            nvidia_key = os.getenv("NVIDIA_KEY")
            if nvidia_key:
                print("✅ NVIDIA integration ready (specialized models)")
                
            print("\n📊 Recommended setup:")
            if openai_key:
                print("   🥇 Primary: GPT-4o Mini (best balance)")
                print("   💰 Cost: ~$0.00013 per page")
            
            if deepseek_key:
                print("   🥈 Alternative: DeepSeek (ultra-cheap)")
                print("   💰 Cost: ~$0.00008 per page")
                
        else:
            print("❌ No API keys found")
            
    else:
        print("❌ Failed to load environment")
        
    return success


if __name__ == "__main__":
    test_env_loading()