#!/bin/bash
# Script to run the LLM test easily

echo "🔑 LLM OCR Test Runner"
echo "====================="

# Check if API key is provided as argument
if [ -z "$1" ]; then
    echo "❌ Please provide your OpenAI API key"
    echo "Usage: ./run_llm_test.sh YOUR_OPENAI_API_KEY"
    echo ""
    echo "Or set environment variable:"
    echo "export OPENAI_API_KEY='your-key-here'"
    echo "./run_llm_test.sh"
    exit 1
fi

# Set the API key
export OPENAI_API_KEY="$1"

echo "✅ API key set"
echo "🚀 Running LLM test on medical form..."
echo ""

# Run the Python script
python3 llm_test_with_key.py "$1"

echo ""
echo "📄 Check the generated JSON file for detailed results"