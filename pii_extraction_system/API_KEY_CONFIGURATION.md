# API Key Configuration Guide

This document explains how to configure API keys for different LLM providers in the PII extraction system.

## Required Environment Variables

### OpenAI
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```
- Required for: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4
- All OpenAI models support vision tasks

### Anthropic
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```
- Required for: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-opus-20240229
- All Anthropic models support vision tasks

### Google
```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```
- Required for: gemini-1.5-pro, gemini-1.5-flash
- All Google models support vision tasks

### Mistral
```bash
export MISTRAL_API_KEY="your_mistral_api_key_here"
```
- Required for: mistral-large, mistral-medium, mistral-small, mistral-tiny
- **Note**: Mistral models currently only support text-based PII extraction (no vision)

### DeepSeek
```bash
# Option 1 (preferred):
export DEEPSEEK_API_KEY="your_deepseek_api_key_here"

# Option 2 (alternative):
export DEEPSEEK_API="your_deepseek_api_key_here"
```
- Required for: deepseek-chat, deepseek-coder
- **Note**: DeepSeek models currently only support text-based PII extraction (no vision)
- DeepSeek uses OpenAI-compatible API

## Model Usage Examples

### Using Model Names Directly (Recommended)
```python
# These will be automatically normalized:
llm_service.extract_pii_from_image(image_data, "gpt-4o-mini")  # → openai/gpt-4o-mini
llm_service.extract_pii_from_text(text, "deepseek-chat")       # → deepseek/deepseek-chat
```

### Using Full Provider/Model Format
```python
# Explicit provider/model format:
llm_service.extract_pii_from_image(image_data, "openai/gpt-4o")
llm_service.extract_pii_from_text(text, "deepseek/deepseek-chat")
```

## Vision vs Text Support

| Provider | Vision Support | Text Support | Models |
|----------|----------------|--------------|--------|
| OpenAI | ✅ | ✅ | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4 |
| Anthropic | ✅ | ✅ | claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus |
| Google | ✅ | ✅ | gemini-1.5-pro, gemini-1.5-flash |
| Mistral | ❌ | ✅ | mistral-large, mistral-medium, mistral-small, mistral-tiny |
| DeepSeek | ❌ | ✅ | deepseek-chat, deepseek-coder |

## Debugging

### Check Available Models
```python
from llm.multimodal_llm_service import llm_service

# Get all available models
models = llm_service.get_available_models()
print(models)

# Get debug information
debug_info = llm_service.debug_model_availability()
print(debug_info)

# Test specific model
test_result = llm_service.test_model_access("gpt-4o-mini")
print(test_result)
```

### Common Issues and Solutions

1. **"Model 'gpt-4o-mini' not available"**
   - Solution: Set OPENAI_API_KEY environment variable
   - The system will now suggest "openai/gpt-4o-mini" as an alternative

2. **"Model 'deepseek-chat' not available"**
   - Solution: Set DEEPSEEK_API_KEY or DEEPSEEK_API environment variable
   - Get your API key from: https://platform.deepseek.com/

3. **"DeepSeek models do not currently support image processing"**
   - This is expected - use DeepSeek only for text-based PII extraction
   - For images, use OpenAI, Anthropic, or Google models

## Environment File Setup

Create a `.env` file in the project root:

```bash
# OpenAI (required for GPT models)
OPENAI_API_KEY=your_openai_key_here

# Anthropic (optional, for Claude models)
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google (optional, for Gemini models)  
GOOGLE_API_KEY=your_google_key_here

# DeepSeek (optional, for cost-effective text processing)
DEEPSEEK_API_KEY=your_deepseek_key_here

# Mistral (optional, for text processing)
MISTRAL_API_KEY=your_mistral_key_here
```

The system will automatically load this file and initialize only the providers with valid API keys.