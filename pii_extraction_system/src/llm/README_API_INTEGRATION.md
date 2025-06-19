# Multi-LLM API Integration for PII Extraction System

## Overview

This module provides a comprehensive multi-LLM API integration wrapper that supports:
- Multiple LLM providers (OpenAI, Anthropic, Google, Mistral)
- Conditional API key handling
- Cost tracking and token usage monitoring
- Error handling and fallback mechanisms
- Dashboard integration with real-time monitoring

## Supported LLM Providers

### Primary Providers (Vision-Enabled)
1. **OpenAI**
   - GPT-4o (Omni) - Best for complex documents
   - GPT-4o Mini - Cost-effective option
   - GPT-4 Turbo - High-quality reasoning
   - GPT-4 - Legacy support

2. **Anthropic**
   - Claude 3.5 Sonnet - Best overall performance
   - Claude 3.5 Haiku - Fast and cost-effective
   - Claude 3 Opus - Premium quality

3. **Google**
   - Gemini 1.5 Pro - Long context support (1M tokens)
   - Gemini 1.5 Flash - Ultra-fast processing

### Text-Only Providers
4. **Mistral**
   - Mistral Large/Medium/Small/Tiny - European language support

## Quick Start

### 1. Configuration

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Basic Usage

```python
from llm import llm_integration

# Check available models
available_models = llm_integration.get_available_models()
print(f"Available models: {list(available_models.keys())}")

# Extract PII from an image
result = llm_integration.extract_pii_with_tracking(
    image_data=base64_image_string,
    model_key="openai/gpt-4o-mini",
    document_type="invoice"
)

if result["success"]:
    print(f"Extracted PII: {result['pii_entities']}")
    print(f"Cost: ${result['usage']['estimated_cost']:.4f}")
```

### 3. Dashboard Access

1. Start the dashboard:
   ```bash
   streamlit run src/dashboard/main.py
   ```

2. Navigate to "LLM API Status" in the sidebar

3. View:
   - API key status for all providers
   - Available models and costs
   - Usage tracking and limits
   - System health monitoring

## Key Features

### 1. Automatic Model Selection

```python
# Get best model for specific task
best_model = llm_integration.get_best_model_for_task("complex_documents")
# Returns: "anthropic/claude-3-5-sonnet-20241022"

# Get models by capability
cost_effective_models = llm_integration.get_models_by_capability("cost_effective")
```

### 2. Batch Processing with Fallback

```python
# Process multiple documents with automatic fallback
results = llm_integration.batch_extract_with_fallback(
    image_data_list=images,
    primary_model="openai/gpt-4o",
    fallback_models=["anthropic/claude-3-5-sonnet-20241022", "google/gemini-1.5-pro"],
    progress_callback=lambda i, total, cost: print(f"Progress: {i}/{total}, Cost: ${cost:.2f}")
)
```

### 3. Cost Tracking

```python
# Get session costs
session_costs = llm_integration.get_cost_summary("session")
print(f"Total session cost: ${session_costs['summary']['total_estimated_cost']:.2f}")

# Set usage limits
llm_integration.set_usage_limits(
    provider="openai",
    daily_limit=10.0,
    monthly_limit=100.0
)
```

### 4. Model Recommendations

```python
# Get recommendations based on requirements
recommendations = llm_integration.get_model_recommendations({
    'cost': 'low',
    'accuracy': 'high',
    'complex_documents': True
})
```

## API Reference

### Core Methods

#### `extract_pii_with_tracking()`
Extract PII from an image with full tracking and error handling.

**Parameters:**
- `image_data` (str): Base64 encoded image
- `model_key` (str): Model identifier (e.g., "openai/gpt-4o")
- `document_type` (str): Type of document (default: "document")
- `user_id` (str, optional): User identifier for tracking
- `document_id` (str, optional): Document identifier

**Returns:**
- Dictionary with extraction results, usage info, and cost tracking

#### `get_available_models()`
Get all available models with their detailed information.

**Returns:**
- Dictionary of model_key -> LLMModelInfo objects

#### `get_cost_summary(time_period)`
Get cost summary for specified time period.

**Parameters:**
- `time_period` (str): 'session', 'daily', 'monthly', or 'all'

**Returns:**
- Cost summary with breakdown by provider and model

## Cost Information

### Pricing (per 1K tokens)

| Provider | Model | Input Cost | Output Cost |
|----------|-------|------------|-------------|
| OpenAI | GPT-4o | $0.0025 | $0.01 |
| OpenAI | GPT-4o Mini | $0.00015 | $0.0006 |
| Anthropic | Claude 3.5 Sonnet | $0.003 | $0.015 |
| Anthropic | Claude 3.5 Haiku | $0.001 | $0.005 |
| Google | Gemini 1.5 Pro | $0.0025 | $0.0075 |
| Google | Gemini 1.5 Flash | $0.000075 | $0.0003 |

## Error Handling

The integration includes comprehensive error handling:

1. **Missing API Keys**: Gracefully handles missing keys and shows available models only
2. **API Failures**: Automatic fallback to alternative models
3. **Rate Limits**: Built-in monitoring and alerts
4. **Cost Limits**: Configurable daily/monthly limits with alerts

## Dashboard Features

The LLM API Status dashboard provides:

1. **API Status Tab**
   - Real-time API key validation
   - Provider availability status
   - Configuration instructions

2. **Available Models Tab**
   - Model comparison table
   - Cost visualization
   - Performance recommendations

3. **Cost Analysis Tab**
   - Session/daily/monthly cost tracking
   - Provider-wise breakdown
   - Usage trends

4. **Usage Limits Tab**
   - Set and monitor spending limits
   - Real-time alerts
   - Usage vs. limit visualization

5. **System Health Tab**
   - Overall system status
   - Model availability by provider
   - Export usage reports

## Best Practices

1. **API Key Security**
   - Never commit `.env` files
   - Use environment variables in production
   - Rotate keys regularly

2. **Cost Optimization**
   - Use GPT-4o Mini or Claude Haiku for simple documents
   - Enable batch processing for efficiency
   - Set appropriate usage limits

3. **Model Selection**
   - Claude 3.5 Sonnet for best overall accuracy
   - GPT-4o Mini for cost-effective processing
   - Gemini 1.5 Flash for ultra-fast processing

4. **Error Recovery**
   - Always configure fallback models
   - Monitor error rates in dashboard
   - Export usage reports regularly

## Troubleshooting

### Common Issues

1. **No models available**
   - Check if API keys are correctly set in `.env`
   - Verify key format and validity
   - Use the validation feature in dashboard

2. **High costs**
   - Review model selection
   - Enable cost limits
   - Use batch processing

3. **API errors**
   - Check rate limits
   - Verify API key permissions
   - Review error logs in dashboard

## Memory Storage

The integration automatically saves important metrics to memory:
```
swarm-development-centralized-1750358285173/api-specialist-recovery/integration-implementation
```

This includes:
- API configuration status
- Cost tracking data
- Usage patterns
- Error logs