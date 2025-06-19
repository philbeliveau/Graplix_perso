# LLM Extraction Tests

This folder contains the enhanced LLM-based PII extraction testing system.

## Files

- `enhanced_llm_test.py` - Main test script with improved prompts and NER-based PII detection
- `enhanced_llm_test_results.json` - Latest test results with actual PII entity extraction

## Usage

```bash
# Run from the pii_extraction_system directory
python tests/llm_extraction/enhanced_llm_test.py
```

## Features

- **Automatic OpenAI API key loading** from `.env` file
- **Advanced PII detection** using regex patterns and NER
- **Password-protected document support** with msoffcrypto
- **Improved prompts** to handle OpenAI safety filters
- **Real entity extraction**: names, emails, phones, dates, addresses, etc.

## Requirements

- OpenAI API key in `.env` file as `OPENAI_API_KEY`
- Required packages: `python-dotenv`, `msoffcrypto-tool`, `openai`, etc.

## Recent Improvements

1. **Fixed OpenAI Safety Filters**: Changed from keyword-based to business compliance-focused prompts
2. **Real PII Extraction**: Now extracts actual names, emails, dates instead of just keywords
3. **Better Document Support**: Handles password-protected Office documents
4. **Cost Optimization**: Reduced API costs while improving accuracy