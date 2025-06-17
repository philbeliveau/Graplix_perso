# PII Extraction System - User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Quick Start Tutorial](#quick-start-tutorial)
3. [Dashboard User Guide](#dashboard-user-guide)
4. [Command Line Usage](#command-line-usage)
5. [Configuration Guide](#configuration-guide)
6. [Best Practices](#best-practices)
7. [Common Use Cases](#common-use-cases)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### What is the PII Extraction System?

The PII Extraction System is a comprehensive tool for automatically detecting, extracting, and managing Personally Identifiable Information (PII) from various document types. It helps organizations comply with privacy regulations like GDPR and Quebec Law 25 by identifying sensitive information in their documents.

### Key Features

- **Multi-format support**: PDF, DOCX, images, and text files
- **Multiple extraction methods**: Rule-based, AI/ML models, and dictionary matching
- **Web dashboard**: User-friendly interface for document processing
- **Batch processing**: Handle multiple documents efficiently
- **Privacy compliance**: Built-in redaction and audit logging
- **Cloud integration**: Support for AWS S3 storage
- **Multi-language**: English and French support

### System Requirements

- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for models and data
- **Operating System**: Windows, macOS, or Linux

---

## Quick Start Tutorial

### Step 1: Installation

```bash
# Clone the repository
git clone <repository-url>
cd pii_extraction_system

# Install dependencies
pip install poetry
poetry install

# Or using pip
pip install -r requirements.txt
```

### Step 2: Basic Configuration

Create a configuration file or use the default settings:

```bash
# Copy default configuration
cp config/default.yaml config/my_config.yaml
```

### Step 3: Your First PII Extraction

#### Option A: Using Python Code (Recommended for Beginners)

Create a file called `my_first_extraction.py`:

```python
from core.pipeline import PIIExtractionPipeline

# Initialize the pipeline
pipeline = PIIExtractionPipeline()

# Extract PII from a text sample
sample_text = """
Dear John Smith,
Thank you for your application. Please contact us at hr@company.com 
or call (555) 123-4567 if you have any questions.
Your application ID is EMP-2024-001.

Best regards,
Jane Doe
HR Department
"""

# Process the text
result = pipeline.extract_from_text(sample_text)

# Display results
print(f"Found {len(result.entities)} PII entities:")
print(f"Overall confidence: {result.confidence_score:.2f}")
print(f"Processing time: {result.processing_time:.3f} seconds")

print("\nDetected PII:")
for entity in result.entities:
    print(f"  • {entity.type}: '{entity.value}' (confidence: {entity.confidence:.2f})")
```

Run the script:

```bash
python my_first_extraction.py
```

Expected output:
```
Found 6 PII entities:
Overall confidence: 0.85
Processing time: 0.045 seconds

Detected PII:
  • name: 'John Smith' (confidence: 0.90)
  • email: 'hr@company.com' (confidence: 0.95)
  • phone: '(555) 123-4567' (confidence: 0.88)
  • employee_id: 'EMP-2024-001' (confidence: 0.92)
  • name: 'Jane Doe' (confidence: 0.87)
  • organization: 'HR Department' (confidence: 0.75)
```

#### Option B: Using the Web Dashboard

1. Start the dashboard:
```bash
cd src/dashboard
streamlit run main.py
```

2. Open your browser to `http://localhost:8501`

3. Navigate to "Document Processing" page

4. Upload a document or paste text directly

5. Click "Process Document" to see results

### Step 4: Process Your First Document

```python
# Process a PDF file
result = pipeline.extract_from_file("path/to/your/document.pdf")

print(f"Document: {result.metadata['file_path']}")
for entity in result.entities:
    print(f"{entity.type}: {entity.value}")
```

---

## Dashboard User Guide

### Accessing the Dashboard

1. **Start the application**:
   ```bash
   cd src/dashboard
   streamlit run main.py
   ```

2. **Open in browser**: Navigate to `http://localhost:8501`

### Dashboard Pages Overview

#### 1. Document Processing 📄
- **Purpose**: Process individual documents
- **Features**:
  - Drag-and-drop file upload
  - Text input for direct processing
  - Real-time PII highlighting
  - Extractor selection
  - Export results

**How to use**:
1. Upload a file or paste text
2. Select desired extractors (or use all)
3. Click "Process Document"
4. Review highlighted PII entities
5. Export results if needed

#### 2. Batch Analysis 📊
- **Purpose**: Process multiple documents at once
- **Features**:
  - Bulk file upload
  - Progress tracking
  - Summary statistics
  - Batch export

**How to use**:
1. Upload multiple files
2. Configure processing options
3. Start batch processing
4. Monitor progress
5. Download summary report

#### 3. Model Comparison 🔍
- **Purpose**: Compare different extraction methods
- **Features**:
  - Side-by-side comparison
  - Performance metrics
  - Accuracy analysis
  - Model selection guidance

#### 4. Error Analysis ⚠️
- **Purpose**: Review and correct extraction errors
- **Features**:
  - False positive/negative identification
  - Manual correction interface
  - Feedback system
  - Model improvement suggestions

#### 5. Performance Metrics 📈
- **Purpose**: Monitor system performance
- **Features**:
  - Processing speed statistics
  - Accuracy metrics
  - Resource usage monitoring
  - Historical trends

#### 6. Data Management 🗃️
- **Purpose**: Manage processed documents and results
- **Features**:
  - Document library
  - Search and filtering
  - Bulk operations
  - Storage management

#### 7. Configuration ⚙️
- **Purpose**: Adjust system settings
- **Features**:
  - Extractor configuration
  - Privacy settings
  - Storage options
  - User preferences

---

## Command Line Usage

### Basic Commands

#### Process a Single File
```bash
python -m core.pipeline --file "document.pdf" --output "results.json"
```

#### Process Multiple Files
```bash
python -m core.pipeline --directory "documents/" --output-dir "results/"
```

#### Batch Processing with Specific Extractors
```bash
python -m core.pipeline \
    --directory "documents/" \
    --extractors "rule_based,ner" \
    --parallel \
    --output-dir "results/"
```

### Advanced Command Line Options

```bash
python -m core.pipeline [OPTIONS]

Options:
  --file TEXT              Single file to process
  --directory TEXT         Directory of files to process
  --text TEXT             Process text directly
  --output TEXT           Output file path
  --output-dir TEXT       Output directory for batch processing
  --extractors TEXT       Comma-separated list of extractors
  --config TEXT           Configuration file path
  --parallel              Enable parallel processing
  --format TEXT           Output format (json, csv, xml)
  --verbose               Enable verbose logging
  --help                  Show this message and exit
```

### Examples

```bash
# Process with custom configuration
python -m core.pipeline \
    --file "resume.pdf" \
    --config "config/production.yaml" \
    --output "resume_pii.json"

# Process directory with CSV output
python -m core.pipeline \
    --directory "hr_documents/" \
    --format "csv" \
    --output-dir "pii_results/" \
    --parallel

# Process text with specific extractors
python -m core.pipeline \
    --text "John Smith, email: john@example.com" \
    --extractors "rule_based" \
    --output "text_result.json"
```

---

## Configuration Guide

### Configuration File Structure

The system uses YAML configuration files. Here's the structure:

```yaml
# config/my_config.yaml
extractors:
  rule_based:
    enabled: true
    language: "en"  # en, fr, or both
    confidence_threshold: 0.7
    patterns:
      email: true
      phone: true
      ssn: true
      # ... other PII types
  
  ner:
    enabled: true
    model_name: "dslim/bert-base-NER"
    device: "cpu"  # or "cuda" for GPU
    confidence_threshold: 0.8
  
  dictionary:
    enabled: true
    dictionaries:
      - "names_english.txt"
      - "names_french.txt"
      - "locations.txt"

storage:
  use_s3: false
  local_path: "./data"
  s3_bucket: "my-pii-bucket"
  s3_region: "us-west-2"

privacy:
  enable_audit_logging: true
  default_redaction_method: "mask"  # mask, remove, replace, anonymize
  audit_log_path: "./logs/audit.log"

processing:
  max_parallel_jobs: 4
  chunk_size: 1000
  timeout_seconds: 300

logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_file: "./logs/pii_extraction.log"
  max_file_size: "10MB"
  backup_count: 5
```

### Environment Variables

You can also configure the system using environment variables:

```bash
# Storage configuration
export PII_USE_S3=true
export PII_S3_BUCKET=my-production-bucket
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

# Model configuration
export PII_NER_MODEL=custom-bert-model
export PII_DEVICE=cuda

# Processing configuration
export PII_MAX_PARALLEL_JOBS=8
export PII_TIMEOUT=600
```

### Customizing Extractors

#### Rule-Based Extractor Patterns

You can add custom regex patterns:

```yaml
extractors:
  rule_based:
    custom_patterns:
      custom_employee_id:
        pattern: "EMP-\\d{4}-\\d{3}"
        confidence: 0.9
        description: "Company employee ID format"
      
      custom_product_code:
        pattern: "PROD-[A-Z]{2}-\\d{6}"
        confidence: 0.8
        description: "Product code format"
```

#### NER Model Configuration

```yaml
extractors:
  ner:
    models:
      - name: "primary"
        model_name: "dslim/bert-base-NER"
        weight: 0.7
      - name: "secondary"
        model_name: "dbmdz/bert-large-cased-finetuned-conll03-english"
        weight: 0.3
```

---

## Best Practices

### 1. Document Preparation

- **Scan quality**: Use high-resolution scans (300+ DPI) for OCR
- **File formats**: PDF and DOCX are preferred over images
- **Text quality**: Ensure documents are not corrupted or password-protected
- **Language**: Specify the correct language for better accuracy

### 2. Extractor Selection

- **Start with rule-based**: Fast and reliable for common PII types
- **Add NER for names**: Better person and organization name detection
- **Use dictionary for specialized terms**: Custom vocabularies for industry-specific PII
- **Combine multiple extractors**: Best results come from ensemble methods

### 3. Performance Optimization

- **Batch processing**: Process multiple files together
- **Parallel processing**: Use multiple cores for large batches
- **GPU acceleration**: Use CUDA for NER models when available
- **Caching**: Enable model caching to avoid repeated loading

### 4. Privacy and Compliance

- **Audit logging**: Always enable audit logging for compliance
- **Data minimization**: Only process necessary documents
- **Secure storage**: Use encrypted storage for sensitive results
- **Access controls**: Implement proper user authentication
- **Regular cleanup**: Remove old processed data regularly

### 5. Quality Assurance

- **Manual review**: Always review high-risk extractions
- **Confidence thresholds**: Set appropriate confidence levels
- **False positive management**: Regularly review and correct errors
- **Testing**: Test with representative sample documents

---

## Common Use Cases

### 1. HR Document Processing

**Scenario**: Process job applications and resumes

```python
# Configure for HR documents
hr_config = {
    "extractors": {
        "rule_based": {"enabled": True, "focus": ["email", "phone", "address"]},
        "ner": {"enabled": True, "focus": ["person", "organization"]},
        "dictionary": {"enabled": True, "dictionaries": ["skills.txt"]}
    }
}

# Process resume directory
results = pipeline.batch_extract(
    glob.glob("resumes/*.pdf"),
    config=hr_config
)
```

### 2. Legal Document Review

**Scenario**: Identify PII in legal contracts

```python
# High precision configuration for legal documents
legal_config = {
    "extractors": {
        "rule_based": {"confidence_threshold": 0.9},
        "ner": {"confidence_threshold": 0.85}
    },
    "privacy": {"enable_audit_logging": True}
}

results = pipeline.extract_from_file("contract.pdf", config=legal_config)
```

### 3. Medical Records Processing

**Scenario**: Extract PII from medical documents

```python
# Medical-specific configuration
medical_config = {
    "extractors": {
        "rule_based": {
            "enabled": True,
            "patterns": ["ssn", "phone", "address", "medical_record_number"]
        },
        "dictionary": {
            "enabled": True,
            "dictionaries": ["medical_terms.txt", "medications.txt"]
        }
    }
}
```

### 4. Financial Document Analysis

**Scenario**: Process bank statements and financial records

```python
# Financial document configuration
financial_config = {
    "extractors": {
        "rule_based": {
            "patterns": ["account_number", "routing_number", "ssn", "credit_card"]
        }
    },
    "privacy": {
        "default_redaction_method": "anonymize"
    }
}
```

### 5. Email Archive Processing

**Scenario**: Process email archives for compliance

```python
# Email processing configuration
email_config = {
    "extractors": {
        "rule_based": {"patterns": ["email", "phone", "ip_address"]},
        "ner": {"focus": ["person", "organization"]}
    },
    "processing": {"chunk_size": 500}  # Smaller chunks for emails
}
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` when running the system

**Solutions**:
```bash
# Ensure you're in the correct directory
cd pii_extraction_system

# Install dependencies
pip install -r requirements.txt

# Or use Poetry
poetry install
```

#### 2. Model Download Issues

**Problem**: NER models fail to download

**Solutions**:
- Check internet connection
- Verify Hugging Face access
- Use local model files:
```python
# Use local model
config = {
    "extractors": {
        "ner": {"model_name": "/path/to/local/model"}
    }
}
```

#### 3. OCR Not Working

**Problem**: Image processing fails

**Solutions**:
```bash
# Install Tesseract
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows: Download from GitHub releases
```

#### 4. Poor Extraction Quality

**Problem**: Many false positives or missed entities

**Solutions**:
- Adjust confidence thresholds
- Use multiple extractors
- Review document quality
- Add custom patterns for specific formats

#### 5. Performance Issues

**Problem**: Slow processing speed

**Solutions**:
- Enable parallel processing
- Use GPU for NER models
- Reduce batch sizes
- Check system resources

### Error Codes Reference

| Code | Error | Solution |
|------|-------|----------|
| E001 | File not found | Check file path and permissions |
| E002 | Unsupported format | Use PDF, DOCX, or image files |
| E003 | Extraction failed | Check logs for detailed error |
| E004 | Configuration error | Validate YAML syntax |
| E005 | Storage error | Check storage permissions |
| E006 | Model load error | Verify model availability |
| E007 | OCR error | Install/configure Tesseract |

### Getting Additional Help

1. **Check logs**: Review log files in the `logs/` directory
2. **Enable debug mode**: Set log level to DEBUG for detailed information
3. **Use the dashboard**: Error Analysis page provides visual debugging
4. **Review configuration**: Ensure all settings are correct
5. **Test with samples**: Use provided sample documents to verify setup

### Performance Tuning Tips

1. **Hardware optimization**:
   - Use SSD storage for better I/O performance
   - Ensure adequate RAM (8GB+ recommended)
   - Use GPU for NER processing when available

2. **Configuration optimization**:
   - Adjust confidence thresholds based on your needs
   - Disable unused extractors
   - Use appropriate chunk sizes for your documents

3. **Workflow optimization**:
   - Process similar documents together
   - Use batch processing for multiple files
   - Enable parallel processing for large batches

---

This user guide provides comprehensive information for getting started with and effectively using the PII Extraction System. For technical details, refer to the API documentation. For deployment information, see the deployment guide.