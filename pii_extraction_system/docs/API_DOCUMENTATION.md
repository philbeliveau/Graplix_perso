# PII Extraction System - API Documentation

## Overview

The PII Extraction System provides a comprehensive API for detecting, extracting, and managing Personally Identifiable Information (PII) from various document types. This documentation covers all major APIs, classes, and methods available in the system.

## Table of Contents

1. [Core Pipeline API](#core-pipeline-api)
2. [Extractor APIs](#extractor-apis)
3. [Document Processing API](#document-processing-api)
4. [Storage API](#storage-api)
5. [Privacy & Redaction API](#privacy--redaction-api)
6. [Configuration API](#configuration-api)
7. [Dashboard API](#dashboard-api)
8. [Code Examples](#code-examples)

---

## Core Pipeline API

### PIIExtractionPipeline

The main pipeline orchestrates all PII extraction processes.

#### Class: `PIIExtractionPipeline`

```python
from core.pipeline import PIIExtractionPipeline

pipeline = PIIExtractionPipeline(config_path="config/default.yaml")
```

#### Methods

##### `extract_from_file(file_path: str, extractor_types: List[str] = None) -> PIIResult`

Extracts PII from a single file.

**Parameters:**
- `file_path` (str): Path to the document file
- `extractor_types` (List[str], optional): List of extractor types to use. Defaults to all available.

**Returns:**
- `PIIResult`: Complete extraction results with entities, confidence scores, and metadata

**Example:**
```python
result = pipeline.extract_from_file("/path/to/document.pdf")
print(f"Found {len(result.entities)} PII entities")
for entity in result.entities:
    print(f"{entity.type}: {entity.value} (confidence: {entity.confidence})")
```

##### `extract_from_text(text: str, extractor_types: List[str] = None) -> PIIResult`

Extracts PII from raw text content.

**Parameters:**
- `text` (str): Text content to analyze
- `extractor_types` (List[str], optional): List of extractor types to use

**Returns:**
- `PIIResult`: Extraction results

**Example:**
```python
text = "Contact John Doe at john.doe@email.com or call 555-123-4567"
result = pipeline.extract_from_text(text)
```

##### `batch_extract(file_paths: List[str], parallel: bool = True) -> List[PIIResult]`

Processes multiple files in batch.

**Parameters:**
- `file_paths` (List[str]): List of file paths to process
- `parallel` (bool): Whether to use parallel processing

**Returns:**
- `List[PIIResult]`: List of extraction results for each file

---

## Extractor APIs

### Rule-Based Extractor

#### Class: `RuleBasedExtractor`

```python
from extractors.rule_based import RuleBasedExtractor

extractor = RuleBasedExtractor(language="en")
```

#### Supported PII Types

- **Personal Identifiers**: Names, Social Security Numbers, Employee IDs
- **Contact Information**: Email addresses, phone numbers, addresses
- **Financial**: Credit card numbers, bank account numbers
- **Medical**: Medical record numbers, health insurance numbers
- **Technical**: IP addresses, URLs, MAC addresses
- **Documents**: Passport numbers, driver's license numbers
- **Temporal**: Dates, times

#### Methods

##### `extract(text: str) -> List[PIIEntity]`

Extracts PII using regex patterns.

**Example:**
```python
entities = extractor.extract("Email: user@domain.com, Phone: +1-555-123-4567")
```

### NER Extractor

#### Class: `NERExtractor`

```python
from extractors.ner_extractor import NERExtractor

extractor = NERExtractor(model_name="dslim/bert-base-NER")
```

#### Methods

##### `extract(text: str) -> List[PIIEntity]`

Uses Hugging Face NER models for extraction.

**Example:**
```python
entities = extractor.extract("John Smith works at Microsoft in Seattle.")
```

### Dictionary Extractor

#### Class: `DictionaryExtractor`

```python
from extractors.dictionary_extractor import DictionaryExtractor

extractor = DictionaryExtractor()
```

#### Methods

##### `extract(text: str) -> List[PIIEntity]`

Uses predefined dictionaries for common names and locations.

---

## Document Processing API

### DocumentProcessor

#### Class: `DocumentProcessor`

```python
from utils.document_processor import DocumentProcessor

processor = DocumentProcessor()
```

#### Methods

##### `process_pdf(file_path: str) -> str`

Extracts text from PDF files.

**Parameters:**
- `file_path` (str): Path to PDF file

**Returns:**
- `str`: Extracted text content

##### `process_docx(file_path: str) -> str`

Extracts text from DOCX files.

##### `process_image(file_path: str, language: str = "eng") -> str`

Extracts text from images using OCR.

**Parameters:**
- `file_path` (str): Path to image file
- `language` (str): OCR language code (eng, fra, etc.)

**Example:**
```python
text = processor.process_image("/path/to/scan.png", language="fra")
```

---

## Storage API

### DataStorage

#### Class: `DataStorage`

```python
from utils.data_storage import DataStorage

storage = DataStorage(use_s3=True, bucket_name="my-pii-bucket")
```

#### Methods

##### `save_result(result: PIIResult, key: str) -> bool`

Saves extraction results to storage.

##### `load_result(key: str) -> PIIResult`

Loads previously saved results.

##### `list_files(prefix: str = "") -> List[str]`

Lists available files in storage.

---

## Privacy & Redaction API

### PIIRedactor

#### Class: `PIIRedactor`

```python
from privacy.redaction import PIIRedactor

redactor = PIIRedactor()
```

#### Methods

##### `redact_text(text: str, entities: List[PIIEntity], method: str = "mask") -> str`

Redacts PII from text using various methods.

**Parameters:**
- `text` (str): Original text
- `entities` (List[PIIEntity]): PII entities to redact
- `method` (str): Redaction method ("mask", "remove", "replace", "anonymize")

**Example:**
```python
redacted = redactor.redact_text(
    "Contact John at john@email.com", 
    entities, 
    method="mask"
)
# Output: "Contact **** at ****@*****.***"
```

##### `create_audit_log(entities: List[PIIEntity]) -> AuditLog`

Creates audit log for compliance.

---

## Configuration API

### Settings

#### Class: `Settings`

```python
from core.config import Settings

settings = Settings()
```

#### Configuration Options

```python
# Extractor Configuration
settings.extractors.rule_based.enabled = True
settings.extractors.rule_based.language = "en"
settings.extractors.rule_based.confidence_threshold = 0.7

# NER Configuration
settings.extractors.ner.model_name = "dslim/bert-base-NER"
settings.extractors.ner.device = "cuda"

# Storage Configuration
settings.storage.use_s3 = False
settings.storage.local_path = "./data"
settings.storage.s3_bucket = "pii-extraction-bucket"

# Privacy Configuration
settings.privacy.enable_audit_logging = True
settings.privacy.default_redaction_method = "mask"
```

---

## Dashboard API

### Streamlit Dashboard

The dashboard provides a web interface for the PII extraction system.

#### Running the Dashboard

```bash
cd src/dashboard
streamlit run main.py
```

#### Available Pages

1. **Document Processing**: Upload and process individual documents
2. **Batch Analysis**: Process multiple documents
3. **Model Comparison**: Compare different extractor performance
4. **Error Analysis**: Review and correct extraction errors
5. **Performance Metrics**: View system performance statistics
6. **Data Management**: Manage processed documents and results
7. **Configuration**: Adjust system settings

---

## Code Examples

### Basic Usage

```python
from core.pipeline import PIIExtractionPipeline

# Initialize pipeline
pipeline = PIIExtractionPipeline()

# Process a document
result = pipeline.extract_from_file("resume.pdf")

# Display results
print(f"Document: {result.metadata['file_path']}")
print(f"Processing time: {result.metadata['processing_time']:.3f}s")
print(f"Confidence score: {result.confidence_score:.2f}")

for entity in result.entities:
    print(f"- {entity.type}: {entity.value} "
          f"(confidence: {entity.confidence:.2f})")
```

### Batch Processing

```python
import glob

# Process all PDFs in a directory
pdf_files = glob.glob("documents/*.pdf")
results = pipeline.batch_extract(pdf_files, parallel=True)

# Analyze results
total_entities = sum(len(r.entities) for r in results)
avg_confidence = sum(r.confidence_score for r in results) / len(results)

print(f"Processed {len(results)} documents")
print(f"Found {total_entities} PII entities")
print(f"Average confidence: {avg_confidence:.2f}")
```

### Custom Extractor Configuration

```python
# Configure specific extractors
pipeline = PIIExtractionPipeline()

# Use only rule-based and NER extractors
result = pipeline.extract_from_file(
    "document.pdf", 
    extractor_types=["rule_based", "ner"]
)
```

### Privacy-Preserving Processing

```python
from privacy.redaction import PIIRedactor

# Extract PII
result = pipeline.extract_from_file("sensitive_document.pdf")

# Redact PII
redactor = PIIRedactor()
redacted_text = redactor.redact_text(
    result.original_text,
    result.entities,
    method="anonymize"
)

# Create audit log
audit_log = redactor.create_audit_log(result.entities)
print(f"Audit log created with {len(audit_log.entries)} entries")
```

### Advanced Configuration

```python
from core.config import Settings

# Load custom configuration
settings = Settings()
settings.extractors.rule_based.confidence_threshold = 0.8
settings.extractors.ner.model_name = "custom-bert-model"
settings.storage.use_s3 = True
settings.storage.s3_bucket = "production-pii-bucket"

# Initialize pipeline with custom settings
pipeline = PIIExtractionPipeline(settings=settings)
```

### Error Handling

```python
from core.exceptions import PIIExtractionError, DocumentProcessingError

try:
    result = pipeline.extract_from_file("document.pdf")
except DocumentProcessingError as e:
    print(f"Failed to process document: {e}")
except PIIExtractionError as e:
    print(f"Extraction failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Data Structures

### PIIEntity

```python
@dataclass
class PIIEntity:
    type: str           # PII type (e.g., "email", "phone", "name")
    value: str          # The actual PII value
    confidence: float   # Confidence score (0.0-1.0)
    start_pos: int      # Start position in text
    end_pos: int        # End position in text
    extractor: str      # Which extractor found this entity
    metadata: Dict[str, Any]  # Additional metadata
```

### PIIResult

```python
@dataclass
class PIIResult:
    entities: List[PIIEntity]       # List of found PII entities
    confidence_score: float         # Overall confidence score
    processing_time: float          # Time taken to process
    original_text: str              # Original text content
    metadata: Dict[str, Any]        # Processing metadata
```

---

## Error Codes

| Code | Error | Description |
|------|-------|-------------|
| E001 | DocumentNotFound | Specified file does not exist |
| E002 | UnsupportedFormat | File format not supported |
| E003 | ExtractionFailed | PII extraction process failed |
| E004 | ConfigurationError | Invalid configuration settings |
| E005 | StorageError | Failed to save/load data |
| E006 | ModelLoadError | Failed to load ML model |
| E007 | OCRError | OCR processing failed |

---

## Performance Guidelines

### Optimization Tips

1. **Use appropriate extractors**: Choose extractors based on your needs
2. **Batch processing**: Process multiple files together for better performance
3. **Parallel processing**: Enable parallel processing for large batches
4. **Model caching**: NER models are cached after first load
5. **Configuration tuning**: Adjust confidence thresholds based on requirements

### Performance Benchmarks

- **Rule-based extractor**: ~0.001s per document
- **NER extractor**: ~0.1-0.5s per document (depending on length)
- **OCR processing**: ~2-5s per page
- **Batch processing**: 10-50x faster than individual processing

---

## Security Considerations

### Data Protection

1. **In-transit encryption**: All data transfers use HTTPS/TLS
2. **At-rest encryption**: S3 storage uses AES-256 encryption
3. **Access controls**: Role-based access to sensitive operations
4. **Audit logging**: All PII access is logged for compliance

### Compliance Features

- **GDPR compliance**: Right to be forgotten, data portability
- **Quebec Law 25**: Privacy impact assessments, consent management
- **Audit trails**: Complete processing history
- **Data minimization**: Only necessary data is stored

---

## Support and Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Model download fails**: Check internet connection and Hugging Face access
3. **OCR not working**: Verify Tesseract installation
4. **S3 access denied**: Check AWS credentials and permissions

### Getting Help

- Check the troubleshooting guide in `docs/TROUBLESHOOTING.md`
- Review system logs in the `logs/` directory
- Use the dashboard's error analysis page for debugging

---

## API Reference Quick Links

- [Core Pipeline](src/core/pipeline.py) - Main processing pipeline
- [Rule-Based Extractor](src/extractors/rule_based.py) - Regex-based extraction
- [NER Extractor](src/extractors/ner_extractor.py) - Neural entity recognition
- [Document Processor](src/utils/document_processor.py) - Document handling
- [Privacy Tools](src/privacy/redaction.py) - PII redaction and privacy
- [Configuration](src/core/config.py) - System configuration

This API documentation is maintained as part of the PII Extraction System project. For the latest updates, please refer to the source code and inline documentation.