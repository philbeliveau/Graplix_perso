# PII Extraction System - System Architecture

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Component Interactions](#component-interactions)
5. [Storage Architecture](#storage-architecture)
6. [Security Architecture](#security-architecture)
7. [Deployment Architecture](#deployment-architecture)
8. [Scalability Design](#scalability-design)

---

## Architecture Overview

The PII Extraction System follows a modular, microservices-inspired architecture designed for scalability, maintainability, and extensibility. The system is built using a layered approach with clear separation of concerns.

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                          │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Dashboard  │  REST API  │  CLI Tools  │  SDK/Library │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│            PII Extraction Pipeline (Core Controller)            │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Document     │  Extractor    │  Privacy      │  Evaluation     │
│  Processor    │  Manager      │  Manager      │  Framework      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXTRACTION ENGINES                            │
├─────────────────────────────────────────────────────────────────┤
│  Rule-Based   │  NER Models   │  Dictionary   │  Layout-Aware   │
│  Extractor    │  (BERT/GPT)   │  Matching     │  Models         │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     UTILITY LAYER                               │
├─────────────────────────────────────────────────────────────────┤
│  Configuration│  Logging      │  Storage      │  Monitoring     │
│  Management   │  System       │  Abstraction  │  & Metrics      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE                               │
├─────────────────────────────────────────────────────────────────┤
│  Local Storage│  AWS S3       │  MLflow       │  Databases      │
│  File System  │  Cloud Storage│  Experiment   │  (SQLite/       │
│               │               │  Tracking     │  PostgreSQL)    │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Principles

1. **Modularity**: Each component has a single responsibility
2. **Extensibility**: Easy to add new extractors or processors
3. **Configurability**: All behavior controlled through configuration
4. **Scalability**: Horizontal and vertical scaling support
5. **Observability**: Comprehensive logging and monitoring
6. **Security**: Privacy-first design with audit trails
7. **Performance**: Optimized for batch processing and parallel execution

---

## Core Components

### 1. PII Extraction Pipeline (`core/pipeline.py`)

**Responsibility**: Main orchestrator that coordinates all extraction activities

**Key Features**:
- Manages extractor lifecycle
- Handles parallel processing
- Aggregates results from multiple extractors
- Provides unified API interface

**Architecture Pattern**: Controller/Orchestrator

```python
class PIIExtractionPipeline:
    def __init__(self, config: Settings):
        self.config = config
        self.extractors = self._initialize_extractors()
        self.document_processor = DocumentProcessor()
        self.storage = DataStorage(config.storage)
    
    def extract_from_file(self, file_path: str) -> PIIResult:
        # 1. Document processing
        # 2. Parallel extraction
        # 3. Result aggregation
        # 4. Confidence scoring
        # 5. Storage (optional)
```

### 2. Document Processor (`utils/document_processor.py`)

**Responsibility**: Handles different document formats and text extraction

**Supported Formats**:
- PDF files (using PyPDF2/pdfplumber)
- DOCX files (using python-docx)
- Images (using Tesseract OCR)
- Plain text files

**Architecture Pattern**: Strategy Pattern

```python
class DocumentProcessor:
    def __init__(self):
        self.processors = {
            '.pdf': PDFProcessor(),
            '.docx': DOCXProcessor(),
            '.png': ImageProcessor(),
            '.jpg': ImageProcessor(),
            # ... other formats
        }
    
    def process(self, file_path: str) -> str:
        processor = self._get_processor(file_path)
        return processor.extract_text(file_path)
```

### 3. Extractor Framework

**Responsibility**: Provides common interface for all PII extraction methods

**Architecture Pattern**: Plugin Architecture with Factory Pattern

#### Base Extractor Interface

```python
class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, text: str) -> List[PIIEntity]:
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[str]:
        pass
```

#### Extractor Implementations

##### Rule-Based Extractor (`extractors/rule_based.py`)
- Uses regex patterns for common PII types
- Fast and deterministic
- Configurable patterns and confidence scores
- Multi-language support

##### NER Extractor (`extractors/ner_extractor.py`)
- Uses Hugging Face Transformers
- Pre-trained models (BERT, RoBERTa, etc.)
- GPU acceleration support
- Model caching for performance

##### Dictionary Extractor (`extractors/dictionary_extractor.py`)
- Uses predefined word lists
- Fuzzy matching capabilities
- Custom dictionaries support
- Efficient trie-based searching

##### Layout-Aware Extractor (`extractors/layout_aware.py`)
- Uses LayoutLM/Donut models
- Understands document structure
- Better for forms and structured documents
- Advanced ML model integration

### 4. Privacy and Compliance Framework (`privacy/`)

**Responsibility**: Handles privacy-preserving operations and compliance

**Components**:
- **Redaction Engine**: Multiple redaction strategies
- **Audit Logger**: GDPR/Law 25 compliant logging
- **Anonymization**: Advanced privacy techniques
- **Compliance Checker**: Regulatory requirement validation

### 5. Storage Abstraction Layer (`utils/data_storage.py`)

**Responsibility**: Provides unified interface for different storage backends

**Architecture Pattern**: Adapter Pattern

```python
class DataStorage:
    def __init__(self, config: StorageConfig):
        if config.use_s3:
            self.backend = S3StorageBackend(config)
        else:
            self.backend = LocalStorageBackend(config)
    
    def save(self, key: str, data: Any) -> bool:
        return self.backend.save(key, data)
```

---

## Data Flow

### 1. Single Document Processing Flow

```
Input Document
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Document    │───▶│ Text         │───▶│ Text        │
│ Upload      │    │ Extraction   │    │ Validation  │
└─────────────┘    └──────────────┘    └─────────────┘
      │                    │                    │
      ▼                    ▼                    ▼
File Type            Format-Specific         Clean Text
Detection            Processing              Output
      │                    │                    │
      └────────────────────┼────────────────────┘
                           ▼
              ┌─────────────────────────┐
              │ Parallel Extraction     │
              │ ┌─────────────────────┐ │
              │ │ Rule-Based Engine   │ │
              │ │ NER Engine          │ │
              │ │ Dictionary Engine   │ │
              │ │ Layout-Aware Engine │ │
              │ └─────────────────────┘ │
              └─────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Result Aggregation      │
              │ • Entity Deduplication  │
              │ • Confidence Scoring    │
              │ • Conflict Resolution   │
              └─────────────────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Post-Processing         │
              │ • Privacy Filtering     │
              │ • Audit Logging         │
              │ • Result Storage        │
              └─────────────────────────┘
                           │
                           ▼
                    Final PIIResult
```

### 2. Batch Processing Flow

```
Document Batch Input
         │
         ▼
┌─────────────────┐
│ Batch Manager   │
│ • Load Balancing│
│ • Queue Management
│ • Progress Tracking
└─────────────────┘
         │
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Worker Thread 1 │    │ Worker Thread 2 │    │ Worker Thread N │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ Document A  │ │    │ │ Document B  │ │    │ │ Document N  │ │
│ │ Processing  │ │    │ │ Processing  │ │    │ │ Processing  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │ Result Aggregation      │
                    │ • Collect all results   │
                    │ • Generate summary      │
                    │ • Export batch report   │
                    └─────────────────────────┘
```

---

## Component Interactions

### 1. Extractor Coordination

```python
# Pipeline coordinates multiple extractors
class PIIExtractionPipeline:
    def _extract_parallel(self, text: str) -> List[PIIEntity]:
        futures = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            for extractor in self.extractors:
                future = executor.submit(extractor.extract, text)
                futures.append((extractor.name, future))
        
        all_entities = []
        for name, future in futures:
            entities = future.result()
            for entity in entities:
                entity.extractor = name
            all_entities.extend(entities)
        
        return self._deduplicate_entities(all_entities)
```

### 2. Configuration Management

```python
# Centralized configuration affects all components
class Settings:
    extractors: ExtractorConfig
    storage: StorageConfig
    privacy: PrivacyConfig
    processing: ProcessingConfig
    
    def __post_init__(self):
        # Validate configuration consistency
        self._validate_extractor_dependencies()
        self._validate_storage_credentials()
        self._setup_logging()
```

### 3. Event-Driven Architecture

```python
# Event system for component communication
class EventManager:
    def __init__(self):
        self.listeners = defaultdict(list)
    
    def emit(self, event: str, data: Any):
        for listener in self.listeners[event]:
            listener(data)
    
    def on(self, event: str, callback: Callable):
        self.listeners[event].append(callback)

# Usage in pipeline
events.emit('document_processed', {'doc_id': doc_id, 'entities': entities})
events.emit('extraction_complete', {'total_entities': len(entities)})
```

---

## Storage Architecture

### 1. Storage Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│                    STORAGE ABSTRACTION                      │
├─────────────────────────────────────────────────────────────┤
│                  DataStorage Interface                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────┐           │           ┌─────────────────┐
│  Local Storage  │           │           │   AWS S3        │
│  Backend        │◄──────────┼──────────►│   Backend       │
│                 │           │           │                 │
│ • File System   │           │           │ • S3 Buckets    │
│ • SQLite DB     │           │           │ • IAM Roles     │
│ • Local Cache   │           │           │ • Encryption    │
└─────────────────┘           │           └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    METADATA STORAGE                         │
├─────────────────────────────────────────────────────────────┤
│ • Document Index                                            │
│ • Processing History                                        │
│ • User Sessions                                             │
│ • Configuration Cache                                       │
└─────────────────────────────────────────────────────────────┘
```

### 2. Data Organization

```
Storage Root/
├── documents/
│   ├── originals/          # Original uploaded documents
│   ├── processed/          # Processed text files
│   └── thumbnails/         # Document previews
├── results/
│   ├── extractions/        # PII extraction results (JSON)
│   ├── reports/           # Analysis reports
│   └── exports/           # User-exported data
├── models/
│   ├── cache/             # Cached ML models
│   ├── custom/            # User-trained models
│   └── embeddings/        # Pre-computed embeddings
├── config/
│   ├── user_settings/     # User-specific configurations
│   └── system/            # System-wide settings
└── logs/
    ├── audit/             # Compliance audit logs
    ├── performance/       # System performance logs
    └── errors/            # Error logs
```

### 3. Database Schema (Metadata)

```sql
-- Document tracking
CREATE TABLE documents (
    id UUID PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(100),
    upload_timestamp TIMESTAMP,
    processing_status ENUM('pending', 'processing', 'completed', 'failed'),
    user_id UUID,
    storage_path VARCHAR(500)
);

-- Extraction results
CREATE TABLE extractions (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id),
    extractor_type VARCHAR(50),
    entity_count INTEGER,
    confidence_score FLOAT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP,
    result_path VARCHAR(500)
);

-- PII entities
CREATE TABLE pii_entities (
    id UUID PRIMARY KEY,
    extraction_id UUID REFERENCES extractions(id),
    entity_type VARCHAR(50),
    entity_value_hash VARCHAR(64), -- Hashed for privacy
    confidence FLOAT,
    start_position INTEGER,
    end_position INTEGER,
    created_at TIMESTAMP
);

-- Audit logs
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY,
    user_id UUID,
    action VARCHAR(100),
    resource_id UUID,
    timestamp TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    result ENUM('success', 'failure', 'error')
);
```

---

## Security Architecture

### 1. Security Layers

```
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION SECURITY                      │
├─────────────────────────────────────────────────────────────┤
│ • Input Validation         • Output Sanitization            │
│ • SQL Injection Prevention • XSS Protection                 │
│ • CSRF Protection         • Rate Limiting                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  AUTHENTICATION & AUTHORIZATION             │
├─────────────────────────────────────────────────────────────┤
│ • Multi-factor Authentication • Role-Based Access Control   │
│ • Session Management          • API Key Management          │
│ • OAuth2/OIDC Integration     • Audit Logging               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     DATA PROTECTION                         │
├─────────────────────────────────────────────────────────────┤
│ • Encryption at Rest (AES-256) • Encryption in Transit      │
│ • Key Management (AWS KMS)     • Data Anonymization         │
│ • Secure Deletion             • Privacy Controls            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE SECURITY                    │
├─────────────────────────────────────────────────────────────┤
│ • Network Segmentation      • Firewall Rules                │
│ • VPC Configuration         • Security Groups               │
│ • Container Security        • Secrets Management            │
└─────────────────────────────────────────────────────────────┘
```

### 2. Privacy-Preserving Architecture

```python
class PrivacyManager:
    def __init__(self, config: PrivacyConfig):
        self.redactor = PIIRedactor()
        self.anonymizer = DataAnonymizer()
        self.audit_logger = AuditLogger()
    
    def process_with_privacy(self, text: str, entities: List[PIIEntity]) -> str:
        # 1. Log access for audit
        self.audit_logger.log_access(entities)
        
        # 2. Apply privacy transformations
        if self.config.redaction_enabled:
            text = self.redactor.redact(text, entities)
        
        # 3. Anonymize sensitive data
        if self.config.anonymization_enabled:
            text = self.anonymizer.anonymize(text, entities)
        
        return text
```

### 3. Compliance Framework

```python
class ComplianceFramework:
    def __init__(self):
        self.gdpr_handler = GDPRComplianceHandler()
        self.law25_handler = Law25ComplianceHandler()
        self.audit_manager = AuditManager()
    
    def ensure_compliance(self, operation: str, data: Any):
        # Check GDPR requirements
        self.gdpr_handler.validate_processing_basis(operation)
        
        # Check Quebec Law 25
        self.law25_handler.validate_consent(data)
        
        # Create audit trail
        self.audit_manager.record_processing(operation, data)
```

---

## Deployment Architecture

### 1. Container Architecture

```dockerfile
# Multi-stage Docker build
FROM python:3.11-slim as base
# Base dependencies and configuration

FROM base as models
# Download and cache ML models

FROM base as app
# Application code and runtime
COPY --from=models /models /app/models
```

### 2. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pii-extraction-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pii-extraction-api
  template:
    spec:
      containers:
      - name: api
        image: pii-extraction:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: PII_CONFIG_PATH
          value: "/config/production.yaml"
        volumeMounts:
        - name: config
          mountPath: /config
        - name: models
          mountPath: /models
```

### 3. AWS Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER LAYER                           │
├─────────────────────────────────────────────────────────────┤
│    CloudFront CDN    │    Route 53 DNS    │   WAF           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  Application Load   │   ECS Fargate     │   Auto Scaling    │
│  Balancer          │   Containers      │   Groups          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      DATA LAYER                             │
├─────────────────────────────────────────────────────────────┤
│   S3 Buckets        │   RDS PostgreSQL  │   ElastiCache     │
│   (Documents)       │   (Metadata)      │   (Sessions)      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                         │
├─────────────────────────────────────────────────────────────┤
│   CloudWatch        │   AWS X-Ray       │   SNS Alerts      │
│   (Metrics/Logs)    │   (Tracing)       │   (Notifications) │
└─────────────────────────────────────────────────────────────┘
```

---

## Scalability Design

### 1. Horizontal Scaling

```python
class ScalableProcessor:
    def __init__(self, config: ProcessingConfig):
        self.worker_pool = ProcessPoolExecutor(
            max_workers=config.max_workers
        )
        self.queue = Queue(maxsize=config.queue_size)
        self.load_balancer = LoadBalancer()
    
    def process_batch(self, documents: List[str]) -> List[PIIResult]:
        # Distribute work across workers
        chunks = self.load_balancer.distribute(documents)
        
        futures = []
        for chunk in chunks:
            future = self.worker_pool.submit(self._process_chunk, chunk)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
        
        return results
```

### 2. Caching Strategy

```python
class ModelCache:
    def __init__(self):
        self.memory_cache = {}  # In-memory for frequently used models
        self.disk_cache = DiskCache()  # Persistent cache for large models
        self.redis_cache = RedisCache()  # Distributed cache for clusters
    
    def get_model(self, model_name: str):
        # 1. Check memory cache
        if model_name in self.memory_cache:
            return self.memory_cache[model_name]
        
        # 2. Check Redis cache (for clusters)
        model = self.redis_cache.get(model_name)
        if model:
            self.memory_cache[model_name] = model
            return model
        
        # 3. Load from disk or download
        model = self._load_model(model_name)
        self._cache_model(model_name, model)
        return model
```

### 3. Performance Optimization

```python
class PerformanceOptimizer:
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.metrics = MetricsCollector()
    
    def optimize_extraction(self, text: str) -> List[PIIEntity]:
        with self.profiler.time("extraction"):
            # 1. Text preprocessing optimization
            chunks = self._optimal_chunking(text)
            
            # 2. Parallel processing with optimal batch size
            batch_size = self._calculate_optimal_batch_size()
            
            # 3. GPU utilization optimization
            if self._gpu_available():
                return self._gpu_optimized_extraction(chunks)
            else:
                return self._cpu_optimized_extraction(chunks)
```

This architecture document provides a comprehensive view of how all components in the PII Extraction System work together to provide a scalable, secure, and maintainable solution for PII detection and management.