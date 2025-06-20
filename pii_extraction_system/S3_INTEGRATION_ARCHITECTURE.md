# S3 Integration for Automated Phase 0 Dataset Creation

## Overview

The platform can be enhanced to connect directly to S3 buckets for automated document processing and labeled dataset creation. This would enable enterprise-scale processing without manual uploads.

## Current Architecture vs S3 Integration

### **Current: Manual Upload Architecture**
```
User → Streamlit Upload → Base64 Storage → Vision Processing → Phase 0 Dataset
```

### **Enhanced: S3 Integration Architecture**
```
S3 Bucket → Auto Discovery → Batch Processing → Vision Processing → Labeled Dataset → S3 Export
```

## Technical Implementation

### 1. **S3 Connection Layer**

The platform already has basic S3 configuration in `core/config.py`:

```python
class DataSourceConfig(BaseModel):
    source_type: str = Field(default="local", description="Data source: 'local' or 's3'")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_region: str = Field(default="us-west-2", description="S3 region")
```

**Enhanced S3 Configuration:**
```python
class S3Config(BaseModel):
    # Connection
    bucket_name: str = Field(description="S3 bucket name")
    region: str = Field(default="us-west-2", description="AWS region")
    access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    
    # Processing
    input_prefix: str = Field(default="documents/", description="S3 prefix for input documents")
    output_prefix: str = Field(default="labeled-datasets/", description="S3 prefix for output datasets")
    processed_prefix: str = Field(default="processed/", description="S3 prefix for processed documents")
    
    # Batch Processing
    batch_size: int = Field(default=100, description="Documents per batch")
    max_file_size_mb: int = Field(default=50, description="Maximum file size")
    supported_formats: List[str] = Field(
        default=["pdf", "docx", "xlsx", "png", "jpg", "jpeg"],
        description="Supported file formats"
    )
    
    # Monitoring
    enable_sns_notifications: bool = Field(default=False, description="Enable SNS notifications")
    sns_topic_arn: Optional[str] = Field(default=None, description="SNS topic for notifications")
```

### 2. **S3 Document Discovery Service**

```python
class S3DocumentDiscovery:
    """Discovers and processes documents from S3 bucket"""
    
    def __init__(self, s3_config: S3Config):
        self.s3_config = s3_config
        self.s3_client = boto3.client(
            's3',
            region_name=s3_config.region,
            aws_access_key_id=s3_config.access_key_id,
            aws_secret_access_key=s3_config.secret_access_key
        )
    
    def discover_new_documents(self) -> List[S3Document]:
        """Discover unprocessed documents in S3 bucket"""
        documents = []
        
        # List objects in input prefix
        paginator = self.s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(
            Bucket=self.s3_config.bucket_name,
            Prefix=self.s3_config.input_prefix
        )
        
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                
                # Check if already processed
                if not self._is_processed(key):
                    # Check file format
                    if self._is_supported_format(key):
                        documents.append(S3Document(
                            bucket=self.s3_config.bucket_name,
                            key=key,
                            size=obj['Size'],
                            last_modified=obj['LastModified']
                        ))
        
        return documents
    
    def _is_processed(self, key: str) -> bool:
        """Check if document already processed"""
        processed_key = key.replace(
            self.s3_config.input_prefix,
            self.s3_config.processed_prefix
        )
        
        try:
            self.s3_client.head_object(
                Bucket=self.s3_config.bucket_name,
                Key=processed_key + '.processed'
            )
            return True
        except:
            return False
```

### 3. **Automated Batch Processing Pipeline**

```python
class S3BatchProcessor:
    """Processes S3 documents in batches for Phase 0 dataset creation"""
    
    def __init__(self, s3_config: S3Config, llm_service: MultimodalLLMService):
        self.s3_config = s3_config
        self.llm_service = llm_service
        self.discovery = S3DocumentDiscovery(s3_config)
        self.cost_tracker = CostTracker()
    
    async def process_bucket(self, password: str = "Hubert") -> BatchProcessingResult:
        """Process all unprocessed documents in S3 bucket"""
        
        # Discover documents
        documents = self.discovery.discover_new_documents()
        logger.info(f"Found {len(documents)} unprocessed documents")
        
        # Process in batches
        results = []
        total_cost = 0.0
        
        for batch_start in range(0, len(documents), self.s3_config.batch_size):
            batch_end = min(batch_start + self.s3_config.batch_size, len(documents))
            batch = documents[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//self.s3_config.batch_size + 1}")
            
            batch_result = await self._process_batch(batch, password)
            results.extend(batch_result.documents)
            total_cost += batch_result.total_cost
            
            # Budget check
            if total_cost > self.s3_config.max_budget:
                logger.warning(f"Budget limit reached: ${total_cost}")
                break
        
        # Create final dataset
        dataset = self._create_labeled_dataset(results)
        
        # Export to S3
        dataset_key = self._export_dataset_to_s3(dataset)
        
        return BatchProcessingResult(
            documents_processed=len(results),
            total_cost=total_cost,
            dataset_s3_key=dataset_key,
            success_rate=len([r for r in results if r.success]) / len(results)
        )
    
    async def _process_batch(self, documents: List[S3Document], password: str) -> BatchResult:
        """Process a batch of documents"""
        results = []
        
        # Download and process documents in parallel
        async with asyncio.Semaphore(5):  # Limit concurrent processing
            tasks = [
                self._process_single_document(doc, password)
                for doc in documents
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return BatchResult(
            documents=results,
            total_cost=sum(r.cost for r in results if not isinstance(r, Exception))
        )
    
    async def _process_single_document(self, document: S3Document, password: str) -> DocumentResult:
        """Process a single S3 document"""
        try:
            # Download document
            file_content = self._download_document(document)
            
            # Convert to images
            images = convert_document_to_images(
                file_content, 
                Path(document.key).suffix, 
                password
            )
            
            if not images:
                return DocumentResult(
                    key=document.key,
                    success=False,
                    error="Could not convert to images"
                )
            
            # Process with LLM
            all_entities = []
            total_cost = 0.0
            classification = None
            
            for image_data in images:
                result = self.llm_service.extract_pii_from_image(
                    image_data,
                    'openai/gpt-4o-mini',  # Use cost-effective model for batch
                    document_type="business_document"
                )
                
                if result.get('success'):
                    all_entities.extend(result.get('pii_entities', []))
                    total_cost += result.get('usage', {}).get('estimated_cost', 0)
                    
                    # Get classification from first page
                    if not classification and result.get('structured_data'):
                        classification = result['structured_data'].get('document_classification')
            
            # Mark as processed
            self._mark_as_processed(document)
            
            return DocumentResult(
                key=document.key,
                success=True,
                entities=all_entities,
                classification=classification,
                cost=total_cost
            )
            
        except Exception as e:
            logger.error(f"Failed to process {document.key}: {e}")
            return DocumentResult(
                key=document.key,
                success=False,
                error=str(e)
            )
```

### 4. **Dashboard Integration**

**Enhanced Phase 0 with S3 Support:**

```python
def show_s3_batch_processing():
    """S3 batch processing interface in Phase 0"""
    st.markdown("### 🗄️ S3 Batch Processing")
    
    # S3 Configuration
    with st.expander("S3 Configuration"):
        bucket_name = st.text_input("S3 Bucket Name")
        input_prefix = st.text_input("Input Prefix", value="documents/")
        password = st.text_input("Document Password", value="Hubert", type="password")
        
        # Model selection for batch processing
        model_options = ["openai/gpt-4o-mini", "openai/gpt-4o", "deepseek/deepseek-chat"]
        selected_model = st.selectbox("Processing Model", model_options)
        
        # Budget controls
        max_budget = st.number_input("Maximum Budget ($)", value=50.0, min_value=1.0)
        batch_size = st.slider("Batch Size", 10, 200, 100)
    
    if st.button("Start S3 Batch Processing"):
        if bucket_name:
            # Initialize S3 processor
            s3_config = S3Config(
                bucket_name=bucket_name,
                input_prefix=input_prefix,
                batch_size=batch_size,
                max_budget=max_budget
            )
            
            processor = S3BatchProcessor(s3_config, llm_service)
            
            # Process with progress tracking
            with st.spinner("Processing S3 documents..."):
                result = asyncio.run(processor.process_bucket(password))
            
            # Show results
            st.success(f"Processed {result.documents_processed} documents")
            st.metric("Total Cost", f"${result.total_cost:.2f}")
            st.metric("Success Rate", f"{result.success_rate:.1%}")
            st.info(f"Dataset exported to: {result.dataset_s3_key}")
            
            # Import into Phase 0 session
            if st.button("Import to Phase 0 Dataset"):
                import_s3_dataset_to_phase0(result.dataset_s3_key)
                st.success("S3 dataset imported to Phase 0!")
```

### 5. **Automated Scheduling**

**Cron Job / Lambda Integration:**

```python
class S3ProcessingScheduler:
    """Scheduled S3 document processing"""
    
    def __init__(self, s3_config: S3Config):
        self.s3_config = s3_config
        self.processor = S3BatchProcessor(s3_config, llm_service)
    
    def scheduled_processing(self):
        """Run scheduled processing (called by cron/Lambda)"""
        try:
            logger.info("Starting scheduled S3 processing")
            
            # Process bucket
            result = asyncio.run(self.processor.process_bucket())
            
            # Send notification
            if self.s3_config.enable_sns_notifications:
                self._send_notification(result)
            
            logger.info(f"Scheduled processing complete: {result.documents_processed} documents")
            
        except Exception as e:
            logger.error(f"Scheduled processing failed: {e}")
            self._send_error_notification(str(e))
```

## Benefits

### **1. Enterprise Scale**
- **Automated Processing**: No manual uploads required
- **Batch Efficiency**: Process thousands of documents automatically
- **Scheduled Runs**: Daily/weekly processing schedules
- **Cost Control**: Budget limits and monitoring

### **2. Integration Benefits**
- **Existing Workflows**: Integrate with document management systems
- **Data Governance**: Maintains S3 access controls and permissions
- **Audit Trail**: Complete processing history in S3
- **Compliance**: Supports enterprise security requirements

### **3. Operational Benefits**
- **Scalability**: Handle enterprise document volumes
- **Reliability**: Retry logic and error handling
- **Monitoring**: SNS notifications and CloudWatch integration
- **Cost Optimization**: Efficient model usage for batch processing

## Deployment Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   S3 Bucket     │    │  Lambda/ECS     │    │   Dashboard     │
│   (Documents)   │───▶│   Processor     │───▶│   (Results)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Processed/     │    │   CloudWatch    │    │   S3 Datasets   │
│  Archive        │    │   Monitoring    │    │   (Labeled)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Configuration Example

```yaml
# s3_processing_config.yaml
s3:
  bucket_name: "company-documents"
  input_prefix: "incoming/"
  output_prefix: "labeled-datasets/"
  processed_prefix: "processed/"
  
processing:
  model: "openai/gpt-4o-mini"  # Cost-effective for batch
  batch_size: 100
  max_budget_daily: 100.0
  password: "Hubert"
  
scheduling:
  enabled: true
  cron: "0 2 * * *"  # Daily at 2 AM
  
notifications:
  sns_topic: "arn:aws:sns:us-west-2:123456789:pii-processing"
  slack_webhook: "https://hooks.slack.com/..."
```

## Implementation Timeline

**Phase 1 (2-3 weeks)**: Basic S3 integration
- S3 document discovery and download
- Batch processing framework
- Dashboard integration

**Phase 2 (2-3 weeks)**: Advanced features
- Automated scheduling (Lambda/ECS)
- Monitoring and notifications
- Error handling and retry logic

**Phase 3 (1-2 weeks)**: Enterprise features
- Advanced configuration options
- Performance optimization
- Documentation and deployment guides

This S3 integration would transform the platform from a manual document processing tool into an enterprise-scale automated PII extraction and dataset labeling system.