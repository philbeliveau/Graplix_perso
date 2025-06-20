Metadata Tagging System Implementation

  Based on my analysis of the codebase, here's how the metadata tagging system is implemented:

  🏗️ Core Architecture

  1. MetadataManager (metadata_manager.py)
  - Central hub for all metadata operations
  - Hierarchical tag structure with parent-child relationships
  - JSON-based storage with in-memory caching
  - Schema validation for data quality

  2. DocumentMetadata Structure
  @dataclass
  class DocumentMetadata:
      document_id: str
      file_path: str
      file_name: str
      file_extension: str
      file_size_bytes: int
      file_hash: str

      # Processing metadata
      processing_status: str
      processing_time: Optional[float]
      content_type: str
      estimated_pages: int
      estimated_words: int
      language: str

      # Tagging system
      tags: List[str]
      custom_attributes: Dict

      # Ground truth integration
      is_ground_truth: bool
      ground_truth_category: Optional[str]

  🏷️ Tag System Structure

  Tag Types:
  - SYSTEM: Core functionality tags
  - USER: User-created organizational tags
  - AUTO_GENERATED: AI-generated based on document analysis
  - QUALITY: Document quality metrics
  - PROCESSING: Processing status and results
  - CONTENT: Document content characteristics

  Tag Scopes:
  - DOCUMENT: Individual document level
  - BATCH: Processing batch level
  - EXPERIMENT: ML experiment level
  - DATASET: Entire dataset level

  🤖 Automatic Tag Generation

  The system automatically tags documents based on:

  def auto_tag_document(self, document_metadata: DocumentMetadata):
      # File format: ".pdf", ".docx", etc.
      # Content type: "form", "resume", "medical_record"
      # Size category: "small", "medium", "large" 
      # Processing status: "pending", "completed", "failed"
      # Ground truth status: "verified", "unverified"

  💾 Storage Architecture

  /memory/metadata/
  ├── tags.json              # All tag definitions
  ├── document_tags.json     # Document-tag mappings
  ├── tag_index.json         # Tag-to-document reverse index
  └── schemas/               # Custom validation schemas
      ├── default_document.json
      └── custom_schemas.json

  🔗 Pipeline Integration

  1. Document Processing
  - Auto-tags during document ingestion
  - Tracks processing status and metrics
  - Links to ground truth data

  2. Batch Processing
  - Tags entire batches with results
  - Aggregates quality metrics
  - Enables filtering by processing status

  3. Experiment Tracking
  - Tags experiments with configuration
  - Tracks dataset composition
  - Enables performance analysis by document type

  📊 Usage Patterns

  Document Discovery:
  # Find PDFs
  pdf_docs = meta_mgr.get_documents_by_tag("format", ".pdf")

  # Get pending documents
  pending = meta_mgr.get_documents_by_tag("processing_status", "pending")

  # Find high-quality documents
  quality_docs = meta_mgr.get_documents_by_tag("quality_score", "high")

  Quality Tracking:
  # Auto-tag with quality metrics
  auto_tags = meta_mgr.auto_tag_document(doc_metadata)

  # Manual quality tags
  meta_mgr.apply_tag_by_name(doc_id, "review_status", "approved")

  🎯 Key Features

  1. Hierarchical Organization: Tags can have parent-child relationships
  2. Schema Validation: Custom schemas ensure metadata consistency
  3. Performance Optimization: Caching and indexing for fast retrieval
  4. Automatic Enrichment: AI-powered tag generation
  5. Multi-Level Tagging: Document, batch, experiment, and dataset scopes
  6. Ground Truth Integration: Links metadata to verified annotations

  🚀 Benefits

  - Searchability: Easy document discovery and filtering
  - Quality Control: Automated quality metrics and validation
  - Batch Management: Track processing status across document sets
  - Experiment Reproducibility: Metadata versioning and snapshots
  - Performance Analytics: Analysis by document characteristics

  The metadata tagging system provides a robust foundation for organizing and analyzing documents throughout the PII extraction pipeline, with strong integration across all system
  components.
