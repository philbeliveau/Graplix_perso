"""
Demo Usage Script for Data Pipeline System

This script demonstrates how to use the comprehensive data pipeline
for document processing, batch operations, and metadata management.
"""

import sys
from pathlib import Path

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data_pipeline import (
    DataLoader, BatchProcessor, ExperimentTracker, 
    MetadataManager, GroundTruthManager,
    ExperimentConfiguration, ExperimentType, DatasetType
)
from core.logging_config import get_logger

logger = get_logger(__name__)

def demo_data_loading():
    """Demonstrate basic data loading capabilities."""
    print("\n=== Data Loading Demo ===")
    
    # Initialize data loader
    loader = DataLoader()
    
    # Discover documents
    documents = loader.discover_documents(recursive=True)
    print(f"Discovered {len(documents)} documents")
    
    # Load metadata for first few documents
    sample_docs = documents[:3] if len(documents) >= 3 else documents
    metadata_list = loader.load_documents_metadata(sample_docs)
    
    for metadata in metadata_list:
        print(f"- {metadata.file_name}: {metadata.file_extension}, "
              f"{metadata.file_size_bytes} bytes, {metadata.content_type}")
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"Statistics: {stats['total_documents']} docs, "
          f"{stats['total_size_mb']} MB total")
    
    return loader, documents

def demo_batch_processing(loader, documents):
    """Demonstrate batch processing capabilities."""
    print("\n=== Batch Processing Demo ===")
    
    # Initialize batch processor
    processor = BatchProcessor(max_workers=2)
    
    # Process a small subset for demo
    sample_docs = documents[:2] if len(documents) >= 2 else documents
    
    if sample_docs:
        print(f"Processing {len(sample_docs)} documents...")
        
        # Process documents
        batch_result = processor.process_documents(sample_docs)
        
        print(f"Batch completed: {batch_result.status.successful_documents}/"
              f"{batch_result.status.total_documents} successful")
        print(f"Total processing time: {batch_result.total_processing_time:.2f}s")
        print(f"Average time per document: {batch_result.average_processing_time:.2f}s")
        
        return batch_result
    else:
        print("No documents to process")
        return None

def demo_experiment_tracking(batch_result):
    """Demonstrate experiment tracking capabilities."""
    print("\n=== Experiment Tracking Demo ===")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker()
    
    # Create experiment configuration
    config = ExperimentConfiguration(
        experiment_name="Demo Pipeline Test",
        experiment_type=ExperimentType.BASELINE,
        description="Demonstration of basic pipeline functionality",
        pipeline_config={
            "max_workers": 2,
            "formats_enabled": ["pdf", "txt", "jpg", "xlsx", "docx"]
        }
    )
    
    # Create and start experiment
    experiment = tracker.create_experiment(config)
    tracker.start_experiment(experiment.experiment_id)
    
    print(f"Created experiment: {experiment.experiment_id}")
    
    # Log batch result if available
    if batch_result:
        tracker.log_batch_result(experiment.experiment_id, batch_result)
        print("Logged batch processing results")
    
    # Complete experiment
    tracker.complete_experiment(experiment.experiment_id)
    
    # Get experiment statistics
    stats = tracker.get_experiment_statistics()
    print(f"Total experiments: {stats['total_experiments']}")
    
    return tracker, experiment

def demo_metadata_management(loader, documents):
    """Demonstrate metadata management capabilities."""
    print("\n=== Metadata Management Demo ===")
    
    # Initialize metadata manager
    meta_mgr = MetadataManager()
    
    # Load some document metadata
    if documents:
        sample_doc = documents[0]
        doc_metadata = loader.load_document_metadata(sample_doc)
        
        # Auto-tag the document
        auto_tags = meta_mgr.auto_tag_document(doc_metadata)
        print(f"Auto-generated {len(auto_tags)} tags for {doc_metadata.file_name}")
        
        # Add custom tags
        meta_mgr.apply_tag_by_name(doc_metadata.document_id, "demo_processed")
        meta_mgr.apply_tag_by_name(doc_metadata.document_id, "priority", "high")
        
        # Get document tags
        doc_tags = meta_mgr.get_document_tags(doc_metadata.document_id)
        print(f"Document tags: {[tag.tag_name for tag in doc_tags]}")
        
        # Get documents by tag
        pdf_docs = meta_mgr.get_documents_by_tag("format", ".pdf")
        print(f"Found {len(pdf_docs)} PDF documents")
    
    # Get tag statistics
    tag_stats = meta_mgr.get_tag_statistics()
    print(f"Tag statistics: {tag_stats['total_tags']} total tags, "
          f"{tag_stats['total_unique_names']} unique names")
    
    return meta_mgr

def demo_ground_truth_management(loader, documents):
    """Demonstrate ground truth management capabilities."""
    print("\n=== Ground Truth Management Demo ===")
    
    # Initialize ground truth manager
    gt_mgr = GroundTruthManager(loader)
    
    # Create ground truth entries for sample documents
    if documents:
        sample_doc = documents[0]
        doc_metadata = loader.load_document_metadata(sample_doc)
        
        # Create ground truth entry
        gt_entry = gt_mgr.create_ground_truth_entry(
            doc_metadata, 
            DatasetType.TRAINING,
            dataset_version="demo_v1.0"
        )
        
        print(f"Created ground truth entry: {gt_entry.entry_id}")
        
        # Add sample annotation
        sample_pii_entities = [
            {
                "text": "John Doe",
                "type": "person_name",
                "start_pos": 0,
                "end_pos": 8,
                "confidence": 0.95
            }
        ]
        
        annotation = gt_mgr.add_annotation(
            gt_entry.entry_id,
            annotator_id="demo_annotator",
            pii_entities=sample_pii_entities,
            document_labels=["contains_pii"]
        )
        
        print(f"Added annotation: {annotation.annotation_id}")
    
    # Get dataset statistics
    dataset_stats = gt_mgr.get_dataset_statistics("demo_v1.0")
    print(f"Dataset statistics: {dataset_stats}")
    
    return gt_mgr

def main():
    """Run all demos."""
    print("Data Pipeline System - Comprehensive Demo")
    print("=" * 50)
    
    try:
        # 1. Data Loading
        loader, documents = demo_data_loading()
        
        # 2. Batch Processing
        batch_result = demo_batch_processing(loader, documents)
        
        # 3. Experiment Tracking
        tracker, experiment = demo_experiment_tracking(batch_result)
        
        # 4. Metadata Management
        meta_mgr = demo_metadata_management(loader, documents)
        
        # 5. Ground Truth Management
        gt_mgr = demo_ground_truth_management(loader, documents)
        
        print("\n=== Demo Completed Successfully ===")
        print("All pipeline components demonstrated:")
        print("✓ Data Loading and Discovery")
        print("✓ Multi-format Document Processing")
        print("✓ Batch Processing with Parallel Execution")
        print("✓ Experiment Tracking and Results Management")
        print("✓ Metadata Tagging and Organization")
        print("✓ Ground Truth Dataset Management")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Demo failed with error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)