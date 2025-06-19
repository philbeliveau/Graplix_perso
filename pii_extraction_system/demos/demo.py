#!/usr/bin/env python3
"""Demonstration script for PII extraction system."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.pipeline import PIIExtractionPipeline
from core.logging_config import get_logger
from utils.document_processor import DocumentProcessor

logger = get_logger(__name__)


def create_sample_document():
    """Create a sample document for testing."""
    sample_text = """
    CONFIDENTIAL EMPLOYEE RECORD
    
    Employee Information:
    Name: Marie Dubois
    Email: marie.dubois@company.com
    Phone: (514) 555-0123
    Employee ID: EMP001234
    
    Personal Details:
    Date of Birth: 15/03/1985
    Social Security: 123-45-6789
    Address: 123 rue Saint-Laurent, Montreal, QC H2X 2T3
    
    Emergency Contact:
    Name: Jean Dubois
    Phone: 514.555.0124
    Email: jean.dubois@gmail.com
    
    Medical Information:
    Patient ID: MRN7890123
    
    Financial Information:
    Credit Card: 4532-1234-5678-9012
    Bank Account: 123456789
    
    Additional Notes:
    Please contact Dr. Smith at dr.smith@clinic.ca or visit www.company-portal.com
    IP Address for system access: 192.168.1.100
    """
    
    # Save sample document
    sample_file = Path("sample_document.txt")
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    return sample_file


def demonstrate_document_processing():
    """Demonstrate document processing capabilities."""
    print("🔍 Document Processing Demonstration")
    print("=" * 50)
    
    # Create sample document
    sample_file = create_sample_document()
    print(f"Created sample document: {sample_file}")
    
    try:
        # Initialize document processor
        processor = DocumentProcessor()
        
        # Process the document
        print("\\nProcessing document...")
        result = processor.process_document(sample_file)
        
        print(f"✅ Document processed successfully!")
        print(f"   - File type: {result['file_type']}")
        print(f"   - File size: {result['file_size_mb']} MB")
        print(f"   - Text length: {len(result['raw_text'])} characters")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing document: {e}")
        return None
    
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()


def demonstrate_pii_extraction():
    """Demonstrate PII extraction capabilities."""
    print("\\n🔐 PII Extraction Demonstration")
    print("=" * 50)
    
    # Create sample document
    sample_file = create_sample_document()
    
    try:
        # Initialize pipeline with local data source
        pipeline = PIIExtractionPipeline(
            data_source="local",
            models=["rule_based"]
        )
        
        print("\\nExtracting PII from document...")
        result = pipeline.extract_from_file(sample_file, save_results=False)
        
        print(f"✅ PII extraction completed!")
        print(f"   - Processing time: {result.processing_time:.3f} seconds")
        print(f"   - Total entities found: {len(result.pii_entities)}")
        
        # Display statistics
        stats = result.get_statistics()
        print(f"   - Unique PII types: {stats['unique_types']}")
        print(f"   - Average confidence: {stats['avg_confidence']:.2f}")
        
        # Display found entities by type
        print("\\n📋 Found PII Entities:")
        print("-" * 30)
        
        entities_by_type = {}
        for entity in result.pii_entities:
            if entity.pii_type not in entities_by_type:
                entities_by_type[entity.pii_type] = []
            entities_by_type[entity.pii_type].append(entity)
        
        for pii_type, entities in sorted(entities_by_type.items()):
            print(f"\\n{pii_type.replace('_', ' ').title()}:")
            for entity in entities:
                print(f"  • {entity.text} (confidence: {entity.confidence:.2f})")
        
        # Display type distribution
        print("\\n📊 PII Type Distribution:")
        print("-" * 30)
        for pii_type, count in stats['type_distribution'].items():
            print(f"  {pii_type.replace('_', ' ').title()}: {count}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during PII extraction: {e}")
        return None
    
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()


def demonstrate_pipeline_info():
    """Demonstrate pipeline information retrieval."""
    print("\\n⚙️  Pipeline Configuration")
    print("=" * 50)
    
    try:
        pipeline = PIIExtractionPipeline()
        info = pipeline.get_pipeline_info()
        
        print("Pipeline Information:")
        print(f"  - Data source: {info['data_source']}")
        print(f"  - Enabled models: {', '.join(info['enabled_models'])}")
        print(f"  - Available extractors: {', '.join(info['extractors'])}")
        print(f"  - Supported formats: {', '.join(info['supported_formats'])}")
        
        print("\\nPrivacy Settings:")
        privacy = info['privacy_settings']
        print(f"  - Redaction enabled: {privacy['redaction_enabled']}")
        print(f"  - GDPR compliance: {privacy['gdpr_compliance']}")
        print(f"  - Law 25 compliance: {privacy['law25_compliance']}")
        
    except Exception as e:
        print(f"❌ Error getting pipeline info: {e}")


def main():
    """Main demonstration function."""
    print("🚀 PII Extraction System Demonstration")
    print("=" * 60)
    
    # Demonstrate document processing
    doc_result = demonstrate_document_processing()
    
    # Demonstrate PII extraction
    pii_result = demonstrate_pii_extraction()
    
    # Demonstrate pipeline information
    demonstrate_pipeline_info()
    
    print("\\n✨ Demonstration completed!")
    print("\\n📚 Next Steps:")
    print("  1. Install dependencies: poetry install")
    print("  2. Run tests: poetry run pytest")
    print("  3. Start dashboard: streamlit run src/dashboard/app.py")
    print("  4. Process real documents using the pipeline")


if __name__ == "__main__":
    main()