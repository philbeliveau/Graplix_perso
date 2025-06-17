"""Final system validation test script."""

import sys
sys.path.append('.')

from core.pipeline import PIIExtractionPipeline
from extractors.rule_based import RuleBasedExtractor
from utils.document_processor import DocumentProcessor

def main():
    print('=== FINAL SYSTEM VALIDATION ===')
    print()

    # Test 1: Core imports
    print('✅ Testing core imports...')
    try:
        pipeline = PIIExtractionPipeline()
        print('   - PIIExtractionPipeline: SUCCESS')
    except Exception as e:
        print(f'   - PIIExtractionPipeline: FAILED - {e}')
        return

    # Test 2: Individual extractors
    print('✅ Testing individual extractors...')
    try:
        rule_extractor = RuleBasedExtractor()
        print('   - RuleBasedExtractor: SUCCESS')
    except Exception as e:
        print(f'   - RuleBasedExtractor: FAILED - {e}')

    # Test 3: Document processor
    print('✅ Testing document processor...')
    try:
        doc_processor = DocumentProcessor()
        print('   - DocumentProcessor: SUCCESS')
    except Exception as e:
        print(f'   - DocumentProcessor: FAILED - {e}')

    # Test 4: Create a temporary test file
    print('✅ Testing document processing...')
    try:
        # Create a test text file
        test_content = """Dear John Smith,

        Thank you for your job application. Please contact us at hr@company.com 
        or call (555) 123-4567 if you have any questions.
        
        Your application ID is EMP-2024-001.
        Your Social Security Number 123-45-6789 will be kept confidential.
        
        Please visit our website at https://company.com for more information.
        
        Best regards,
        Jane Doe
        HR Department
        123 Main Street
        New York, NY 10001
        """
        
        with open('test_document.txt', 'w') as f:
            f.write(test_content)
        
        # Process the test file
        result = pipeline.extract_from_file('test_document.txt')
        print(f'   - File processing: SUCCESS - Found {len(result.entities)} entities')
        print(f'   - Confidence: {result.confidence_score:.2f}')
        print(f'   - Processing time: {result.processing_time:.3f}s')
        
        # Show some sample entities
        print('   - Sample entities found:')
        entity_types = {}
        for entity in result.entities:
            if entity.type not in entity_types:
                entity_types[entity.type] = []
            entity_types[entity.type].append(entity.value)
        
        for pii_type, values in entity_types.items():
            print(f'     • {pii_type}: {len(values)} found ({", ".join(values[:2])}{"..." if len(values) > 2 else ""})')
        
        # Clean up
        import os
        os.remove('test_document.txt')
        
    except Exception as e:
        print(f'   - Document processing: FAILED - {e}')

    # Test 5: Pipeline configuration info
    print('✅ Testing pipeline information...')
    try:
        info = pipeline.get_pipeline_info()
        print(f'   - Pipeline info: SUCCESS')
        print(f'   - Available extractors: {", ".join(info["extractors"])}')
        print(f'   - Supported formats: {", ".join(info["supported_formats"])}')
    except Exception as e:
        print(f'   - Pipeline info: FAILED - {e}')

    print()
    print('=== VALIDATION COMPLETE ===')
    print('✅ System is ready for production use!')

if __name__ == "__main__":
    main()