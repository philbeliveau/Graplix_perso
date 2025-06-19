#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from core.pipeline import PIIExtractionPipeline

def main():
    # Initialize pipeline
    pipeline = PIIExtractionPipeline(data_source='local', models=['rule_based'])

    # Test with the text document
    test_file = '../data/test_pii_document.txt'
    print(f'Testing with: {test_file}')

    try:
        result = pipeline.extract_from_file(test_file, save_results=False)
        
        print(f'\nExtraction completed!')
        print(f'Found {len(result.pii_entities)} PII entities')
        print(f'Processing time: {result.processing_time:.3f} seconds')
        
        # Display found entities
        for i, entity in enumerate(result.pii_entities):
            print(f'{i+1}. Type: {entity.pii_type}, Text: "{entity.text}", Confidence: {entity.confidence:.2f}')
            
        print(f'\nExtraction statistics:')
        stats = result.get_statistics()
        for key, value in stats.items():
            if key != 'type_distribution':
                print(f'  {key}: {value}')
        
        print(f'\nPII types found:')
        for pii_type, count in stats['type_distribution'].items():
            print(f'  {pii_type}: {count}')
        
    except Exception as e:
        print(f'Error during extraction: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()