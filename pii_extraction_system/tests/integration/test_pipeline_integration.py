"""Integration tests for the complete PII extraction pipeline."""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

from src.core.pipeline import PIIExtractionPipeline
from src.core.config import PIIConfig
from src.utils.document_processor import DocumentProcessor
from src.utils.data_storage import DataStorageManager


class TestPipelineIntegration:
    """Integration tests for the complete PII extraction pipeline."""
    
    @classmethod
    def setup_class(cls):
        """Set up integration test environment."""
        # Create temporary directories for testing
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.test_data_dir = cls.temp_dir / "test_data"
        cls.test_output_dir = cls.temp_dir / "test_output"
        
        cls.test_data_dir.mkdir(exist_ok=True)
        cls.test_output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        cls.config = PIIConfig()
        cls.config.output_dir = str(cls.test_output_dir)
        cls.config.use_local_storage = True
        
        cls.pipeline = PIIExtractionPipeline(cls.config)
        cls.doc_processor = DocumentProcessor(cls.config)
        cls.storage_manager = DataStorageManager(cls.config)
        
        # Create test documents
        cls._create_test_documents()
    
    @classmethod
    def _create_test_documents(cls):
        """Create test documents for integration testing."""
        # Text document
        text_content = """
        Employee Information System
        
        Name: Alice Johnson
        Email: alice.johnson@techcorp.com
        Phone: (416) 555-0123
        Employee ID: EMP-001234
        Department: Engineering
        Start Date: 2022-01-15
        Address: 456 Bay Street, Toronto, ON M5H 2Y4
        Emergency Contact: Bob Johnson at (416) 555-0456
        
        Banking Information:
        Account: 1234567890
        Transit: 12345
        Institution: 001
        
        Security Clearance: Level 2
        Badge Number: B789456
        """
        
        (cls.test_data_dir / "employee_info.txt").write_text(text_content)
        
        # Create a simple test image with text (would need OCR in real scenario)
        # For now, we'll create a metadata file that simulates OCR results
        ocr_content = {
            "file_path": str(cls.test_data_dir / "id_card.jpg"),
            "ocr_text": "Driver License\nName: Sarah Wilson\nLicense: D123456789\nDOB: 03/22/1988\nAddress: 789 Queen St W",
            "confidence": 0.92
        }
        
        with open(cls.test_data_dir / "id_card_ocr.json", 'w') as f:
            json.dump(ocr_content, f)
    
    def test_complete_pipeline_processing(self):
        """Test complete pipeline from document input to PII output."""
        # Process text document
        input_file = self.test_data_dir / "employee_info.txt"
        
        # Run complete pipeline
        results = self.pipeline.process_document(str(input_file))
        
        # Validate results
        assert results is not None
        assert hasattr(results, 'pii_entities')
        assert len(results.pii_entities) > 0
        
        # Check for expected PII types
        found_types = set(entity.pii_type for entity in results.pii_entities)
        expected_types = {'email_address', 'phone_number', 'person_name', 'employee_id', 'address'}
        
        # Should find most expected types
        intersection = found_types.intersection(expected_types)
        assert len(intersection) >= len(expected_types) * 0.6
        
        # Validate processing metadata
        assert results.processing_time > 0
        assert results.metadata is not None
        assert 'extractor' in results.metadata
    
    def test_batch_processing(self):
        """Test batch processing of multiple documents."""
        input_files = [
            str(self.test_data_dir / "employee_info.txt"),
        ]
        
        # Process batch
        batch_results = self.pipeline.process_batch(input_files)
        
        # Validate batch results
        assert isinstance(batch_results, list)
        assert len(batch_results) == len(input_files)
        
        for result in batch_results:
            assert result is not None
            assert hasattr(result, 'pii_entities')
            assert hasattr(result, 'processing_time')
    
    def test_document_processor_integration(self):
        """Test document processor integration with different file types."""
        # Test text file processing
        text_file = self.test_data_dir / "employee_info.txt"
        
        processed_doc = self.doc_processor.process_document(text_file)
        
        assert processed_doc is not None
        assert 'raw_text' in processed_doc
        assert 'file_type' in processed_doc
        assert 'file_path' in processed_doc
        assert len(processed_doc['raw_text']) > 0
        assert processed_doc['file_type'] == '.txt'
    
    def test_storage_manager_integration(self):
        """Test storage manager integration for saving/loading results."""
        # Create sample results
        sample_results = {
            'file_path': str(self.test_data_dir / "employee_info.txt"),
            'entities_found': 10,
            'processing_time': 1.23,
            'pii_types': ['email_address', 'phone_number', 'person_name']
        }
        
        # Test saving results
        result_id = self.storage_manager.save_results(sample_results)
        assert result_id is not None
        
        # Test loading results
        loaded_results = self.storage_manager.load_results(result_id)
        assert loaded_results is not None
        assert loaded_results['entities_found'] == sample_results['entities_found']
    
    def test_configuration_integration(self):
        """Test configuration integration across components."""
        # Test that all components use the same configuration
        assert self.pipeline.config == self.config
        assert self.doc_processor.config == self.config
        assert self.storage_manager.config == self.config
        
        # Test configuration updates propagate
        original_threshold = self.config.confidence_threshold
        self.config.confidence_threshold = 0.9
        
        # Components should reflect the updated config
        assert self.pipeline.config.confidence_threshold == 0.9
        
        # Restore original value
        self.config.confidence_threshold = original_threshold
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test with non-existent file
        non_existent_file = str(self.test_data_dir / "does_not_exist.txt")
        
        result = self.pipeline.process_document(non_existent_file)
        
        # Should handle error gracefully
        assert result is not None
        assert hasattr(result, 'error')
        assert result.error is not None
        assert 'not found' in result.error.lower() or 'exist' in result.error.lower()
    
    def test_performance_integration(self):
        """Test performance characteristics of integrated pipeline."""
        input_file = str(self.test_data_dir / "employee_info.txt")
        
        # Run multiple times to get average performance
        processing_times = []
        
        for _ in range(5):
            result = self.pipeline.process_document(input_file)
            processing_times.append(result.processing_time)
        
        # Calculate performance metrics
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        
        # Performance assertions
        assert avg_time < 5.0, f"Average processing time too high: {avg_time:.3f}s"
        assert max_time < 10.0, f"Maximum processing time too high: {max_time:.3f}s"
        
        # Consistency check - times shouldn't vary too much
        time_variance = max(processing_times) - min(processing_times)
        assert time_variance < avg_time * 2, "Processing times too inconsistent"
    
    def test_memory_usage_integration(self):
        """Test memory usage of integrated pipeline."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple documents
        input_file = str(self.test_data_dir / "employee_info.txt")
        
        for _ in range(10):
            result = self.pipeline.process_document(input_file)
            assert result is not None
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500, f"Memory usage increased too much: {memory_increase:.1f}MB"
    
    def test_concurrent_processing_integration(self):
        """Test concurrent processing capabilities."""
        import threading
        import time
        
        input_file = str(self.test_data_dir / "employee_info.txt")
        results = {}
        errors = {}
        
        def process_document(thread_id):
            try:
                result = self.pipeline.process_document(input_file)
                results[thread_id] = result
            except Exception as e:
                errors[thread_id] = str(e)
        
        # Create multiple threads
        threads = []
        num_threads = 3
        
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=process_document, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Validate results
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == num_threads
        
        # All results should be valid
        for thread_id, result in results.items():
            assert result is not None
            assert hasattr(result, 'pii_entities')
            assert len(result.pii_entities) > 0
        
        # Concurrent processing shouldn't take much longer than sequential
        total_time = end_time - start_time
        assert total_time < 15.0, f"Concurrent processing took too long: {total_time:.2f}s"
    
    def test_data_flow_integration(self):
        """Test complete data flow from input to output storage."""
        input_file = str(self.test_data_dir / "employee_info.txt")
        
        # Process document
        result = self.pipeline.process_document(input_file)
        
        # Convert result to dictionary for storage
        result_dict = {
            'file_path': input_file,
            'entities': [
                {
                    'text': entity.text,
                    'type': entity.pii_type,
                    'confidence': entity.confidence,
                    'start_pos': entity.start_pos,
                    'end_pos': entity.end_pos
                }
                for entity in result.pii_entities
            ],
            'processing_time': result.processing_time,
            'metadata': result.metadata
        }
        
        # Store results
        result_id = self.storage_manager.save_results(result_dict)
        
        # Retrieve and validate
        loaded_result = self.storage_manager.load_results(result_id)
        
        assert loaded_result is not None
        assert loaded_result['file_path'] == input_file
        assert len(loaded_result['entities']) == len(result.pii_entities)
        assert loaded_result['processing_time'] == result.processing_time
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])