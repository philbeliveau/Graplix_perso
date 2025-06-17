# Performance tests for PII Extraction Pipeline
# Agent 5: DevOps & CI/CD Specialist

import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock

# Import pytest-benchmark for performance testing
pytest_benchmark = pytest.importorskip("pytest_benchmark")


class TestPipelinePerformance:
    """Performance tests for the PII extraction pipeline."""
    
    @pytest.fixture
    def large_document_text(self):
        """Generate large document text for performance testing."""
        base_text = """
        John Doe is a software engineer at TechCorp Inc.
        His email is john.doe@techcorp.com and phone number is +1-555-123-4567.
        Social Security Number: 123-45-6789
        Address: 123 Main Street, Anytown, ST 12345
        Date of Birth: January 15, 1985
        """
        # Repeat text to create larger document
        return base_text * 100
    
    @pytest.mark.performance
    def test_rule_based_extractor_performance(self, benchmark, large_document_text):
        """Test performance of rule-based extractor on large documents."""
        from src.extractors.rule_based import RuleBasedExtractor
        
        extractor = RuleBasedExtractor()
        
        def extract_pii():
            return extractor.extract(large_document_text)
        
        result = benchmark(extract_pii)
        
        # Performance assertions
        assert len(result.entities) > 0
        assert benchmark.stats.stats.mean < 1.0  # Should complete in under 1 second
    
    @pytest.mark.performance 
    def test_document_processing_performance(self, benchmark, temp_dir):
        """Test performance of document processing utilities."""
        from src.utils.document_processor import DocumentProcessor
        
        # Create test document
        test_file = temp_dir / "large_test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content with PII: john@email.com, +1-555-1234\n" * 1000)
        
        processor = DocumentProcessor()
        
        def process_document():
            return processor.process_document(str(test_file))
        
        result = benchmark(process_document)
        
        # Performance assertions
        assert result is not None
        assert benchmark.stats.stats.mean < 2.0  # Should complete in under 2 seconds
    
    @pytest.mark.performance
    def test_pipeline_throughput(self, benchmark):
        """Test overall pipeline throughput with multiple documents."""
        from src.core.pipeline import PIIExtractionPipeline
        
        # Mock configuration
        config = {
            "extractors": ["rule_based"],
            "storage": {"type": "local"},
            "processing": {"batch_size": 10}
        }
        
        pipeline = PIIExtractionPipeline(config)
        
        # Test data - multiple small documents
        documents = [
            f"Document {i}: john{i}@email.com, phone: +1-555-{i:04d}"
            for i in range(50)
        ]
        
        def process_batch():
            results = []
            for doc in documents:
                result = pipeline.process_text(doc)
                results.append(result)
            return results
        
        results = benchmark(process_batch)
        
        # Performance assertions
        assert len(results) == 50
        documents_per_second = 50 / benchmark.stats.stats.mean
        assert documents_per_second > 10  # Should process at least 10 docs/second
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_memory_usage(self):
        """Test memory usage during processing."""
        import psutil
        import os
        from src.core.pipeline import PIIExtractionPipeline
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = {"extractors": ["rule_based"]}
        pipeline = PIIExtractionPipeline(config)
        
        # Process large amount of data
        large_text = "Test document with PII: test@email.com\n" * 10000
        
        for _ in range(100):
            pipeline.process_text(large_text)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not increase dramatically
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
    
    @pytest.mark.performance
    def test_concurrent_processing(self, benchmark):
        """Test performance under concurrent load."""
        import concurrent.futures
        from src.core.pipeline import PIIExtractionPipeline
        
        config = {"extractors": ["rule_based"]}
        
        def process_document(doc_id):
            pipeline = PIIExtractionPipeline(config)
            text = f"Document {doc_id}: user{doc_id}@email.com, +1-555-{doc_id:04d}"
            return pipeline.process_text(text)
        
        def concurrent_processing():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_document, i) for i in range(20)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            return results
        
        results = benchmark(concurrent_processing)
        
        # Performance assertions
        assert len(results) == 20
        assert benchmark.stats.stats.mean < 5.0  # Should complete in under 5 seconds


class TestModelPerformance:
    """Performance tests for ML models."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_model_inference_speed(self, benchmark):
        """Test ML model inference performance."""
        # Mock model for testing
        class MockModel:
            def predict(self, text):
                time.sleep(0.1)  # Simulate model inference time
                return [{"entity": "PERSON", "text": "John Doe", "confidence": 0.95}]
        
        model = MockModel()
        test_text = "John Doe works at TechCorp. Email: john@techcorp.com"
        
        def model_inference():
            return model.predict(test_text)
        
        result = benchmark(model_inference)
        
        # Performance assertions
        assert len(result) > 0
        assert benchmark.stats.stats.mean < 0.2  # Should complete in under 200ms
    
    @pytest.mark.performance
    def test_batch_processing_efficiency(self, benchmark):
        """Test efficiency of batch processing vs individual processing."""
        # This would test actual model batch processing
        # For now, using mock implementation
        
        class MockBatchModel:
            def predict_batch(self, texts):
                # Simulate batch processing efficiency
                time.sleep(len(texts) * 0.05)  # More efficient than individual
                return [
                    [{"entity": "EMAIL", "text": f"user{i}@email.com"}] 
                    for i in range(len(texts))
                ]
        
        model = MockBatchModel()
        test_texts = [f"User {i}: user{i}@email.com" for i in range(10)]
        
        def batch_inference():
            return model.predict_batch(test_texts)
        
        results = benchmark(batch_inference)
        
        # Performance assertions
        assert len(results) == 10
        # Batch processing should be more efficient than 10 individual calls
        expected_individual_time = 0.1 * 10  # 10 individual calls at 100ms each
        assert benchmark.stats.stats.mean < expected_individual_time


# Performance configuration
def pytest_configure(config):
    """Configure performance testing."""
    config.addinivalue_line(
        "markers", "benchmark: Performance benchmark tests"
    )


# Custom performance reporting
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    """Custom teardown for performance tests."""
    if hasattr(item, 'benchmark') and item.benchmark:
        # Log performance metrics
        stats = item.benchmark.stats.stats
        print(f"\n Performance Stats for {item.name}:")
        print(f"  Mean: {stats.mean:.4f}s")
        print(f"  Min:  {stats.min:.4f}s") 
        print(f"  Max:  {stats.max:.4f}s")
        print(f"  Std:  {stats.stddev:.4f}s")