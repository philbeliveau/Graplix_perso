"""Comprehensive unit tests for all PII extractors."""

import json
import os
import pytest
import time
from pathlib import Path
from typing import Dict, List, Any

from src.extractors.rule_based import RuleBasedExtractor
from src.extractors.base import PIIEntity, PIIExtractionResult


class TestComprehensiveExtractors:
    """Comprehensive test suite for all PII extractors."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with synthetic dataset."""
        test_data_path = Path(__file__).parent.parent / "test_data" / "synthetic_pii_dataset.json"
        with open(test_data_path, 'r', encoding='utf-8') as f:
            cls.test_dataset = json.load(f)
        
        cls.extractors = {
            'rule_based': RuleBasedExtractor()
        }
        
        # Performance tracking
        cls.performance_metrics = {}
    
    def test_all_extractors_basic_functionality(self):
        """Test basic functionality of all extractors."""
        for extractor_name, extractor in self.extractors.items():
            with pytest.raises(Exception, match=""):
                # Should not raise exception for valid input
                document = {
                    'raw_text': 'Contact john.doe@example.com',
                    'file_type': '.txt'
                }
                result = extractor.extract_pii(document)
                assert isinstance(result, PIIExtractionResult)
                assert hasattr(result, 'pii_entities')
                assert hasattr(result, 'processing_time')
    
    @pytest.mark.parametrize("doc_data", 
                           [doc for doc in json.load(open(Path(__file__).parent.parent / "test_data" / "synthetic_pii_dataset.json"))['documents'][:5]])
    def test_synthetic_dataset_extraction(self, doc_data):
        """Test extraction against synthetic dataset."""
        document = {
            'raw_text': doc_data['text'],
            'file_type': '.txt'
        }
        
        # Test with rule-based extractor
        extractor = self.extractors['rule_based']
        result = extractor.extract_pii(document)
        
        # Basic validation
        assert result is not None
        assert isinstance(result.pii_entities, list)
        assert result.processing_time > 0
        
        # Check if we found reasonable number of entities
        expected_count = len(doc_data['expected_entities'])
        found_count = len(result.pii_entities)
        
        # Allow some tolerance for different extraction patterns
        assert found_count >= expected_count * 0.5, f"Found {found_count} entities, expected at least {expected_count * 0.5}"
        
        # Validate entity types
        found_types = set(entity.pii_type for entity in result.pii_entities)
        expected_types = set(entity['type'] for entity in doc_data['expected_entities'])
        
        # Should find at least 50% of expected types
        common_types = found_types.intersection(expected_types)
        assert len(common_types) >= len(expected_types) * 0.5
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        edge_cases = self.test_dataset['test_scenarios']['edge_cases']
        
        for case in edge_cases:
            document = {
                'raw_text': case['text'],
                'file_type': '.txt'
            }
            
            for extractor_name, extractor in self.extractors.items():
                result = extractor.extract_pii(document)
                
                # Should not crash
                assert result is not None
                
                # Check expected counts for specific cases
                if case['name'] == 'empty_document':
                    assert len(result.pii_entities) == 0
                elif case['name'] == 'no_pii_document':
                    assert len(result.pii_entities) == 0
                elif case['name'] == 'malformed_emails':
                    # Should not detect malformed emails
                    email_entities = [e for e in result.pii_entities if e.pii_type == 'email_address']
                    assert len(email_entities) == 0
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks for all extractors."""
        # Test with various document sizes
        test_texts = [
            "Short text with john@example.com",
            "Medium text. " * 100 + "Contact sarah@company.com and call (555) 123-4567",
            "Long text. " * 1000 + "Email: admin@system.org, Phone: +1-800-555-0199, Date: 01/01/2000"
        ]
        
        for i, text in enumerate(test_texts):
            document = {
                'raw_text': text,
                'file_type': '.txt'
            }
            
            for extractor_name, extractor in self.extractors.items():
                start_time = time.time()
                result = extractor.extract_pii(document)
                end_time = time.time()
                
                processing_time = end_time - start_time
                
                # Performance thresholds
                max_times = [0.1, 0.5, 2.0]  # seconds for short, medium, long texts
                assert processing_time < max_times[i], f"{extractor_name} took {processing_time:.3f}s for text size {i}"
                
                # Store metrics
                key = f"{extractor_name}_size_{i}"
                if key not in self.performance_metrics:
                    self.performance_metrics[key] = []
                self.performance_metrics[key].append(processing_time)
    
    def test_multilingual_support(self):
        """Test multilingual PII extraction."""
        multilingual_cases = self.test_dataset['test_scenarios']['multilingual']
        
        for case in multilingual_cases:
            document = {
                'raw_text': case['text'],
                'file_type': '.txt'
            }
            
            for extractor_name, extractor in self.extractors.items():
                result = extractor.extract_pii(document)
                
                # Should find entities
                assert len(result.pii_entities) > 0
                
                # Should find expected types
                found_types = set(entity.pii_type for entity in result.pii_entities)
                expected_types = set(case['expected_types'])
                
                # Allow some tolerance for multilingual variations
                common_types = found_types.intersection(expected_types)
                assert len(common_types) >= len(expected_types) * 0.6
    
    def test_confidence_scores(self):
        """Test confidence score calculation."""
        high_confidence_text = "Email: john.doe@company.com, Phone: (555) 123-4567"
        low_confidence_text = "Maybe contact John Smith or visit 123 Main Street"
        
        document_high = {'raw_text': high_confidence_text, 'file_type': '.txt'}
        document_low = {'raw_text': low_confidence_text, 'file_type': '.txt'}
        
        for extractor_name, extractor in self.extractors.items():
            result_high = extractor.extract_pii(document_high)
            result_low = extractor.extract_pii(document_low)
            
            # High confidence entities should have higher average confidence
            if result_high.pii_entities and result_low.pii_entities:
                avg_high = sum(e.confidence for e in result_high.pii_entities) / len(result_high.pii_entities)
                avg_low = sum(e.confidence for e in result_low.pii_entities) / len(result_low.pii_entities)
                
                assert avg_high > avg_low, f"High confidence case should have higher scores than low confidence case"
    
    def test_pii_type_coverage(self):
        """Test coverage of all supported PII types."""
        expected_types = self.test_dataset['metadata']['entity_types']
        
        # Create test document with examples of all types
        test_document = {
            'raw_text': '''
            Personal Information:
            Name: John Smith
            Email: john.smith@example.com
            Phone: (555) 123-4567
            SSN: 123-45-6789
            Credit Card: 4111-1111-1111-1111
            DOB: 01/15/1985
            Address: 123 Main Street
            Postal Code: M5V 3A8
            ZIP: 90210
            Driver License: A12345678
            Medical Record: MRN-123456
            Employee ID: EMP-789
            IP: 192.168.1.1
            Website: https://example.com
            IBAN: GB82 WEST 1234 5698 7654 32
            ''',
            'file_type': '.txt'
        }
        
        for extractor_name, extractor in self.extractors.items():
            result = extractor.extract_pii(test_document)
            
            found_types = set(entity.pii_type for entity in result.pii_entities)
            
            # Should cover at least 70% of supported types
            coverage = len(found_types.intersection(expected_types)) / len(expected_types)
            assert coverage >= 0.7, f"{extractor_name} only covers {coverage:.2%} of expected PII types"
    
    def test_entity_validation(self):
        """Test entity validation logic."""
        # Test valid entities
        valid_emails = ["test@example.com", "user.name@domain.org"]
        invalid_emails = ["@invalid.com", "missing@", "incomplete@email"]
        
        for email in valid_emails:
            document = {'raw_text': f"Contact: {email}", 'file_type': '.txt'}
            result = self.extractors['rule_based'].extract_pii(document)
            
            email_entities = [e for e in result.pii_entities if e.pii_type == 'email_address']
            assert len(email_entities) > 0, f"Valid email {email} was not detected"
        
        for email in invalid_emails:
            document = {'raw_text': f"Contact: {email}", 'file_type': '.txt'}
            result = self.extractors['rule_based'].extract_pii(document)
            
            email_entities = [e for e in result.pii_entities if e.pii_type == 'email_address']
            # Should either not detect or have low confidence
            for entity in email_entities:
                if entity.text == email:
                    assert entity.confidence < 0.5, f"Invalid email {email} had high confidence"
    
    def test_context_awareness(self):
        """Test context-aware confidence scoring."""
        # Email with clear context
        text_with_context = "Please send your report to the email address: john@company.com"
        
        # Email without clear context
        text_without_context = "The configuration file john@company.com contains the settings"
        
        doc_with = {'raw_text': text_with_context, 'file_type': '.txt'}
        doc_without = {'raw_text': text_without_context, 'file_type': '.txt'}
        
        extractor = self.extractors['rule_based']
        
        result_with = extractor.extract_pii(doc_with)
        result_without = extractor.extract_pii(doc_without)
        
        # Find email entities
        email_with = [e for e in result_with.pii_entities if e.pii_type == 'email_address'][0]
        email_without = [e for e in result_without.pii_entities if e.pii_type == 'email_address'][0]
        
        # Context should increase confidence
        assert email_with.confidence > email_without.confidence
    
    @pytest.mark.slow
    def test_stress_testing(self):
        """Stress test with large documents and many entities."""
        # Generate large document with many PII instances
        large_text = ""
        for i in range(100):
            large_text += f"Employee {i}: user{i}@company.com, Phone: (555) {i:03d}-{(i*7)%10000:04d}, "
            large_text += f"Address: {i} Main Street, ID: EMP-{i:06d}\n"
        
        document = {'raw_text': large_text, 'file_type': '.txt'}
        
        for extractor_name, extractor in self.extractors.items():
            start_time = time.time()
            result = extractor.extract_pii(document)
            processing_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert processing_time < 10.0, f"{extractor_name} took too long: {processing_time:.2f}s"
            
            # Should find many entities
            assert len(result.pii_entities) > 100, f"Should find many entities in large document"
            
            # Should not have memory issues
            assert result.error is None or "memory" not in result.error.lower()
    
    def teardown_class(cls):
        """Clean up and report performance metrics."""
        if cls.performance_metrics:
            print("\n=== Performance Metrics ===")
            for key, times in cls.performance_metrics.items():
                avg_time = sum(times) / len(times)
                print(f"{key}: {avg_time:.4f}s average")


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])