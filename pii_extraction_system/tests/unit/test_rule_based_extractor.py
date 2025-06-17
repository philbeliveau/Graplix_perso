"""Unit tests for rule-based PII extractor."""

import pytest
from src.extractors.rule_based import RuleBasedExtractor
from src.extractors.base import PIIEntity


class TestRuleBasedExtractor:
    """Test cases for rule-based PII extractor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = RuleBasedExtractor()
    
    def test_email_extraction(self):
        """Test email address extraction."""
        document = {
            'raw_text': 'Contact John at john.doe@example.com for more information.',
            'file_type': '.txt'
        }
        
        result = self.extractor.extract_pii(document)
        
        assert len(result.pii_entities) >= 1
        email_entities = [e for e in result.pii_entities if e.pii_type == 'email_address']
        assert len(email_entities) == 1
        assert email_entities[0].text == 'john.doe@example.com'
        assert email_entities[0].confidence > 0.8
    
    def test_phone_number_extraction(self):
        """Test phone number extraction."""
        document = {
            'raw_text': 'Call me at (555) 123-4567 or 555.987.6543',
            'file_type': '.txt'
        }
        
        result = self.extractor.extract_pii(document)
        
        phone_entities = [e for e in result.pii_entities if e.pii_type == 'phone_number']
        assert len(phone_entities) >= 1
        
        # Check that we found the phone numbers
        phone_texts = [e.text for e in phone_entities]
        assert any('555' in text for text in phone_texts)
    
    def test_multiple_pii_types(self):
        """Test extraction of multiple PII types."""
        document = {
            'raw_text': '''
            Patient: John Smith
            Email: john.smith@hospital.com
            Phone: (555) 123-4567
            DOB: 01/15/1985
            Address: 123 Main Street
            ''',
            'file_type': '.txt'
        }
        
        result = self.extractor.extract_pii(document)
        
        # Should find multiple types
        unique_types = result.get_unique_entity_types()
        assert len(unique_types) >= 3
        
        # Check statistics
        stats = result.get_statistics()
        assert stats['total_entities'] >= 4
        assert stats['avg_confidence'] > 0.4
    
    def test_empty_document(self):
        """Test handling of empty documents."""
        document = {
            'raw_text': '',
            'file_type': '.txt'
        }
        
        result = self.extractor.extract_pii(document)
        
        assert len(result.pii_entities) == 0
        assert result.error is not None
    
    def test_no_pii_document(self):
        """Test handling of documents with no PII."""
        document = {
            'raw_text': 'This is a simple document with no personal information whatsoever.',
            'file_type': '.txt'
        }
        
        result = self.extractor.extract_pii(document)
        
        # Should complete without error even if no PII found
        assert result.error is None
        assert len(result.pii_entities) == 0


if __name__ == '__main__':
    pytest.main([__file__])