"""Test suite for Agent 2 deliverables - PII extraction core functionality."""

import pytest
from pathlib import Path
import tempfile
import json

# Import the extractors
from src.extractors.rule_based import RuleBasedExtractor
from src.extractors.evaluation import PIIEvaluator, GroundTruthEntity, EvaluationMetrics
from src.extractors.dictionary_extractor import DictionaryExtractor, DictionaryConfig
from src.extractors.base import PIIEntity, PIIExtractionResult


class TestRuleBasedExtractor:
    """Test rule-based PII extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = RuleBasedExtractor()
    
    def test_email_extraction(self):
        """Test email address extraction."""
        document = {
            'raw_text': 'Contact John at john.doe@example.com or jane@test.org for more info.'
        }
        
        result = self.extractor.extract_pii(document)
        
        # Should find both email addresses
        email_entities = [e for e in result.pii_entities if e.pii_type == 'email_address']
        assert len(email_entities) >= 2
        
        # Check specific emails
        email_texts = [e.text for e in email_entities]
        assert 'john.doe@example.com' in email_texts
        assert 'jane@test.org' in email_texts
    
    def test_phone_number_extraction(self):
        """Test phone number extraction."""
        document = {
            'raw_text': 'Call us at (555) 123-4567 or 1-800-555-0123. French: 01 23 45 67 89'
        }
        
        result = self.extractor.extract_pii(document)
        
        # Should find phone numbers
        phone_entities = [e for e in result.pii_entities if e.pii_type == 'phone_number']
        assert len(phone_entities) >= 2
    
    def test_multilingual_support(self):
        """Test French/English multilingual processing."""
        document = {
            'raw_text': 'Contact: john@example.com, né le 15 janvier 1990. Phone: (555) 123-4567'
        }
        
        result = self.extractor.extract_pii(document)
        
        # Should extract entities from both languages
        assert len(result.pii_entities) >= 2
        
        # Check for French date pattern
        date_entities = [e for e in result.pii_entities if e.pii_type == 'date_of_birth']
        assert len(date_entities) >= 1
    
    def test_confidence_scores(self):
        """Test confidence score calculation."""
        document = {
            'raw_text': 'Email: john@example.com with high confidence. Maybe test@invalid'
        }
        
        result = self.extractor.extract_pii(document)
        
        # All entities should have valid confidence scores
        for entity in result.pii_entities:
            assert 0.0 <= entity.confidence <= 1.0
    
    def test_context_extraction(self):
        """Test context extraction around PII entities."""
        document = {
            'raw_text': 'The patient john.doe@hospital.com was admitted on 2023-01-15.'
        }
        
        result = self.extractor.extract_pii(document)
        
        # Entities should have context
        for entity in result.pii_entities:
            assert entity.context is not None
            assert len(entity.context) > 0


class TestNERExtractor:
    """Test NER-based PII extraction."""
    
    def test_ner_initialization(self):
        """Test NER extractor can be initialized."""
        try:
            from src.extractors.ner_extractor import NERExtractor
            # This might fail if transformers is not installed, which is expected
            extractor = NERExtractor()
            assert extractor.name == "ner_extractor"
        except ImportError:
            # Expected if transformers is not installed
            pytest.skip("Transformers library not available")
    
    def test_model_configs(self):
        """Test NER model configurations."""
        from src.extractors.ner_extractor import NERExtractor
        
        models_info = NERExtractor.get_available_models()
        
        # Should have predefined models
        assert len(models_info) >= 3
        assert 'multilingual_ner' in models_info
        assert 'french_ner' in models_info
        
        # Check model info structure
        for model_name, info in models_info.items():
            assert 'name' in info
            assert 'model_name' in info
            assert 'supported_languages' in info


class TestDictionaryExtractor:
    """Test dictionary-based PII extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_path = Path(temp_dir)
            self.extractor = DictionaryExtractor(dictionaries_path=self.temp_path)
    
    def test_default_dictionaries_creation(self):
        """Test creation of default dictionaries."""
        # Should have created default dictionaries
        assert len(self.extractor.dictionaries) >= 6
        
        # Check specific dictionaries
        dict_names = list(self.extractor.dictionaries.keys())
        assert 'healthcare_identifiers' in dict_names
        assert 'government_identifiers' in dict_names
        assert 'financial_identifiers' in dict_names
    
    def test_healthcare_identifier_extraction(self):
        """Test healthcare identifier extraction."""
        document = {
            'raw_text': 'Patient ID: P123456, MRN: 7890123, health card number HC456789'
        }
        
        result = self.extractor.extract_pii(document)
        
        # Should find healthcare identifiers
        healthcare_entities = [e for e in result.pii_entities if e.pii_type == 'medical_record_number']
        assert len(healthcare_entities) >= 1
    
    def test_multilingual_dictionary_support(self):
        """Test French/English dictionary support."""
        document = {
            'raw_text': 'Identifiant patient: P123456, numéro de dossier médical: 789012'
        }
        
        result = self.extractor.extract_pii(document)
        
        # Should find French identifiers
        assert len(result.pii_entities) >= 1
    
    def test_custom_dictionary_addition(self):
        """Test adding custom dictionaries."""
        custom_config = DictionaryConfig(
            name="test_custom",
            description="Test custom dictionary",
            pii_type="custom_id",
            terms=["custom_term", "test_identifier"],
            confidence_score=0.9
        )
        
        success = self.extractor.add_dictionary(custom_config)
        assert success
        
        # Should be in dictionaries
        assert 'test_custom' in self.extractor.dictionaries
        
        # Test extraction
        document = {
            'raw_text': 'The custom_term is important for test_identifier validation.'
        }
        
        result = self.extractor.extract_pii(document)
        custom_entities = [e for e in result.pii_entities if e.pii_type == 'custom_id']
        assert len(custom_entities) >= 1


class TestEvaluationFramework:
    """Test evaluation framework for PII extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = PIIEvaluator()
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator.position_tolerance == 5
        assert self.evaluator.text_similarity_threshold == 0.8
    
    def test_ground_truth_entity_creation(self):
        """Test ground truth entity creation."""
        gt_entity = GroundTruthEntity(
            text="john@example.com",
            pii_type="email_address",
            start_pos=10,
            end_pos=26
        )
        
        assert gt_entity.text == "john@example.com"
        assert gt_entity.pii_type == "email_address"
        
        # Test serialization
        gt_dict = gt_entity.to_dict()
        assert gt_dict['text'] == "john@example.com"
        
        # Test deserialization
        gt_restored = GroundTruthEntity.from_dict(gt_dict)
        assert gt_restored.text == gt_entity.text
    
    def test_perfect_match_evaluation(self):
        """Test evaluation with perfect matches."""
        # Create predicted entities
        predicted_entities = [
            PIIEntity(
                text="john@example.com",
                pii_type="email_address",
                confidence=0.9,
                start_pos=10,
                end_pos=26,
                extractor="test"
            )
        ]
        
        predicted_result = PIIExtractionResult(
            pii_entities=predicted_entities,
            confidence_scores=[0.9],
            processing_time=0.1
        )
        
        # Create ground truth
        ground_truth = [
            GroundTruthEntity(
                text="john@example.com",
                pii_type="email_address",
                start_pos=10,
                end_pos=26
            )
        ]
        
        # Evaluate
        metrics = self.evaluator.evaluate_extraction_result(
            predicted_result, ground_truth
        )
        
        # Should have perfect scores
        assert metrics.overall_precision == 1.0
        assert metrics.overall_recall == 1.0
        assert metrics.overall_f1_score == 1.0
    
    def test_partial_match_evaluation(self):
        """Test evaluation with partial matches."""
        # Create predicted entities (missing one)
        predicted_entities = [
            PIIEntity(
                text="john@example.com",
                pii_type="email_address",
                confidence=0.9,
                start_pos=10,
                end_pos=26,
                extractor="test"
            )
        ]
        
        predicted_result = PIIExtractionResult(
            pii_entities=predicted_entities,
            confidence_scores=[0.9],
            processing_time=0.1
        )
        
        # Create ground truth (has two entities)
        ground_truth = [
            GroundTruthEntity(
                text="john@example.com",
                pii_type="email_address",
                start_pos=10,
                end_pos=26
            ),
            GroundTruthEntity(
                text="jane@test.org",
                pii_type="email_address",
                start_pos=30,
                end_pos=43
            )
        ]
        
        # Evaluate
        metrics = self.evaluator.evaluate_extraction_result(
            predicted_result, ground_truth
        )
        
        # Should have partial scores
        assert metrics.overall_precision == 1.0  # No false positives
        assert metrics.overall_recall == 0.5     # Missed one entity
        assert 0.0 < metrics.overall_f1_score < 1.0
    
    def test_false_positive_evaluation(self):
        """Test evaluation with false positives."""
        # Create predicted entities (one correct, one false positive)
        predicted_entities = [
            PIIEntity(
                text="john@example.com",
                pii_type="email_address",
                confidence=0.9,
                start_pos=10,
                end_pos=26,
                extractor="test"
            ),
            PIIEntity(
                text="not_an_email",
                pii_type="email_address",
                confidence=0.6,
                start_pos=30,
                end_pos=42,
                extractor="test"
            )
        ]
        
        predicted_result = PIIExtractionResult(
            pii_entities=predicted_entities,
            confidence_scores=[0.9, 0.6],
            processing_time=0.1
        )
        
        # Create ground truth (only one entity)
        ground_truth = [
            GroundTruthEntity(
                text="john@example.com",
                pii_type="email_address",
                start_pos=10,
                end_pos=26
            )
        ]
        
        # Evaluate
        metrics = self.evaluator.evaluate_extraction_result(
            predicted_result, ground_truth
        )
        
        # Should have reduced precision due to false positive
        assert metrics.overall_precision == 0.5  # 1 TP, 1 FP
        assert metrics.overall_recall == 1.0     # Found the correct entity
        assert 0.0 < metrics.overall_f1_score < 1.0
    
    def test_metrics_serialization(self):
        """Test evaluation metrics serialization."""
        metrics = EvaluationMetrics()
        metrics.overall_precision = 0.85
        metrics.overall_recall = 0.90
        metrics.overall_f1_score = 0.875
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert metrics_dict['overall_precision'] == 0.85
        assert metrics_dict['overall_recall'] == 0.90
        assert metrics_dict['overall_f1_score'] == 0.875


class TestIntegration:
    """Integration tests for all Agent 2 components."""
    
    def test_end_to_end_extraction_and_evaluation(self):
        """Test complete extraction and evaluation workflow."""
        # Create extractors
        rule_extractor = RuleBasedExtractor()
        evaluator = PIIEvaluator()
        
        # Test document
        document = {
            'raw_text': 'Contact John Doe at john.doe@example.com or call (555) 123-4567. Born on January 15, 1990.'
        }
        
        # Extract PII
        result = rule_extractor.extract_pii(document)
        
        # Should find multiple entities
        assert len(result.pii_entities) >= 3
        
        # Create ground truth for evaluation
        ground_truth = [
            GroundTruthEntity(
                text="john.doe@example.com",
                pii_type="email_address",
                start_pos=27,
                end_pos=47
            ),
            GroundTruthEntity(
                text="(555) 123-4567",
                pii_type="phone_number",
                start_pos=56,
                end_pos=70
            )
        ]
        
        # Evaluate
        metrics = evaluator.evaluate_extraction_result(result, ground_truth)
        
        # Should have reasonable performance
        assert metrics.overall_precision >= 0.5
        assert metrics.overall_recall >= 0.5
        assert metrics.overall_f1_score >= 0.5
    
    def test_multilingual_integration(self):
        """Test multilingual support across all extractors."""
        # Test French and English content
        document = {
            'raw_text': '''
            Patient Information / Informations Patient:
            Name: Jean Dupont / John Smith
            Email: jean.dupont@hospital.fr / john@clinic.com
            Phone: +33 1 23 45 67 89 / (555) 123-4567
            Born: 15 janvier 1980 / January 15, 1985
            Patient ID: P123456 / MRN: 789012
            '''
        }
        
        # Test with rule-based extractor
        rule_extractor = RuleBasedExtractor()
        result = rule_extractor.extract_pii(document)
        
        # Should extract entities from both languages
        assert len(result.pii_entities) >= 6
        
        # Check for specific types
        email_entities = [e for e in result.pii_entities if e.pii_type == 'email_address']
        phone_entities = [e for e in result.pii_entities if e.pii_type == 'phone_number']
        date_entities = [e for e in result.pii_entities if e.pii_type == 'date_of_birth']
        
        assert len(email_entities) >= 2
        assert len(phone_entities) >= 2
        assert len(date_entities) >= 1  # French date should be detected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])