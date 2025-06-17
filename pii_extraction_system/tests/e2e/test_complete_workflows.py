"""End-to-end tests for complete PII extraction workflows."""

import json
import pytest
import tempfile
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any
import requests
from unittest.mock import Mock, patch

from src.core.pipeline import PIIExtractionPipeline
from src.core.config import PIIConfig
from src.utils.document_processor import DocumentProcessor
from src.utils.data_storage import DataStorageManager


class TestCompleteWorkflows:
    """End-to-end tests for complete PII extraction workflows."""
    
    @classmethod
    def setup_class(cls):
        """Set up end-to-end test environment."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.test_data_dir = cls.temp_dir / "e2e_data"
        cls.test_output_dir = cls.temp_dir / "e2e_output"
        
        cls.test_data_dir.mkdir(exist_ok=True)
        cls.test_output_dir.mkdir(exist_ok=True)
        
        # Initialize complete system
        cls.config = PIIConfig()
        cls.config.output_dir = str(cls.test_output_dir)
        cls.config.use_local_storage = True
        cls.config.confidence_threshold = 0.5
        
        cls.pipeline = PIIExtractionPipeline(cls.config)
        
        # Create comprehensive test data
        cls._create_comprehensive_test_data()
    
    @classmethod
    def _create_comprehensive_test_data(cls):
        """Create comprehensive test data for E2E testing."""
        
        # Employee onboarding form
        employee_form = """
        EMPLOYEE ONBOARDING FORM
        
        Personal Information:
        Full Name: Jennifer Martinez
        Email Address: jennifer.martinez@newcompany.com
        Personal Email: jenny.m@gmail.com
        Phone Number: (647) 555-0123
        Mobile: 647.555.0987
        Date of Birth: March 15, 1992
        Social Insurance Number: 123-456-789
        
        Address:
        Street: 789 College Street, Apt 4B
        City: Toronto
        Province: Ontario
        Postal Code: M6G 1C5
        
        Emergency Contact:
        Name: Carlos Martinez
        Relationship: Spouse
        Phone: (647) 555-0456
        Email: carlos.martinez@email.com
        
        Banking Information:
        Bank: TD Canada Trust
        Transit Number: 12345
        Account Number: 9876543210
        
        Employment Details:
        Employee ID: EMP-2024-001
        Department: Software Engineering
        Start Date: 2024-01-15
        Manager: Dr. Sarah Thompson
        Manager Email: sarah.thompson@newcompany.com
        
        IT Information:
        Workstation IP: 192.168.100.45
        VPN Access: jennifer.martinez
        Company Website: https://newcompany.com/employee-portal
        """
        
        (cls.test_data_dir / "employee_onboarding.txt").write_text(employee_form)
        
        # Medical record
        medical_record = """
        MEDICAL RECORD
        
        Patient: Mr. Robert Chen
        Medical Record Number: MRN-789456123
        Date of Birth: 08/22/1975
        Health Card: 1234-567-890-ON
        
        Contact Information:
        Phone: (416) 555-0789
        Email: robert.chen@outlook.com
        Address: 1234 Yonge Street, Toronto, ON M4W 2L2
        
        Emergency Contact:
        Name: Lisa Chen
        Phone: (416) 555-0321
        Relationship: Wife
        
        Insurance Information:
        Provider: Sun Life Financial
        Policy Number: SL-987654321
        Group Number: GRP-456789
        
        Visit Information:
        Date: 2024-01-20
        Doctor: Dr. Amanda Wilson
        Doctor ID: DOC-12345
        
        Prescription:
        Medication: Lisinopril 10mg
        Prescription Number: RX-789456123
        Pharmacy: Shoppers Drug Mart
        Pharmacist: Dr. Michael Johnson
        """
        
        (cls.test_data_dir / "medical_record.txt").write_text(medical_record)
        
        # Financial document
        financial_doc = """
        FINANCIAL SERVICES APPLICATION
        
        Client Information:
        Name: Ms. Amanda Foster
        Date of Birth: 12/05/1988
        Social Insurance Number: 987-654-321
        Driver's License: F123456789
        
        Contact Details:
        Home Phone: (905) 555-0123
        Cell Phone: 905.555.0456
        Email: amanda.foster@rogers.com
        Address: 567 Main Street East, Hamilton, ON L8N 1K7
        
        Financial Information:
        Bank Account 1: 1111-2222-3333-4444 (Checking)
        Bank Account 2: 5555-6666-7777-8888 (Savings)
        Credit Card: 4532-1234-5678-9012 (Visa)
        Credit Card: 5555-4444-3333-2222 (MasterCard)
        
        Investment Accounts:
        TFSA: TFSA-789456123
        RRSP: RRSP-456789123
        Investment Advisor: John Smith, CFA
        Advisor Email: john.smith@investment.com
        Advisor Phone: (416) 555-0999
        
        Online Access:
        Username: afoster2024
        Website: https://secure.financialservices.com
        Last Login: 2024-01-18 14:30:00
        IP Address: 24.114.200.45
        """
        
        (cls.test_data_dir / "financial_application.txt").write_text(financial_doc)
        
        # Mixed language document (English/French)
        mixed_language = """
        FORMULAIRE BILINGUE / BILINGUAL FORM
        
        Informations personnelles / Personal Information:
        Nom / Name: Pierre Dubois
        Courriel / Email: pierre.dubois@gouvernement.qc.ca
        Téléphone / Phone: (514) 555-0123
        Adresse / Address: 456 rue Saint-Denis, Montréal, QC H2X 3K3
        
        Date de naissance / Date of Birth: 25 juin 1985 / June 25, 1985
        Numéro d'assurance sociale / Social Insurance Number: 456-789-123
        
        Coordonnées d'urgence / Emergency Contact:
        Nom / Name: Marie Dubois
        Téléphone / Phone: (514) 555-0456
        Courriel / Email: marie.dubois@hotmail.fr
        
        Informations bancaires / Banking Information:
        Institution: Banque Nationale / National Bank
        Numéro de transit / Transit Number: 54321
        Numéro de compte / Account Number: 1357924680
        
        Site web / Website: https://www.gouvernement.qc.ca/services
        Adresse IP / IP Address: 192.168.2.100
        """
        
        (cls.test_data_dir / "bilingual_form.txt").write_text(mixed_language)
    
    def test_complete_employee_onboarding_workflow(self):
        """Test complete employee onboarding document processing workflow."""
        input_file = str(self.test_data_dir / "employee_onboarding.txt")
        
        # Process document
        start_time = time.time()
        result = self.pipeline.process_document(input_file)
        processing_time = time.time() - start_time
        
        # Validate basic result structure
        assert result is not None
        assert hasattr(result, 'pii_entities')
        assert len(result.pii_entities) > 0
        assert result.error is None
        
        # Expected PII types for employee onboarding
        expected_types = {
            'person_name', 'email_address', 'phone_number', 'date_of_birth',
            'social_security_number', 'address', 'postal_code', 'employee_id', 'ip_address', 'url'
        }
        
        found_types = set(entity.pii_type for entity in result.pii_entities)
        
        # Should find at least 70% of expected types
        coverage = len(found_types.intersection(expected_types)) / len(expected_types)
        assert coverage >= 0.7, f"Coverage only {coverage:.2%}, found types: {found_types}"
        
        # Validate specific high-confidence entities
        high_conf_entities = [e for e in result.pii_entities if e.confidence > 0.8]
        assert len(high_conf_entities) > 0, "Should have high-confidence entities"
        
        # Performance validation
        assert processing_time < 10.0, f"Processing took too long: {processing_time:.2f}s"
        
        # Validate entity positions are reasonable
        for entity in result.pii_entities:
            assert entity.start_pos >= 0
            assert entity.end_pos > entity.start_pos
            assert entity.end_pos <= len(result.metadata.get('text_length', 1000))
    
    def test_medical_record_workflow(self):
        """Test medical record processing workflow with sensitive data."""
        input_file = str(self.test_data_dir / "medical_record.txt")
        
        result = self.pipeline.process_document(input_file)
        
        # Medical records should contain multiple sensitive PII types
        expected_medical_types = {
            'person_name', 'medical_record_number', 'phone_number',
            'email_address', 'address', 'postal_code', 'date_of_birth'
        }
        
        found_types = set(entity.pii_type for entity in result.pii_entities)
        
        # Should find medical-specific PII
        medical_entities = [e for e in result.pii_entities if e.pii_type == 'medical_record_number']
        assert len(medical_entities) > 0, "Should find medical record numbers"
        
        # Validate high confidence for medical record numbers
        for entity in medical_entities:
            assert entity.confidence > 0.7, f"Medical record number should have high confidence: {entity.confidence}"
        
        # Should find multiple person names (patient, doctor, contacts)
        name_entities = [e for e in result.pii_entities if e.pii_type == 'person_name']
        assert len(name_entities) >= 2, "Should find multiple person names in medical record"
    
    def test_financial_document_workflow(self):
        """Test financial document processing with multiple card numbers."""
        input_file = str(self.test_data_dir / "financial_application.txt")
        
        result = self.pipeline.process_document(input_file)
        
        # Financial documents should contain financial PII
        expected_financial_types = {
            'credit_card_number', 'social_security_number', 'driver_license',
            'person_name', 'email_address', 'phone_number', 'ip_address', 'url'
        }
        
        found_types = set(entity.pii_type for entity in result.pii_entities)
        
        # Should find credit cards
        cc_entities = [e for e in result.pii_entities if e.pii_type == 'credit_card_number']
        assert len(cc_entities) >= 2, "Should find multiple credit card numbers"
        
        # Validate credit card confidence
        for entity in cc_entities:
            assert entity.confidence > 0.6, f"Credit card should have reasonable confidence: {entity.confidence}"
            # Validate credit card format
            assert len(entity.text.replace('-', '').replace(' ', '')) >= 13
        
        # Should find driver's license
        dl_entities = [e for e in result.pii_entities if e.pii_type == 'driver_license']
        assert len(dl_entities) > 0, "Should find driver's license"
    
    def test_multilingual_workflow(self):
        """Test multilingual document processing workflow."""
        input_file = str(self.test_data_dir / "bilingual_form.txt")
        
        result = self.pipeline.process_document(input_file)
        
        # Should handle both English and French content
        assert result is not None
        assert len(result.pii_entities) > 0
        
        # Should find entities in both languages
        found_types = set(entity.pii_type for entity in result.pii_entities)
        
        # Common types that should work in both languages
        expected_multilingual_types = {'email_address', 'phone_number', 'postal_code', 'ip_address', 'url'}
        
        found_multilingual = found_types.intersection(expected_multilingual_types)
        assert len(found_multilingual) >= 3, f"Should find multilingual PII types: {found_multilingual}"
        
        # Should find person names (works for both languages)
        name_entities = [e for e in result.pii_entities if e.pii_type == 'person_name']
        assert len(name_entities) >= 1, "Should find person names in multilingual document"
    
    def test_batch_processing_workflow(self):
        """Test batch processing workflow with multiple documents."""
        input_files = [
            str(self.test_data_dir / "employee_onboarding.txt"),
            str(self.test_data_dir / "medical_record.txt"),
            str(self.test_data_dir / "financial_application.txt")
        ]
        
        start_time = time.time()
        batch_results = self.pipeline.process_batch(input_files)
        total_time = time.time() - start_time
        
        # Validate batch results
        assert len(batch_results) == len(input_files)
        
        total_entities = 0
        for i, result in enumerate(batch_results):
            assert result is not None
            assert hasattr(result, 'pii_entities')
            assert len(result.pii_entities) > 0
            total_entities += len(result.pii_entities)
            
            # Each document should be processed successfully
            assert result.error is None or len(result.pii_entities) > 0
        
        # Should find substantial number of entities across all documents
        assert total_entities > 20, f"Batch processing should find many entities: {total_entities}"
        
        # Batch processing should be reasonably efficient
        avg_time_per_doc = total_time / len(input_files)
        assert avg_time_per_doc < 15.0, f"Average time per document too high: {avg_time_per_doc:.2f}s"
    
    def test_error_recovery_workflow(self):
        """Test error recovery in complete workflows."""
        # Test with corrupted input
        corrupted_file = self.test_data_dir / "corrupted.txt"
        corrupted_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')  # Binary garbage
        
        result = self.pipeline.process_document(str(corrupted_file))
        
        # Should handle gracefully
        assert result is not None
        # Either successful processing or graceful error
        assert result.error is not None or len(result.pii_entities) >= 0
        
        # Test with non-existent file
        result2 = self.pipeline.process_document("/nonexistent/file.txt")
        assert result2 is not None
        assert result2.error is not None
    
    def test_large_document_workflow(self):
        """Test workflow with large documents."""
        # Create large document
        large_content = """
        LARGE EMPLOYEE DATABASE EXPORT
        
        """
        
        # Add many employee records
        for i in range(100):
            employee_block = f"""
            Employee #{i+1:03d}:
            Name: User{i:03d} Test{i:03d}
            Email: user{i:03d}@company{i%5:01d}.com
            Phone: (555) {i:03d}-{(i*7)%10000:04d}
            Employee ID: EMP-{i:06d}
            Address: {i+1} Main Street, Unit {i%50+1}
            DOB: {(i%12)+1:02d}/{(i%28)+1:02d}/{1970+(i%50)}
            """
            large_content += employee_block
        
        large_file = self.test_data_dir / "large_employee_db.txt"
        large_file.write_text(large_content)
        
        start_time = time.time()
        result = self.pipeline.process_document(str(large_file))
        processing_time = time.time() - start_time
        
        # Should handle large document
        assert result is not None
        assert len(result.pii_entities) > 50, "Should find many entities in large document"
        
        # Should complete in reasonable time
        assert processing_time < 30.0, f"Large document processing took too long: {processing_time:.2f}s"
        
        # Should find diverse entity types
        found_types = set(entity.pii_type for entity in result.pii_entities)
        assert len(found_types) >= 4, f"Should find diverse PII types: {found_types}"
    
    @pytest.mark.slow
    def test_stress_testing_workflow(self):
        """Stress test the complete workflow."""
        input_file = str(self.test_data_dir / "employee_onboarding.txt")
        
        # Process same document multiple times rapidly
        results = []
        start_time = time.time()
        
        for i in range(20):
            result = self.pipeline.process_document(input_file)
            results.append(result)
        
        total_time = time.time() - start_time
        
        # All results should be valid
        for i, result in enumerate(results):
            assert result is not None, f"Result {i} is None"
            assert len(result.pii_entities) > 0, f"Result {i} has no entities"
        
        # Results should be consistent
        entity_counts = [len(r.pii_entities) for r in results]
        count_variance = max(entity_counts) - min(entity_counts)
        assert count_variance <= max(entity_counts) * 0.2, "Results not consistent across runs"
        
        # Should maintain reasonable performance under load
        avg_time = total_time / len(results)
        assert avg_time < 5.0, f"Average processing time under load: {avg_time:.3f}s"
    
    def test_data_persistence_workflow(self):
        """Test complete workflow with data persistence."""
        input_file = str(self.test_data_dir / "employee_onboarding.txt")
        
        # Process document
        result = self.pipeline.process_document(input_file)
        
        # Create storage manager and save results
        storage_manager = DataStorageManager(self.config)
        
        result_data = {
            'file_path': input_file,
            'entities': [
                {
                    'text': e.text,
                    'type': e.pii_type,
                    'confidence': e.confidence,
                    'start_pos': e.start_pos,
                    'end_pos': e.end_pos
                }
                for e in result.pii_entities
            ],
            'processing_time': result.processing_time,
            'metadata': result.metadata
        }
        
        # Save and retrieve
        result_id = storage_manager.save_results(result_data)
        assert result_id is not None
        
        loaded_data = storage_manager.load_results(result_id)
        assert loaded_data is not None
        assert len(loaded_data['entities']) == len(result.pii_entities)
        
        # Validate data integrity
        for i, entity_data in enumerate(loaded_data['entities']):
            original_entity = result.pii_entities[i]
            assert entity_data['text'] == original_entity.text
            assert entity_data['type'] == original_entity.pii_type
            assert abs(entity_data['confidence'] - original_entity.confidence) < 0.001
    
    def test_configuration_workflow(self):
        """Test workflow with different configurations."""
        input_file = str(self.test_data_dir / "employee_onboarding.txt")
        
        # Test with high confidence threshold
        high_conf_config = PIIConfig()
        high_conf_config.confidence_threshold = 0.9
        high_conf_config.use_local_storage = True
        
        high_conf_pipeline = PIIExtractionPipeline(high_conf_config)
        result_high = high_conf_pipeline.process_document(input_file)
        
        # Test with low confidence threshold
        low_conf_config = PIIConfig()
        low_conf_config.confidence_threshold = 0.1
        low_conf_config.use_local_storage = True
        
        low_conf_pipeline = PIIExtractionPipeline(low_conf_config)
        result_low = low_conf_pipeline.process_document(input_file)
        
        # Low threshold should find more entities
        assert len(result_low.pii_entities) >= len(result_high.pii_entities)
        
        # Validate confidence filtering works
        for entity in result_high.pii_entities:
            assert entity.confidence >= 0.9, f"High threshold should filter low confidence: {entity.confidence}"
    
    @classmethod
    def teardown_class(cls):
        """Clean up end-to-end test environment."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])