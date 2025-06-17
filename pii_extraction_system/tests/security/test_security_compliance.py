# Security and compliance tests for PII Extraction System
# Agent 5: DevOps & CI/CD Specialist

import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import security testing libraries
try:
    import bandit
    import safety
except ImportError:
    pytest.skip("Security testing libraries not available", allow_module_level=True)


class TestSecurityCompliance:
    """Security and compliance tests."""
    
    @pytest.mark.security
    def test_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded in the codebase."""
        src_dir = Path(__file__).parent.parent.parent / "src"
        
        # Patterns that indicate potential secrets
        secret_patterns = [
            "password",
            "secret",
            "api_key",
            "aws_access_key",
            "aws_secret",
            "private_key",
            "token"
        ]
        
        violations = []
        
        for py_file in src_dir.rglob("*.py"):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
                for pattern in secret_patterns:
                    if f"{pattern} =" in content or f'"{pattern}"' in content:
                        # Check if it's a configuration key or actual secret
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if pattern in line and '=' in line:
                                # Skip if it's just a config key
                                if not any(x in line for x in ['config', 'key', 'env']):
                                    violations.append(f"{py_file}:{i+1} - {line.strip()}")
        
        assert len(violations) == 0, f"Potential hardcoded secrets found: {violations}"
    
    @pytest.mark.security
    def test_input_validation(self):
        """Test input validation and sanitization."""
        from src.extractors.rule_based import RuleBasedExtractor
        
        extractor = RuleBasedExtractor()
        
        # Test malicious inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "\x00\x01\x02\x03",  # Binary data
            "A" * 10000,  # Very long input
        ]
        
        for malicious_input in malicious_inputs:
            try:
                result = extractor.extract(malicious_input)
                # Should handle gracefully without crashing
                assert result is not None
            except Exception as e:
                # Should not expose sensitive error information
                assert "password" not in str(e).lower()
                assert "secret" not in str(e).lower()
                assert "internal" not in str(e).lower()
    
    @pytest.mark.security
    def test_file_upload_security(self, temp_dir):
        """Test file upload security measures."""
        from src.utils.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Test dangerous file types
        dangerous_files = [
            "malicious.exe",
            "script.js", 
            "payload.php",
            "backdoor.py"
        ]
        
        for filename in dangerous_files:
            test_file = temp_dir / filename
            test_file.write_text("malicious content")
            
            # Should reject dangerous file types
            with pytest.raises(ValueError, match="Unsupported file type"):
                processor.process_document(str(test_file))
    
    @pytest.mark.security
    def test_data_encryption(self):
        """Test that sensitive data is properly encrypted."""
        from src.utils.data_storage import DataStorage
        
        # Mock configuration with encryption enabled
        config = {
            "storage": {
                "type": "local",
                "encryption": {"enabled": True}
            }
        }
        
        storage = DataStorage(config)
        
        # Test data encryption
        sensitive_data = {
            "ssn": "123-45-6789",
            "email": "john.doe@email.com"
        }
        
        # Data should be encrypted before storage
        encrypted_data = storage._encrypt_data(json.dumps(sensitive_data))
        assert encrypted_data != json.dumps(sensitive_data)
        
        # Data should be decryptable
        decrypted_data = storage._decrypt_data(encrypted_data)
        assert json.loads(decrypted_data) == sensitive_data
    
    @pytest.mark.security
    def test_access_control(self):
        """Test access control mechanisms."""
        # This would test authentication and authorization
        # For now, testing basic access patterns
        
        from src.core.config import Settings
        
        settings = Settings()
        
        # Should require authentication for sensitive operations
        assert hasattr(settings, 'require_auth')
        assert settings.require_auth is True
        
        # Should have role-based access control
        assert hasattr(settings, 'roles')
        assert 'admin' in settings.roles
        assert 'user' in settings.roles
    
    @pytest.mark.security
    def test_logging_security(self):
        """Test that logs don't contain sensitive information."""
        from src.core.logging_config import setup_logging
        import logging
        
        # Setup test logging
        logger = setup_logging("test", level="DEBUG")
        
        # Test that PII is not logged
        pii_data = {
            "ssn": "123-45-6789",
            "email": "john.doe@email.com",
            "phone": "+1-555-123-4567"
        }
        
        with patch('logging.Logger.info') as mock_log:
            logger.info(f"Processing document with data: {pii_data}")
            
            # Check that sensitive data is redacted in logs
            logged_message = mock_log.call_args[0][0]
            assert "123-45-6789" not in logged_message
            assert "john.doe@email.com" not in logged_message
            assert "[REDACTED]" in logged_message or "***" in logged_message

    @pytest.mark.security
    def test_gdpr_compliance(self):
        """Test GDPR compliance features."""
        from src.core.pipeline import PIIExtractionPipeline
        
        config = {
            "privacy": {
                "gdpr_compliance": True,
                "data_retention_days": 30
            }
        }
        
        pipeline = PIIExtractionPipeline(config)
        
        # Should support data deletion
        assert hasattr(pipeline, 'delete_user_data')
        
        # Should support data export
        assert hasattr(pipeline, 'export_user_data')
        
        # Should track consent
        assert hasattr(pipeline, 'track_consent')
    
    @pytest.mark.security
    def test_vulnerability_scanning(self):
        """Test for known vulnerabilities in dependencies."""
        # This would integrate with safety or similar tools
        import subprocess
        import sys
        
        try:
            # Run safety check
            result = subprocess.run(
                [sys.executable, "-m", "safety", "check", "--json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                vulnerabilities = json.loads(result.stdout)
                assert len(vulnerabilities) == 0, f"Vulnerabilities found: {vulnerabilities}"
        
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            pytest.skip("Safety check not available or failed")
    
    @pytest.mark.security
    def test_secure_configuration(self):
        """Test secure configuration practices."""
        from src.core.config import Settings
        
        settings = Settings()
        
        # Should use environment variables for sensitive config
        assert settings.aws_access_key_id is None or settings.aws_access_key_id.startswith("${")
        assert settings.aws_secret_access_key is None or settings.aws_secret_access_key.startswith("${")
        
        # Should have secure defaults
        assert settings.debug is False or os.getenv("ENV") == "development"
        assert settings.log_level != "DEBUG" or os.getenv("ENV") in ["development", "testing"]


class TestPrivacyCompliance:
    """Privacy-specific compliance tests."""
    
    @pytest.mark.security
    def test_pii_redaction(self):
        """Test PII redaction capabilities."""
        from src.extractors.rule_based import RuleBasedExtractor
        
        extractor = RuleBasedExtractor()
        
        text_with_pii = """
        John Doe's information:
        SSN: 123-45-6789
        Email: john.doe@email.com
        Phone: +1-555-123-4567
        """
        
        # Extract PII
        result = extractor.extract(text_with_pii)
        
        # Should be able to redact PII
        redacted_text = extractor.redact_pii(text_with_pii, result.entities)
        
        # Sensitive information should be redacted
        assert "123-45-6789" not in redacted_text
        assert "john.doe@email.com" not in redacted_text
        assert "+1-555-123-4567" not in redacted_text
        
        # Should contain redaction markers
        assert "[REDACTED]" in redacted_text or "***" in redacted_text
    
    @pytest.mark.security
    def test_data_anonymization(self):
        """Test data anonymization features."""
        from src.utils.privacy import DataAnonymizer
        
        anonymizer = DataAnonymizer()
        
        original_data = {
            "name": "John Doe",
            "email": "john.doe@email.com",
            "age": 30,
            "city": "New York"
        }
        
        anonymized_data = anonymizer.anonymize(original_data)
        
        # Personal identifiers should be anonymized
        assert anonymized_data["name"] != "John Doe"
        assert anonymized_data["email"] != "john.doe@email.com"
        
        # Non-sensitive data should be preserved or generalized
        assert anonymized_data["age"] is not None
        assert anonymized_data["city"] is not None
    
    @pytest.mark.security
    def test_consent_tracking(self):
        """Test consent tracking and management."""
        from src.utils.privacy import ConsentManager
        
        consent_manager = ConsentManager()
        
        user_id = "test_user_123"
        
        # Record consent
        consent_manager.record_consent(
            user_id=user_id,
            consent_type="data_processing",
            granted=True
        )
        
        # Check consent status
        assert consent_manager.has_consent(user_id, "data_processing") is True
        
        # Revoke consent
        consent_manager.revoke_consent(user_id, "data_processing")
        
        # Should respect revoked consent
        assert consent_manager.has_consent(user_id, "data_processing") is False


# Security test configuration
def pytest_configure(config):
    """Configure security testing."""
    config.addinivalue_line(
        "markers", "security: Security and compliance tests"
    )