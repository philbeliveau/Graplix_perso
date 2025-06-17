"""Privacy-preserving techniques for PII handling."""

import re
import hashlib
import secrets
import base64
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image, ImageDraw
import json
import logging
# Optional cryptography imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
from extractors.base import PIIEntity, PIIExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class RedactionConfig:
    """Configuration for redaction operations."""
    redaction_char: str = "█"
    replacement_patterns: Dict[str, str] = field(default_factory=lambda: {
        'person_name': '[NAME]',
        'email_address': '[EMAIL]',
        'phone_number': '[PHONE]',
        'address': '[ADDRESS]',
        'date_of_birth': '[DOB]',
        'social_security_number': '[SSN]',
        'credit_card_number': '[CARD]',
        'organization': '[ORG]',
        'location': '[LOCATION]',
        'misc_pii': '[PII]'
    })
    preserve_format: bool = True  # Maintain original text length/structure
    redaction_color: Tuple[int, int, int] = (0, 0, 0)  # Black for image redaction
    margin: int = 2  # Margin around redacted areas in images


@dataclass
class TokenizationConfig:
    """Configuration for tokenization operations."""
    use_format_preserving: bool = True
    preserve_length: bool = True
    use_deterministic: bool = False  # Same input -> same token
    token_prefix: str = "TOK_"
    encryption_key: Optional[bytes] = None
    salt_length: int = 16


class PIIRedactor:
    """Handles redaction of PII in text and images."""
    
    def __init__(self, config: RedactionConfig = None):
        """Initialize redactor."""
        self.config = config or RedactionConfig()
    
    def redact_text(self, text: str, entities: List[PIIEntity]) -> str:
        """Redact PII entities in text."""
        if not entities:
            return text
        
        # Sort entities by position (reverse order to maintain positions)
        sorted_entities = sorted(entities, key=lambda x: x.start_pos, reverse=True)
        
        redacted_text = text
        for entity in sorted_entities:
            if entity.start_pos >= 0 and entity.end_pos > entity.start_pos:
                replacement = self._get_replacement_text(entity)
                redacted_text = (
                    redacted_text[:entity.start_pos] + 
                    replacement + 
                    redacted_text[entity.end_pos:]
                )
        
        return redacted_text
    
    def redact_image(self, image: Image.Image, entities: List[PIIEntity]) -> Image.Image:
        """Redact PII entities in image using bounding boxes."""
        redacted_image = image.copy()
        draw = ImageDraw.Draw(redacted_image)
        
        for entity in entities:
            bbox = entity.metadata.get('bounding_box')
            if bbox:
                # Extract coordinates
                x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                
                # Add margin
                x1 = max(0, x1 - self.config.margin)
                y1 = max(0, y1 - self.config.margin)
                x2 = min(image.width, x2 + self.config.margin)
                y2 = min(image.height, y2 + self.config.margin)
                
                # Draw filled rectangle
                draw.rectangle(
                    [x1, y1, x2, y2],
                    fill=self.config.redaction_color
                )
        
        return redacted_image
    
    def _get_replacement_text(self, entity: PIIEntity) -> str:
        """Get replacement text for an entity."""
        replacement = self.config.replacement_patterns.get(
            entity.pii_type, 
            f"[{entity.pii_type.upper()}]"
        )
        
        if self.config.preserve_format:
            original_length = len(entity.text)
            
            if entity.pii_type == 'email_address':
                # Preserve email structure
                parts = entity.text.split('@')
                if len(parts) == 2:
                    username_len = len(parts[0])
                    domain_len = len(parts[1])
                    return f"{'█' * username_len}@{'█' * domain_len}"
            
            elif entity.pii_type == 'phone_number':
                # Preserve phone structure
                redacted = re.sub(r'\d', '█', entity.text)
                return redacted
            
            elif entity.pii_type == 'credit_card_number':
                # Show only last 4 digits
                digits_only = re.sub(r'\D', '', entity.text)
                if len(digits_only) >= 4:
                    masked = '█' * (len(digits_only) - 4) + digits_only[-4:]
                    # Restore original formatting
                    result = entity.text
                    digit_pos = 0
                    for i, char in enumerate(entity.text):
                        if char.isdigit():
                            if digit_pos < len(masked):
                                result = result[:i] + masked[digit_pos] + result[i+1:]
                                digit_pos += 1
                    return result
            
            # Default: replace with redaction characters of same length
            if len(replacement) != original_length:
                return self.config.redaction_char * original_length
        
        return replacement


class PIITokenizer:
    """Handles tokenization of PII for reversible anonymization."""
    
    def __init__(self, config: TokenizationConfig = None):
        """Initialize tokenizer."""
        self.config = config or TokenizationConfig()
        self.token_vault = {}  # Maps tokens to original values
        self.reverse_vault = {}  # Maps original values to tokens
        self.cipher_suite = None
        
        if self.config.encryption_key:
            self.cipher_suite = Fernet(self.config.encryption_key)
        else:
            self._generate_encryption_key()
    
    def _generate_encryption_key(self):
        """Generate encryption key for token vault."""
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(self.config.salt_length)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher_suite = Fernet(key)
        
        # Store salt for key regeneration if needed
        self.config.encryption_key = key
        self._salt = salt
    
    def tokenize_text(self, text: str, entities: List[PIIEntity]) -> Tuple[str, Dict[str, str]]:
        """Tokenize PII entities in text."""
        if not entities:
            return text, {}
        
        # Sort entities by position (reverse order)
        sorted_entities = sorted(entities, key=lambda x: x.start_pos, reverse=True)
        
        tokenized_text = text
        token_mapping = {}
        
        for entity in sorted_entities:
            if entity.start_pos >= 0 and entity.end_pos > entity.start_pos:
                token = self._generate_token(entity)
                token_mapping[token] = entity.text
                
                tokenized_text = (
                    tokenized_text[:entity.start_pos] + 
                    token + 
                    tokenized_text[entity.end_pos:]
                )
        
        return tokenized_text, token_mapping
    
    def detokenize_text(self, tokenized_text: str, token_mapping: Dict[str, str]) -> str:
        """Restore original text from tokenized version."""
        detokenized_text = tokenized_text
        
        for token, original_value in token_mapping.items():
            detokenized_text = detokenized_text.replace(token, original_value)
        
        return detokenized_text
    
    def _generate_token(self, entity: PIIEntity) -> str:
        """Generate token for a PII entity."""
        if self.config.use_deterministic and entity.text in self.reverse_vault:
            return self.reverse_vault[entity.text]
        
        # Create base token
        if self.config.use_format_preserving:
            token = self._generate_format_preserving_token(entity)
        else:
            token_id = secrets.token_hex(8)
            token = f"{self.config.token_prefix}{entity.pii_type.upper()}_{token_id}"
        
        # Store in vaults
        self.token_vault[token] = entity.text
        if self.config.use_deterministic:
            self.reverse_vault[entity.text] = token
        
        return token
    
    def _generate_format_preserving_token(self, entity: PIIEntity) -> str:
        """Generate format-preserving token."""
        original = entity.text
        
        if entity.pii_type == 'email_address':
            parts = original.split('@')
            if len(parts) == 2:
                username_token = self._random_string(len(parts[0]), alphanumeric=True)
                domain_parts = parts[1].split('.')
                domain_token = '.'.join([
                    self._random_string(len(part), alphanumeric=True) 
                    for part in domain_parts
                ])
                return f"{username_token}@{domain_token}"
        
        elif entity.pii_type == 'phone_number':
            # Preserve non-digit characters
            result = ""
            for char in original:
                if char.isdigit():
                    result += str(secrets.randbelow(10))
                else:
                    result += char
            return result
        
        elif entity.pii_type == 'person_name':
            # Generate name-like token
            words = original.split()
            token_words = []
            for word in words:
                if self.config.preserve_length:
                    token_word = self._random_string(len(word), alpha_only=True)
                    # Preserve capitalization pattern
                    if word and word[0].isupper():
                        token_word = token_word[0].upper() + token_word[1:].lower()
                    token_words.append(token_word)
                else:
                    token_words.append(self._random_string(6, alpha_only=True).title())
            return ' '.join(token_words)
        
        elif entity.pii_type == 'address':
            # Generate address-like token
            words = original.split()
            token_words = []
            for word in words:
                if word.isdigit():
                    token_words.append(str(secrets.randbelow(9999) + 1))
                else:
                    token_words.append(self._random_string(len(word), alpha_only=True).title())
            return ' '.join(token_words)
        
        # Default: random string of same length
        if self.config.preserve_length:
            return self._random_string(len(original))
        else:
            token_id = secrets.token_hex(6)
            return f"{self.config.token_prefix}{token_id}"
    
    def _random_string(self, length: int, alphanumeric: bool = False, alpha_only: bool = False) -> str:
        """Generate random string of specified length."""
        if alpha_only:
            chars = 'abcdefghijklmnopqrstuvwxyz'
        elif alphanumeric:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        else:
            chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        return ''.join(secrets.choice(chars) for _ in range(length))
    
    def save_token_vault(self, filepath: str):
        """Save encrypted token vault to file."""
        if not self.cipher_suite:
            raise ValueError("No encryption key available")
        
        vault_data = {
            'token_vault': self.token_vault,
            'reverse_vault': self.reverse_vault,
            'config': {
                'use_deterministic': self.config.use_deterministic,
                'token_prefix': self.config.token_prefix
            }
        }
        
        # Encrypt and save
        encrypted_data = self.cipher_suite.encrypt(
            json.dumps(vault_data).encode('utf-8')
        )
        
        with open(filepath, 'wb') as f:
            f.write(encrypted_data)
        
        logger.info(f"Token vault saved to {filepath}")
    
    def load_token_vault(self, filepath: str):
        """Load encrypted token vault from file."""
        if not self.cipher_suite:
            raise ValueError("No encryption key available")
        
        with open(filepath, 'rb') as f:
            encrypted_data = f.read()
        
        # Decrypt and load
        decrypted_data = self.cipher_suite.decrypt(encrypted_data)
        vault_data = json.loads(decrypted_data.decode('utf-8'))
        
        self.token_vault = vault_data['token_vault']
        self.reverse_vault = vault_data['reverse_vault']
        
        logger.info(f"Token vault loaded from {filepath}")


class DifferentialPrivacyProcessor:
    """Implements differential privacy techniques for PII protection."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """Initialize differential privacy processor."""
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta     # Probability of privacy breach
    
    def add_noise_to_counts(self, counts: Dict[str, int]) -> Dict[str, int]:
        """Add Laplace noise to entity counts."""
        sensitivity = 1  # Maximum change in count from one record
        scale = sensitivity / self.epsilon
        
        noisy_counts = {}
        for key, count in counts.items():
            noise = np.random.laplace(0, scale)
            noisy_count = max(0, int(count + noise))  # Ensure non-negative
            noisy_counts[key] = noisy_count
        
        return noisy_counts
    
    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Gaussian noise for differential privacy."""
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise


class PrivacyComplianceValidator:
    """Validates privacy compliance for different regulations."""
    
    def __init__(self):
        """Initialize compliance validator."""
        self.gdpr_requirements = {
            'data_minimization': False,
            'purpose_limitation': False,
            'storage_limitation': False,
            'accuracy': False,
            'security': False,
            'accountability': False
        }
        
        self.law25_requirements = {
            'consent': False,
            'notification': False,
            'data_protection_officer': False,
            'privacy_impact_assessment': False,
            'breach_notification': False
        }
    
    def validate_gdpr_compliance(self, processing_record: Dict[str, Any]) -> Dict[str, bool]:
        """Validate GDPR compliance."""
        compliance = self.gdpr_requirements.copy()
        
        # Check data minimization
        if processing_record.get('purpose_defined', False):
            compliance['data_minimization'] = True
            compliance['purpose_limitation'] = True
        
        # Check retention policy
        if processing_record.get('retention_policy', False):
            compliance['storage_limitation'] = True
        
        # Check data accuracy measures
        if processing_record.get('validation_enabled', False):
            compliance['accuracy'] = True
        
        # Check security measures
        security_measures = processing_record.get('security_measures', [])
        if 'encryption' in security_measures and 'access_control' in security_measures:
            compliance['security'] = True
        
        # Check documentation
        if processing_record.get('processing_documented', False):
            compliance['accountability'] = True
        
        return compliance
    
    def validate_law25_compliance(self, processing_record: Dict[str, Any]) -> Dict[str, bool]:
        """Validate Quebec Law 25 compliance."""
        compliance = self.law25_requirements.copy()
        
        # Check consent mechanism
        if processing_record.get('consent_obtained', False):
            compliance['consent'] = True
        
        # Check privacy policy
        if processing_record.get('privacy_policy_available', False):
            compliance['notification'] = True
        
        # Check DPO designation
        if processing_record.get('dpo_designated', False):
            compliance['data_protection_officer'] = True
        
        # Check PIA completion
        if processing_record.get('pia_completed', False):
            compliance['privacy_impact_assessment'] = True
        
        # Check breach procedures
        if processing_record.get('breach_procedures', False):
            compliance['breach_notification'] = True
        
        return compliance


class ComprehensivePrivacyProcessor:
    """Comprehensive privacy processing combining all techniques."""
    
    def __init__(self, 
                 redaction_config: RedactionConfig = None,
                 tokenization_config: TokenizationConfig = None,
                 enable_differential_privacy: bool = False):
        """Initialize comprehensive privacy processor."""
        self.redactor = PIIRedactor(redaction_config)
        self.tokenizer = PIITokenizer(tokenization_config)
        self.dp_processor = DifferentialPrivacyProcessor() if enable_differential_privacy else None
        self.compliance_validator = PrivacyComplianceValidator()
    
    def process_document(self, 
                        document: Dict[str, Any], 
                        extraction_result: PIIExtractionResult,
                        processing_mode: str = "redaction") -> Dict[str, Any]:
        """Process document with privacy-preserving techniques."""
        
        processed_document = document.copy()
        processing_metadata = {}
        
        if processing_mode == "redaction":
            # Redact text
            if 'text' in document:
                redacted_text = self.redactor.redact_text(
                    document['text'], 
                    extraction_result.pii_entities
                )
                processed_document['text'] = redacted_text
                processing_metadata['redaction_applied'] = True
            
            # Redact image
            if 'image' in document and isinstance(document['image'], Image.Image):
                redacted_image = self.redactor.redact_image(
                    document['image'], 
                    extraction_result.pii_entities
                )
                processed_document['image'] = redacted_image
                processing_metadata['image_redaction_applied'] = True
        
        elif processing_mode == "tokenization":
            # Tokenize text
            if 'text' in document:
                tokenized_text, token_mapping = self.tokenizer.tokenize_text(
                    document['text'], 
                    extraction_result.pii_entities
                )
                processed_document['text'] = tokenized_text
                processed_document['token_mapping'] = token_mapping
                processing_metadata['tokenization_applied'] = True
        
        elif processing_mode == "mixed":
            # Apply different techniques based on PII type
            sensitive_types = ['social_security_number', 'credit_card_number']
            redact_entities = [e for e in extraction_result.pii_entities 
                             if e.pii_type in sensitive_types]
            tokenize_entities = [e for e in extraction_result.pii_entities 
                               if e.pii_type not in sensitive_types]
            
            # Redact highly sensitive PII
            if redact_entities and 'text' in document:
                redacted_text = self.redactor.redact_text(
                    document['text'], redact_entities
                )
                processed_document['text'] = redacted_text
            
            # Tokenize other PII
            if tokenize_entities and 'text' in processed_document:
                tokenized_text, token_mapping = self.tokenizer.tokenize_text(
                    processed_document['text'], tokenize_entities
                )
                processed_document['text'] = tokenized_text
                processed_document['token_mapping'] = token_mapping
            
            processing_metadata['mixed_processing_applied'] = True
        
        # Add differential privacy if enabled
        if self.dp_processor:
            entity_counts = {}
            for entity in extraction_result.pii_entities:
                entity_counts[entity.pii_type] = entity_counts.get(entity.pii_type, 0) + 1
            
            noisy_counts = self.dp_processor.add_noise_to_counts(entity_counts)
            processing_metadata['dp_entity_counts'] = noisy_counts
        
        # Add processing metadata
        processed_document['privacy_processing'] = processing_metadata
        
        return processed_document