"""Rule-based PII extractor using regex patterns."""

import re
from typing import Dict, List, Any

from .base import PIIExtractorBase, PIIExtractionResult, PIIEntity
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class RuleBasedExtractor(PIIExtractorBase):
    """Rule-based PII extraction using regular expressions."""
    
    def __init__(self):
        """Initialize the rule-based extractor."""
        super().__init__("rule_based")
        self.patterns = self._compile_patterns()
        logger.info("Rule-based extractor initialized with regex patterns")
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile all regex patterns for PII detection."""
        patterns = {}
        
        # Email addresses
        patterns['email_address'] = [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.MULTILINE)
        ]
        
        # Phone numbers (various formats for North America and international)
        patterns['phone_number'] = [
            # North American formats
            re.compile(r'\b(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            re.compile(r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'),
            # International formats
            re.compile(r'\+\d{1,3}[-.\s]?\d{1,14}\b'),
            # French formats
            re.compile(r'\b0[1-9](?:[-.\s]?\d{2}){4}\b'),
            re.compile(r'\+33[-.\s]?[1-9](?:[-.\s]?\d{2}){4}\b')
        ]
        
        # Social Security Numbers (US and Canada)
        patterns['social_security_number'] = [
            # US SSN
            re.compile(r'\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b'),
            # Canadian SIN
            re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3}\b')
        ]
        
        # Credit card numbers
        patterns['credit_card_number'] = [
            # Visa (4xxx)
            re.compile(r'\b4\d{3}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'),
            # MasterCard (5xxx)
            re.compile(r'\b5\d{3}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'),
            # American Express (3xxx)
            re.compile(r'\b3[47]\d{2}[-.\s]?\d{6}[-.\s]?\d{5}\b'),
            # General 16-digit pattern
            re.compile(r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b')
        ]
        
        # Dates (various formats)
        patterns['date_of_birth'] = [
            # MM/DD/YYYY
            re.compile(r'\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(19|20)\d{2}\b'),
            # DD/MM/YYYY
            re.compile(r'\b(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/(19|20)\d{2}\b'),
            # YYYY-MM-DD
            re.compile(r'\b(19|20)\d{2}-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b'),
            # DD-MM-YYYY
            re.compile(r'\b(0?[1-9]|[12]\d|3[01])-(0?[1-9]|1[0-2])-(19|20)\d{2}\b'),
            # Month DD, YYYY
            re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(0?[1-9]|[12]\d|3[01]),?\s+(19|20)\d{2}\b', re.IGNORECASE),
            # DD Month YYYY
            re.compile(r'\b(0?[1-9]|[12]\d|3[01])\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(19|20)\d{2}\b', re.IGNORECASE),
            # French months
            re.compile(r'\b(0?[1-9]|[12]\d|3[01])\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(19|20)\d{2}\b', re.IGNORECASE)
        ]
        
        # Canadian postal codes
        patterns['postal_code'] = [
            re.compile(r'\b[A-Z]\d[A-Z][-.\s]?\d[A-Z]\d\b', re.IGNORECASE),
        ]
        
        # US ZIP codes
        patterns['zip_code'] = [
            re.compile(r'\b\d{5}(-\d{4})?\b'),
        ]
        
        # Driver's license numbers (simplified patterns for common formats)
        patterns['driver_license'] = [
            # Various alphanumeric patterns
            re.compile(r'\b[A-Z]\d{8}\b'),  # X12345678
            re.compile(r'\b\d{8,9}\b'),     # 12345678 or 123456789
            re.compile(r'\b[A-Z]{2}\d{6}\b'),  # XX123456
        ]
        
        # IP addresses
        patterns['ip_address'] = [
            re.compile(r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'),
            # IPv6 (simplified)
            re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b')
        ]
        
        # URLs
        patterns['url'] = [
            re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?', re.IGNORECASE),
            re.compile(r'www\.(?:[-\w.])+\.(?:[a-z]{2,4})(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?', re.IGNORECASE)
        ]
        
        # IBAN (International Bank Account Number)
        patterns['iban'] = [
            re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b')
        ]
        
        # Medical record numbers (simplified patterns)
        patterns['medical_record_number'] = [
            re.compile(r'\bMRN[-.\s]?\d{6,10}\b', re.IGNORECASE),
            re.compile(r'\bMR[-.\s]?\d{6,10}\b', re.IGNORECASE),
            re.compile(r'\bpatient[-.\s]?id[-.\s]?\d{6,10}\b', re.IGNORECASE)
        ]
        
        # Employee IDs
        patterns['employee_id'] = [
            re.compile(r'\bEMP[-.\s]?\d{4,8}\b', re.IGNORECASE),
            re.compile(r'\bemployee[-.\s]?id[-.\s]?\d{4,8}\b', re.IGNORECASE),
            re.compile(r'\bE\d{4,8}\b')
        ]
        
        # Person names (basic patterns for common formats)
        patterns['person_name'] = [
            # Title + First + Last
            re.compile(r'\b(Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b'),
            # First + Last (capitalized)
            re.compile(r'\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b'),
            # French titles
            re.compile(r'\b(M\.?|Mme\.?|Mlle\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b')
        ]
        
        # Addresses (simplified patterns)
        patterns['address'] = [
            # Street address with number
            re.compile(r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Lane|Ln\.?|Boulevard|Blvd\.?)\b', re.IGNORECASE),
            # French addresses
            re.compile(r'\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(rue|avenue|boulevard|place|square|impasse)\b', re.IGNORECASE)
        ]
        
        return patterns
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """Extract PII using rule-based patterns."""
        start_time = __import__('time').time()
        
        # Get text content from document
        text_content = self._extract_text_content(document)
        
        if not text_content:
            logger.warning("No text content found in document")
            return PIIExtractionResult(
                pii_entities=[],
                confidence_scores=[],
                processing_time=__import__('time').time() - start_time,
                error="No text content available"
            )
        
        entities = []
        confidence_scores = []
        
        # Apply each pattern type
        for pii_type, patterns in self.patterns.items():
            type_entities = self._extract_entities_by_type(
                text_content, pii_type, patterns
            )
            entities.extend(type_entities)
            confidence_scores.extend([entity.confidence for entity in type_entities])
        
        # Post-process entities
        entities = self._post_process_entities(entities, text_content)
        
        processing_time = __import__('time').time() - start_time
        
        logger.info(f"Rule-based extraction found {len(entities)} entities in {processing_time:.3f}s")
        
        return PIIExtractionResult(
            pii_entities=entities,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            metadata={
                'extractor': self.name,
                'patterns_used': list(self.patterns.keys()),
                'text_length': len(text_content)
            }
        )
    
    def _extract_text_content(self, document: Dict[str, Any]) -> str:
        """Extract all text content from a processed document."""
        text_parts = []
        
        # Raw text
        if 'raw_text' in document and document['raw_text']:
            text_parts.append(document['raw_text'])
        
        # OCR text if available and raw text is insufficient
        if 'ocr_text' in document and document['ocr_text']:
            if not text_parts or len(text_parts[0]) < 50:
                text_parts.append(document['ocr_text'])
        
        # Combine all text
        return '\n'.join(text_parts)
    
    def _extract_entities_by_type(self, 
                                 text: str, 
                                 pii_type: str, 
                                 patterns: List[re.Pattern]) -> List[PIIEntity]:
        """Extract entities of a specific type using given patterns."""
        entities = []
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                # Get match details
                matched_text = match.group().strip()
                start_pos = match.start()
                end_pos = match.end()
                
                # Skip if text is too short or invalid
                if len(matched_text) < 2:
                    continue
                
                # Calculate confidence based on pattern specificity and context
                confidence = self._calculate_confidence(matched_text, pii_type, text, start_pos)
                
                # Extract context
                context = self._extract_context(text, start_pos, end_pos)
                
                # Create entity
                entity = self._create_entity(
                    text=matched_text,
                    pii_type=pii_type,
                    confidence=confidence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    context=context,
                    pattern_used=pattern.pattern
                )
                
                # Validate entity before adding
                if self.validate_entity(entity):
                    entities.append(entity)
        
        return entities
    
    def _calculate_confidence(self, 
                            matched_text: str, 
                            pii_type: str, 
                            full_text: str, 
                            position: int) -> float:
        """Calculate confidence score for a matched entity."""
        base_confidence = 0.7  # Base confidence for regex matches
        
        # Adjust based on PII type specificity
        type_confidence_adjustments = {
            'email_address': 0.9,
            'phone_number': 0.8,
            'social_security_number': 0.9,
            'credit_card_number': 0.8,
            'url': 0.9,
            'ip_address': 0.8,
            'iban': 0.85,
            'date_of_birth': 0.6,  # Lower because dates are common
            'person_name': 0.5,    # Lower due to false positives
            'address': 0.6,
            'driver_license': 0.7,
            'medical_record_number': 0.8,
            'employee_id': 0.7,
            'postal_code': 0.8,
            'zip_code': 0.8
        }
        
        confidence = type_confidence_adjustments.get(pii_type, base_confidence)
        
        # Adjust based on context clues
        context_window = 100
        start_context = max(0, position - context_window)
        end_context = min(len(full_text), position + len(matched_text) + context_window)
        context = full_text[start_context:end_context].lower()
        
        # Context keywords that increase confidence
        positive_keywords = {
            'email_address': ['email', 'mail', 'contact', 'courriel', '@'],
            'phone_number': ['phone', 'tel', 'telephone', 'call', 'numéro', 'tél'],
            'social_security_number': ['ssn', 'social security', 'nas', 'sin'],
            'date_of_birth': ['birth', 'born', 'dob', 'naissance', 'né'],
            'person_name': ['name', 'nom', 'patient', 'client', 'employee'],
            'address': ['address', 'adresse', 'street', 'rue', 'city', 'ville'],
            'credit_card_number': ['card', 'credit', 'carte', 'payment'],
            'driver_license': ['license', 'licence', 'driver', 'permit']
        }
        
        if pii_type in positive_keywords:
            for keyword in positive_keywords[pii_type]:
                if keyword in context:
                    confidence = min(1.0, confidence + 0.1)
                    break
        
        # Reduce confidence for very common patterns that might be false positives
        if pii_type == 'person_name':
            common_false_positives = ['John Doe', 'Jane Doe', 'Test Test', 'Example Example']
            if matched_text in common_false_positives:
                confidence = 0.2
        
        # Ensure confidence remains in valid range
        return max(0.0, min(1.0, confidence))
    
    def _post_process_entities(self, entities: List[PIIEntity], text: str) -> List[PIIEntity]:
        """Post-process entities to remove duplicates and improve quality."""
        # Remove exact duplicates
        seen = set()
        unique_entities = []
        
        for entity in entities:
            entity_key = (entity.text, entity.pii_type, entity.start_pos)
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        # Sort by position for consistent ordering
        unique_entities.sort(key=lambda e: e.start_pos)
        
        # Apply additional validation
        validated_entities = []
        for entity in unique_entities:
            if self._additional_validation(entity):
                validated_entities.append(entity)
        
        return validated_entities
    
    def _additional_validation(self, entity: PIIEntity) -> bool:
        """Additional validation for specific PII types."""
        
        # Email validation
        if entity.pii_type == 'email_address':
            return '@' in entity.text and '.' in entity.text.split('@')[-1]
        
        # Phone number validation
        if entity.pii_type == 'phone_number':
            digits = re.sub(r'\D', '', entity.text)
            return 10 <= len(digits) <= 15
        
        # Credit card validation (basic Luhn algorithm check)
        if entity.pii_type == 'credit_card_number':
            digits = re.sub(r'\D', '', entity.text)
            return len(digits) in [13, 14, 15, 16, 19] and self._luhn_check(digits)
        
        # SSN validation
        if entity.pii_type == 'social_security_number':
            digits = re.sub(r'\D', '', entity.text)
            return len(digits) in [9, 10]  # US SSN or Canadian SIN
        
        return True
    
    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        try:
            digits = [int(d) for d in card_number]
            for i in range(len(digits) - 2, -1, -2):
                digits[i] *= 2
                if digits[i] > 9:
                    digits[i] -= 9
            return sum(digits) % 10 == 0
        except (ValueError, IndexError):
            return False