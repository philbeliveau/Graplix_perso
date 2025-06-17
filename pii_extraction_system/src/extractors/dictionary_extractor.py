"""Dictionary-based PII extractor for known identifiers and specific terms."""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from dataclasses import dataclass, field

from .base import PIIExtractorBase, PIIExtractionResult, PIIEntity
from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DictionaryConfig:
    """Configuration for dictionary-based extraction."""
    name: str
    description: str
    pii_type: str
    terms: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    case_sensitive: bool = False
    word_boundaries: bool = True
    confidence_score: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'pii_type': self.pii_type,
            'terms': self.terms,
            'patterns': self.patterns,
            'case_sensitive': self.case_sensitive,
            'word_boundaries': self.word_boundaries,
            'confidence_score': self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DictionaryConfig':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            pii_type=data['pii_type'],
            terms=data.get('terms', []),
            patterns=data.get('patterns', []),
            case_sensitive=data.get('case_sensitive', False),
            word_boundaries=data.get('word_boundaries', True),
            confidence_score=data.get('confidence_score', 0.8)
        )


class DictionaryExtractor(PIIExtractorBase):
    """Dictionary-based PII extraction using lookup operations."""
    
    def __init__(self, dictionaries_path: Optional[Path] = None):
        """Initialize the dictionary extractor."""
        super().__init__("dictionary_extractor")
        
        # Set default dictionaries path
        if dictionaries_path is None:
            dictionaries_path = Path(__file__).parent.parent.parent / "data" / "dictionaries"
        
        self.dictionaries_path = Path(dictionaries_path)
        self.dictionaries: Dict[str, DictionaryConfig] = {}
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        
        # Load dictionaries
        self._load_dictionaries()
        
        # Create default dictionaries if none exist
        if not self.dictionaries:
            self._create_default_dictionaries()
        
        logger.info(f"Dictionary extractor initialized with {len(self.dictionaries)} dictionaries")
    
    def _load_dictionaries(self):
        """Load dictionary configurations from files."""
        if not self.dictionaries_path.exists():
            self.dictionaries_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created dictionaries directory: {self.dictionaries_path}")
            return
        
        # Load JSON dictionary files
        for dict_file in self.dictionaries_path.glob("*.json"):
            try:
                with open(dict_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # Multiple dictionaries in one file
                    for dict_data in data:
                        config = DictionaryConfig.from_dict(dict_data)
                        self.dictionaries[config.name] = config
                else:
                    # Single dictionary
                    config = DictionaryConfig.from_dict(data)
                    self.dictionaries[config.name] = config
                
                logger.info(f"Loaded dictionary from: {dict_file}")
                
            except Exception as e:
                logger.error(f"Error loading dictionary {dict_file}: {e}")
        
        # Compile patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for each dictionary."""
        for dict_name, config in self.dictionaries.items():
            patterns = []
            
            # Compile term patterns
            for term in config.terms:
                flags = 0 if config.case_sensitive else re.IGNORECASE
                
                if config.word_boundaries:
                    pattern = rf'\b{re.escape(term)}\b'
                else:
                    pattern = re.escape(term)
                
                try:
                    compiled_pattern = re.compile(pattern, flags)
                    patterns.append(compiled_pattern)
                except re.error as e:
                    logger.warning(f"Invalid term pattern '{term}' in {dict_name}: {e}")
            
            # Compile regex patterns
            for pattern_str in config.patterns:
                flags = 0 if config.case_sensitive else re.IGNORECASE
                
                try:
                    compiled_pattern = re.compile(pattern_str, flags)
                    patterns.append(compiled_pattern)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern_str}' in {dict_name}: {e}")
            
            self.compiled_patterns[dict_name] = patterns
    
    def _create_default_dictionaries(self):
        """Create default dictionary configurations."""
        default_dicts = [
            DictionaryConfig(
                name="healthcare_identifiers",
                description="Healthcare-related identifiers and terms",
                pii_type="medical_record_number",
                terms=[
                    "patient id", "patient identifier", "medical record number",
                    "mrn", "health card", "medicare number", "hospital number",
                    "identifiant patient", "numéro de dossier médical", "carte santé"
                ],
                patterns=[
                    r"patient\s*#?\s*\d{6,}",
                    r"mrn\s*:?\s*\d{6,}",
                    r"dossier\s*#?\s*\d{6,}"
                ],
                case_sensitive=False,
                confidence_score=0.85
            ),
            
            DictionaryConfig(
                name="government_identifiers",
                description="Government-issued identifiers",
                pii_type="government_id",
                terms=[
                    "social insurance number", "sin", "social security number", "ssn",
                    "driver's license", "driver license", "passport number",
                    "numéro d'assurance sociale", "nas", "permis de conduire",
                    "numéro de passeport"
                ],
                patterns=[
                    r"sin\s*:?\s*\d{3}[-\s]?\d{3}[-\s]?\d{3}",
                    r"ssn\s*:?\s*\d{3}[-\s]?\d{2}[-\s]?\d{4}",
                    r"nas\s*:?\s*\d{3}[-\s]?\d{3}[-\s]?\d{3}"
                ],
                case_sensitive=False,
                confidence_score=0.9
            ),
            
            DictionaryConfig(
                name="financial_identifiers",
                description="Financial account identifiers",
                pii_type="financial_account",
                terms=[
                    "account number", "account #", "bank account", "credit card",
                    "debit card", "iban", "swift", "routing number",
                    "numéro de compte", "compte bancaire", "carte de crédit",
                    "carte de débit"
                ],
                patterns=[
                    r"account\s*#?\s*:?\s*\d{8,}",
                    r"card\s*#?\s*:?\s*\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
                    r"iban\s*:?\s*[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}[A-Z0-9]{0,16}"
                ],
                case_sensitive=False,
                confidence_score=0.8
            ),
            
            DictionaryConfig(
                name="employee_identifiers",
                description="Employee and organizational identifiers",
                pii_type="employee_id",
                terms=[
                    "employee id", "employee number", "staff id", "badge number",
                    "worker id", "personnel number", "emp id", "emp #",
                    "identifiant employé", "numéro d'employé", "badge",
                    "numéro de personnel"
                ],
                patterns=[
                    r"emp\s*#?\s*:?\s*[A-Z]?\d{4,8}",
                    r"employee\s*#?\s*:?\s*[A-Z]?\d{4,8}",
                    r"badge\s*#?\s*:?\s*\d{4,8}"
                ],
                case_sensitive=False,
                confidence_score=0.75
            ),
            
            DictionaryConfig(
                name="contact_identifiers",
                description="Contact information identifiers",
                pii_type="contact_info",
                terms=[
                    "phone", "telephone", "mobile", "cell", "fax",
                    "email", "e-mail", "contact", "reach",
                    "téléphone", "mobile", "cellulaire", "courriel",
                    "contact", "joindre"
                ],
                patterns=[],
                case_sensitive=False,
                confidence_score=0.6
            ),
            
            DictionaryConfig(
                name="education_identifiers",
                description="Educational institution identifiers",
                pii_type="student_id",
                terms=[
                    "student id", "student number", "registration number",
                    "matriculation number", "student card", "school id",
                    "identifiant étudiant", "numéro d'étudiant", "matricule",
                    "carte étudiante"
                ],
                patterns=[
                    r"student\s*#?\s*:?\s*[A-Z]?\d{6,10}",
                    r"matricul\w*\s*:?\s*[A-Z]?\d{6,10}"
                ],
                case_sensitive=False,
                confidence_score=0.8
            )
        ]
        
        # Add default dictionaries
        for config in default_dicts:
            self.dictionaries[config.name] = config
        
        # Compile patterns
        self._compile_patterns()
        
        # Save default dictionaries
        self._save_default_dictionaries(default_dicts)
        
        logger.info("Created default dictionaries")
    
    def _save_default_dictionaries(self, configs: List[DictionaryConfig]):
        """Save default dictionaries to files."""
        try:
            self.dictionaries_path.mkdir(parents=True, exist_ok=True)
            
            output_file = self.dictionaries_path / "default_dictionaries.json"
            data = [config.to_dict() for config in configs]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved default dictionaries to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving default dictionaries: {e}")
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """Extract PII using dictionary lookup."""
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
        
        # Apply each dictionary
        for dict_name, config in self.dictionaries.items():
            dict_entities = self._extract_entities_with_dictionary(
                text_content, dict_name, config
            )
            entities.extend(dict_entities)
            confidence_scores.extend([entity.confidence for entity in dict_entities])
        
        # Post-process entities
        entities = self._post_process_entities(entities, text_content)
        
        processing_time = __import__('time').time() - start_time
        
        logger.info(f"Dictionary extraction found {len(entities)} entities in {processing_time:.3f}s")
        
        return PIIExtractionResult(
            pii_entities=entities,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            metadata={
                'extractor': self.name,
                'dictionaries_used': list(self.dictionaries.keys()),
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
    
    def _extract_entities_with_dictionary(self,
                                        text: str,
                                        dict_name: str,
                                        config: DictionaryConfig) -> List[PIIEntity]:
        """Extract entities using a specific dictionary."""
        entities = []
        patterns = self.compiled_patterns.get(dict_name, [])
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                matched_text = match.group().strip()
                start_pos = match.start()
                end_pos = match.end()
                
                # Skip if text is too short
                if len(matched_text) < 2:
                    continue
                
                # Calculate confidence with context
                confidence = self._calculate_confidence(
                    matched_text, config, text, start_pos
                )
                
                # Extract context
                context = self._extract_context(text, start_pos, end_pos)
                
                # Create entity
                entity = self._create_entity(
                    text=matched_text,
                    pii_type=config.pii_type,
                    confidence=confidence,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    context=context,
                    dictionary_used=dict_name,
                    dictionary_description=config.description
                )
                
                # Validate entity before adding
                if self.validate_entity(entity):
                    entities.append(entity)
        
        return entities
    
    def _calculate_confidence(self,
                            matched_text: str,
                            config: DictionaryConfig,
                            full_text: str,
                            position: int) -> float:
        """Calculate confidence score for a dictionary match."""
        base_confidence = config.confidence_score
        
        # Adjust based on context
        context_window = 100
        start_context = max(0, position - context_window)
        end_context = min(len(full_text), position + len(matched_text) + context_window)
        context = full_text[start_context:end_context].lower()
        
        # Context keywords that increase confidence
        positive_keywords = {
            'medical_record_number': ['patient', 'medical', 'hospital', 'clinic', 'doctor'],
            'government_id': ['government', 'official', 'issued', 'identification'],
            'financial_account': ['bank', 'financial', 'payment', 'transaction'],
            'employee_id': ['employee', 'staff', 'work', 'company', 'organization'],
            'contact_info': ['contact', 'reach', 'call', 'email', 'phone'],
            'student_id': ['student', 'school', 'university', 'college', 'education']
        }
        
        if config.pii_type in positive_keywords:
            for keyword in positive_keywords[config.pii_type]:
                if keyword in context:
                    base_confidence = min(1.0, base_confidence + 0.1)
                    break
        
        # Reduce confidence if text seems like a false positive
        false_positive_patterns = [
            r'^\d+$',  # Just numbers
            r'^[a-zA-Z]+$',  # Just letters
            r'^[^\w\s]+$'  # Just symbols
        ]
        
        for fp_pattern in false_positive_patterns:
            if re.match(fp_pattern, matched_text):
                base_confidence *= 0.8
                break
        
        return max(0.0, min(1.0, base_confidence))
    
    def _post_process_entities(self, entities: List[PIIEntity], text: str) -> List[PIIEntity]:
        """Post-process entities to remove duplicates and improve quality."""
        # Remove exact duplicates
        seen = set()
        unique_entities = []
        
        for entity in entities:
            entity_key = (entity.text.lower(), entity.pii_type, entity.start_pos)
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        # Sort by confidence score (descending)
        unique_entities.sort(key=lambda e: e.confidence, reverse=True)
        
        # Additional validation
        validated_entities = []
        for entity in unique_entities:
            if self._additional_validation(entity):
                validated_entities.append(entity)
        
        return validated_entities
    
    def _additional_validation(self, entity: PIIEntity) -> bool:
        """Additional validation for dictionary entities."""
        # Skip very short entities
        if len(entity.text.strip()) < 2:
            return False
        
        # Skip entities that are mostly punctuation
        if len(re.sub(r'[^\w\s]', '', entity.text)) < len(entity.text) * 0.3:
            return False
        
        # PII type specific validation
        if entity.pii_type == 'medical_record_number':
            # Should contain some numbers
            if not re.search(r'\d', entity.text):
                return False
        
        if entity.pii_type == 'financial_account':
            # Should contain numbers for account info
            if not re.search(r'\d{4,}', entity.text):
                return False
        
        return True
    
    def add_dictionary(self, config: DictionaryConfig) -> bool:
        """Add a new dictionary configuration."""
        try:
            self.dictionaries[config.name] = config
            
            # Compile patterns for this dictionary
            patterns = []
            
            for term in config.terms:
                flags = 0 if config.case_sensitive else re.IGNORECASE
                
                if config.word_boundaries:
                    pattern = rf'\b{re.escape(term)}\b'
                else:
                    pattern = re.escape(term)
                
                compiled_pattern = re.compile(pattern, flags)
                patterns.append(compiled_pattern)
            
            for pattern_str in config.patterns:
                flags = 0 if config.case_sensitive else re.IGNORECASE
                compiled_pattern = re.compile(pattern_str, flags)
                patterns.append(compiled_pattern)
            
            self.compiled_patterns[config.name] = patterns
            
            logger.info(f"Added dictionary: {config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding dictionary {config.name}: {e}")
            return False
    
    def remove_dictionary(self, dict_name: str) -> bool:
        """Remove a dictionary configuration."""
        try:
            if dict_name in self.dictionaries:
                del self.dictionaries[dict_name]
                
            if dict_name in self.compiled_patterns:
                del self.compiled_patterns[dict_name]
            
            logger.info(f"Removed dictionary: {dict_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing dictionary {dict_name}: {e}")
            return False
    
    def get_dictionary_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded dictionaries."""
        info = {}
        
        for dict_name, config in self.dictionaries.items():
            info[dict_name] = {
                'description': config.description,
                'pii_type': config.pii_type,
                'terms_count': len(config.terms),
                'patterns_count': len(config.patterns),
                'confidence_score': config.confidence_score,
                'case_sensitive': config.case_sensitive,
                'word_boundaries': config.word_boundaries
            }
        
        return info
    
    def get_supported_pii_types(self) -> List[str]:
        """Get list of PII types this extractor can identify."""
        base_types = super().get_supported_pii_types()
        
        # Add dictionary-specific types
        dict_types = [config.pii_type for config in self.dictionaries.values()]
        
        # Combine and deduplicate
        all_types = list(set(base_types + dict_types))
        return all_types