"""NER-based PII extractor using Hugging Face Transformers."""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

try:
    from transformers import (
        AutoTokenizer, AutoModelForTokenClassification, 
        pipeline, TokenClassificationPipeline
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from .base import PIIExtractorBase, PIIExtractionResult, PIIEntity
from ..core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class NERModelConfig:
    """Configuration for NER models."""
    name: str
    model_name: str
    supported_languages: List[str]
    entity_mapping: Dict[str, str]
    confidence_threshold: float = 0.7


class NERExtractor(PIIExtractorBase):
    """NER-based PII extraction using pre-trained models from Hugging Face."""
    
    # Predefined model configurations
    AVAILABLE_MODELS = {
        "multilingual_ner": NERModelConfig(
            name="multilingual_ner",
            model_name="dbmdz/bert-large-cased-finetuned-conll03-english",
            supported_languages=["en", "fr"],
            entity_mapping={
                "PER": "person_name",
                "PERSON": "person_name", 
                "ORG": "organization",
                "ORGANIZATION": "organization",
                "LOC": "location",
                "LOCATION": "location",
                "MISC": "miscellaneous"
            }
        ),
        "french_ner": NERModelConfig(
            name="french_ner",
            model_name="Jean-Baptiste/camembert-ner",
            supported_languages=["fr"],
            entity_mapping={
                "PER": "person_name",
                "ORG": "organization", 
                "LOC": "location",
                "MISC": "miscellaneous"
            }
        ),
        "distilbert_ner": NERModelConfig(
            name="distilbert_ner",
            model_name="distilbert-base-uncased",
            supported_languages=["en"],
            entity_mapping={
                "PERSON": "person_name",
                "ORG": "organization",
                "LOC": "location"
            }
        )
    }
    
    def __init__(self, model_name: str = "multilingual_ner"):
        """Initialize the NER extractor."""
        super().__init__("ner_extractor")
        
        if not HF_AVAILABLE:
            raise ImportError(
                "Hugging Face Transformers is required for NER extraction. "
                "Install with: pip install transformers torch"
            )
        
        self.model_config = self.AVAILABLE_MODELS.get(model_name)
        if not self.model_config:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
        # Initialize the model
        self._initialize_model()
        
        logger.info(f"NER extractor initialized with model: {self.model_config.model_name}")
    
    def _initialize_model(self):
        """Initialize the Hugging Face model and tokenizer."""
        try:
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_name,
                cache_dir="./data/models"
            )
            
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_config.model_name,
                cache_dir="./data/models"
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=-1  # Use CPU by default
            )
            
            logger.info(f"Successfully loaded model: {self.model_config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize NER model: {e}")
            raise
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """Extract PII using NER models."""
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
        
        try:
            # Process text in chunks to handle long documents
            chunks = self._split_text_into_chunks(text_content)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_entities = self._extract_entities_from_chunk(
                    chunk, chunk_idx, len(chunks)
                )
                entities.extend(chunk_entities)
                confidence_scores.extend([entity.confidence for entity in chunk_entities])
            
            # Post-process entities
            entities = self._post_process_entities(entities, text_content)
            
            processing_time = __import__('time').time() - start_time
            
            logger.info(f"NER extraction found {len(entities)} entities in {processing_time:.3f}s")
            
            return PIIExtractionResult(
                pii_entities=entities,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                metadata={
                    'extractor': self.name,
                    'model_name': self.model_config.model_name,
                    'text_length': len(text_content),
                    'chunks_processed': len(chunks)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            return PIIExtractionResult(
                pii_entities=[],
                confidence_scores=[],
                processing_time=__import__('time').time() - start_time,
                error=str(e)
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
    
    def _split_text_into_chunks(self, text: str, max_length: int = 512) -> List[str]:
        """Split text into chunks for processing."""
        # Simple sentence-based chunking to avoid breaking entities
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed max length
            if len(current_chunk) + len(sentence) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    # Single sentence is too long, split by words
                    words = sentence.split()
                    for word in words:
                        if len(current_chunk) + len(word) + 1 > max_length:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = word
                            else:
                                chunks.append(word)  # Single word longer than max_length
                        else:
                            current_chunk += " " + word if current_chunk else word
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:max_length]]
    
    def _extract_entities_from_chunk(self, 
                                   chunk: str, 
                                   chunk_idx: int, 
                                   total_chunks: int) -> List[PIIEntity]:
        """Extract entities from a single text chunk."""
        entities = []
        
        try:
            # Run NER pipeline
            ner_results = self.pipeline(chunk)
            
            for ner_entity in ner_results:
                # Map NER entity type to PII type
                pii_type = self._map_ner_to_pii_type(ner_entity['entity_group'])
                
                if pii_type and ner_entity['score'] >= self.model_config.confidence_threshold:
                    # Extract context around entity
                    start_pos = max(0, ner_entity['start'] - 50)
                    end_pos = min(len(chunk), ner_entity['end'] + 50)
                    context = chunk[start_pos:end_pos]
                    
                    # Create entity
                    entity = self._create_entity(
                        text=ner_entity['word'].strip(),
                        pii_type=pii_type,
                        confidence=float(ner_entity['score']),
                        start_pos=ner_entity['start'],
                        end_pos=ner_entity['end'],
                        context=context,
                        ner_label=ner_entity['entity_group'],
                        chunk_index=chunk_idx
                    )
                    
                    if self.validate_entity(entity):
                        entities.append(entity)
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_idx+1}/{total_chunks}: {e}")
        
        return entities
    
    def _map_ner_to_pii_type(self, ner_label: str) -> Optional[str]:
        """Map NER entity labels to PII types."""
        # Clean the label (remove B- I- prefixes)
        clean_label = re.sub(r'^[BI]-', '', ner_label.upper())
        
        # Get mapping from model config
        return self.model_config.entity_mapping.get(clean_label)
    
    def _post_process_entities(self, entities: List[PIIEntity], text: str) -> List[PIIEntity]:
        """Post-process entities to improve quality."""
        # Remove duplicates and merge overlapping entities
        unique_entities = []
        seen_entities = set()
        
        for entity in entities:
            # Create a key for deduplication
            entity_key = (entity.text.lower(), entity.pii_type)
            
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                
                # Additional validation
                if self._validate_ner_entity(entity):
                    unique_entities.append(entity)
        
        # Sort by confidence score (descending)
        unique_entities.sort(key=lambda e: e.confidence, reverse=True)
        
        return unique_entities
    
    def _validate_ner_entity(self, entity: PIIEntity) -> bool:
        """Additional validation for NER entities."""
        # Skip very short entities
        if len(entity.text.strip()) < 2:
            return False
        
        # Skip entities that are just punctuation or numbers
        if re.match(r'^[^a-zA-ZÀ-ÿ]+$', entity.text):
            return False
        
        # Person name validation
        if entity.pii_type == 'person_name':
            # Skip common false positives
            false_positives = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among', 'le', 'la', 'les', 'un',
                'une', 'des', 'du', 'de', 'et', 'ou', 'mais', 'dans', 'sur', 'avec'
            }
            
            if entity.text.lower() in false_positives:
                return False
            
            # Must contain at least one uppercase letter
            if not re.search(r'[A-ZÀ-Ÿ]', entity.text):
                return False
        
        # Organization validation
        if entity.pii_type == 'organization':
            # Must be at least 2 characters and contain letters
            if len(entity.text) < 2 or not re.search(r'[a-zA-ZÀ-ÿ]', entity.text):
                return False
        
        # Location validation
        if entity.pii_type == 'location':
            # Must contain letters and be at least 2 characters
            if len(entity.text) < 2 or not re.search(r'[a-zA-ZÀ-ÿ]', entity.text):
                return False
        
        return True
    
    def get_supported_pii_types(self) -> List[str]:
        """Get list of PII types this extractor can identify."""
        base_types = super().get_supported_pii_types()
        
        # Add NER-specific types
        ner_specific_types = [
            'person_name',
            'organization', 
            'location',
            'miscellaneous'
        ]
        
        # Combine and deduplicate
        all_types = list(set(base_types + ner_specific_types))
        return all_types
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_config:
            return {}
        
        return {
            'model_name': self.model_config.model_name,
            'supported_languages': self.model_config.supported_languages,
            'entity_mapping': self.model_config.entity_mapping,
            'confidence_threshold': self.model_config.confidence_threshold
        }
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all available models."""
        models_info = {}
        
        for model_key, config in cls.AVAILABLE_MODELS.items():
            models_info[model_key] = {
                'name': config.name,
                'model_name': config.model_name,
                'supported_languages': config.supported_languages,
                'confidence_threshold': config.confidence_threshold
            }
        
        return models_info