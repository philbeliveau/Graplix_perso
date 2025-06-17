"""Base classes for PII extraction."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class PIIEntity:
    """Represents a single PII entity found in a document."""
    
    text: str                          # The actual PII text
    pii_type: str                     # Type of PII (email, phone, name, etc.)
    confidence: float                 # Confidence score (0-1)
    start_pos: int = 0               # Start position in text
    end_pos: int = 0                 # End position in text
    context: str = ""                # Surrounding context
    extractor: str = ""              # Which extractor found this
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'type': self.pii_type,
            'confidence': self.confidence,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'context': self.context,
            'extractor': self.extractor,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PIIEntity':
        """Create from dictionary."""
        return cls(
            text=data['text'],
            pii_type=data.get('type', data.get('pii_type', 'unknown')),
            confidence=data['confidence'],
            start_pos=data.get('start_pos', 0),
            end_pos=data.get('end_pos', 0),
            context=data.get('context', ''),
            extractor=data.get('extractor', ''),
            metadata=data.get('metadata', {})
        )


@dataclass
class PIIExtractionResult:
    """Results from PII extraction process."""
    
    pii_entities: List[PIIEntity] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pii_entities': [entity.to_dict() for entity in self.pii_entities],
            'confidence_scores': self.confidence_scores,
            'processing_time': self.processing_time,
            'error': self.error,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PIIExtractionResult':
        """Create from dictionary."""
        entities = [PIIEntity.from_dict(e) for e in data.get('pii_entities', [])]
        
        timestamp = datetime.now()
        if 'timestamp' in data:
            try:
                timestamp = datetime.fromisoformat(data['timestamp'])
            except ValueError:
                pass
        
        return cls(
            pii_entities=entities,
            confidence_scores=data.get('confidence_scores', []),
            processing_time=data.get('processing_time', 0.0),
            error=data.get('error'),
            metadata=data.get('metadata', {}),
            timestamp=timestamp
        )
    
    def get_entities_by_type(self, pii_type: str) -> List[PIIEntity]:
        """Get all entities of a specific type."""
        return [entity for entity in self.pii_entities if entity.pii_type == pii_type]
    
    def get_unique_entity_types(self) -> List[str]:
        """Get list of unique PII types found."""
        return list(set(entity.pii_type for entity in self.pii_entities))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        if not self.pii_entities:
            return {
                'total_entities': 0,
                'unique_types': 0,
                'avg_confidence': 0.0,
                'processing_time': self.processing_time
            }
        
        confidence_scores = [entity.confidence for entity in self.pii_entities]
        type_counts = {}
        
        for entity in self.pii_entities:
            type_counts[entity.pii_type] = type_counts.get(entity.pii_type, 0) + 1
        
        return {
            'total_entities': len(self.pii_entities),
            'unique_types': len(self.get_unique_entity_types()),
            'type_distribution': type_counts,
            'avg_confidence': sum(confidence_scores) / len(confidence_scores),
            'min_confidence': min(confidence_scores),
            'max_confidence': max(confidence_scores),
            'processing_time': self.processing_time
        }


class PIIExtractorBase(ABC):
    """Abstract base class for all PII extractors."""
    
    def __init__(self, name: str):
        """Initialize the extractor."""
        self.name = name
        self.supported_languages = ['en', 'fr']  # Default to English and French
    
    @abstractmethod
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """
        Extract PII from a processed document.
        
        Args:
            document: Processed document dictionary containing text and metadata
            
        Returns:
            PIIExtractionResult with found entities
        """
        pass
    
    def _create_entity(self,
                      text: str,
                      pii_type: str,
                      confidence: float,
                      start_pos: int = 0,
                      end_pos: int = 0,
                      context: str = "",
                      **metadata) -> PIIEntity:
        """Helper method to create a PII entity."""
        return PIIEntity(
            text=text,
            pii_type=pii_type,
            confidence=confidence,
            start_pos=start_pos,
            end_pos=end_pos,
            context=context,
            extractor=self.name,
            metadata=metadata
        )
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, 
                        context_window: int = 50) -> str:
        """Extract context around a PII entity."""
        context_start = max(0, start_pos - context_window)
        context_end = min(len(text), end_pos + context_window)
        
        context = text[context_start:context_end]
        
        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
        
        return context
    
    def _time_extraction(self, func, *args, **kwargs) -> tuple:
        """Time the extraction process."""
        start_time = time.time()
        result = func(*args, **kwargs)
        processing_time = time.time() - start_time
        return result, processing_time
    
    def get_supported_pii_types(self) -> List[str]:
        """Get list of PII types this extractor can identify."""
        return [
            'person_name',
            'email_address',
            'phone_number',
            'address',
            'date_of_birth',
            'social_security_number',
            'credit_card_number',
            'driver_license',
            'passport_number',
            'bank_account',
            'organization',
            'location',
            'url',
            'ip_address',
            'iban',
            'medical_record_number',
            'employee_id'
        ]
    
    def validate_entity(self, entity: PIIEntity) -> bool:
        """Validate a PII entity before adding to results."""
        # Basic validation
        if not entity.text or not entity.text.strip():
            return False
        
        if entity.confidence < 0 or entity.confidence > 1:
            return False
        
        if entity.pii_type not in self.get_supported_pii_types():
            return False
        
        return True