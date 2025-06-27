"""PII Extraction System - A comprehensive solution for document PII extraction."""

__version__ = "0.1.0"
__author__ = "AI Development Team"
__description__ = "Comprehensive PII extraction system with ML and privacy compliance"

from .core import settings, get_pipeline
from .extractors import PIIExtractorBase, PIIEntity, PIIExtractionResult
from .utils import DocumentProcessor, storage_manager

__all__ = [
    'get_pipeline', 'settings',
    'PIIExtractorBase', 'PIIEntity', 'PIIExtractionResult',
    'DocumentProcessor', 'storage_manager'
]