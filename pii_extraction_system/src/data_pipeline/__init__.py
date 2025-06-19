"""
Data Pipeline Module for Dashboard Refactoring

This module provides robust data handling capabilities including:
- Document loading from various sources
- Multi-format support (PDF, JPG, Excel, DOC, TXT)
- Batch processing capabilities
- Ground truth management system
- Experiment results save/load functionality
- Metadata tagging system
"""

__version__ = "1.0.0"

from .data_loader import DataLoader, DocumentMetadata
from .format_handlers import (
    FormatHandler, PDFHandler, ImageHandler, ExcelHandler, 
    DocHandler, TextHandler, FormatHandlerRegistry
)
from .batch_processor import BatchProcessor, BatchResult
from .ground_truth_manager import GroundTruthManager, GroundTruthEntry
from .experiment_tracker import ExperimentTracker, ExperimentResult
from .metadata_manager import MetadataManager, DocumentTag

__all__ = [
    'DataLoader', 'DocumentMetadata',
    'FormatHandler', 'PDFHandler', 'ImageHandler', 'ExcelHandler', 
    'DocHandler', 'TextHandler', 'FormatHandlerRegistry',
    'BatchProcessor', 'BatchResult',
    'GroundTruthManager', 'GroundTruthEntry',
    'ExperimentTracker', 'ExperimentResult',
    'MetadataManager', 'DocumentTag'
]