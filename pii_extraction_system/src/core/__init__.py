"""Core system components and pipeline management."""

# Import config first to avoid circular dependencies
from .config import settings
from .logging_config import get_logger, audit_log

# Import pipeline only when needed to avoid relative import issues
def get_pipeline():
    """Get PIIExtractionPipeline instance."""
    from .pipeline import PIIExtractionPipeline
    return PIIExtractionPipeline

__all__ = ['settings', 'get_logger', 'audit_log', 'get_pipeline']