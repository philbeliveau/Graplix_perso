"""LLM-based OCR and text extraction module."""

from .llm_config import (
    LLMProvider,
    LLMModel,
    OCRTaskType,
    LLMOCRConfig,
    LLMModelRegistry,
    CostCalculator,
    LLMCostTracker,
    llm_config,
    cost_tracker
)

# Try to import LLM processor, but handle missing dependencies gracefully
try:
    from .llm_ocr_processor import (
        LLMOCRProcessor,
        llm_ocr_processor
    )
    LLM_PROCESSOR_AVAILABLE = True
except ImportError as e:
    # Create dummy classes for when dependencies aren't available
    class LLMOCRProcessor:
        def __init__(self):
            raise ImportError(f"LLM dependencies not available: {e}")
    
    llm_ocr_processor = None
    LLM_PROCESSOR_AVAILABLE = False

__all__ = [
    "LLMProvider",
    "LLMModel", 
    "OCRTaskType",
    "LLMOCRConfig",
    "LLMModelRegistry",
    "CostCalculator",
    "LLMCostTracker",
    "llm_config",
    "cost_tracker",
    "LLMOCRProcessor",
    "llm_ocr_processor",
    "LLM_PROCESSOR_AVAILABLE"
]