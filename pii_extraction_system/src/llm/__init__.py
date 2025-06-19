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

# Import new multi-LLM integration components
try:
    from .multimodal_llm_service import MultimodalLLMService, llm_service
    from .api_key_manager import APIKeyManager, api_key_manager
    from .cost_tracker import CostTracker as EnhancedCostTracker, default_cost_tracker, TokenUsageMonitor, token_monitor
    from .api_integration_wrapper import MultiLLMIntegrationWrapper, llm_integration, LLMModelInfo
    MULTI_LLM_AVAILABLE = True
except ImportError as e:
    # Create dummy classes for when dependencies aren't available
    MultimodalLLMService = None
    llm_service = None
    APIKeyManager = None
    api_key_manager = None
    EnhancedCostTracker = None
    default_cost_tracker = None
    TokenUsageMonitor = None
    token_monitor = None
    MultiLLMIntegrationWrapper = None
    llm_integration = None
    LLMModelInfo = None
    MULTI_LLM_AVAILABLE = False

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
    "LLM_PROCESSOR_AVAILABLE",
    # Multi-LLM integration exports
    "MultimodalLLMService",
    "llm_service",
    "APIKeyManager",
    "api_key_manager",
    "EnhancedCostTracker",
    "default_cost_tracker",
    "TokenUsageMonitor",
    "token_monitor",
    "MultiLLMIntegrationWrapper",
    "llm_integration",
    "LLMModelInfo",
    "MULTI_LLM_AVAILABLE"
]