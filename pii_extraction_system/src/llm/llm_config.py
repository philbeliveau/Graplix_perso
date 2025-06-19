"""LLM configuration and cost analysis for OCR enhancement."""

import os
from datetime import datetime
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    NVIDIA = "nvidia"


class LLMModel(BaseModel):
    """LLM model configuration."""
    
    provider: LLMProvider
    model_name: str
    display_name: str
    input_cost_per_1k_tokens: float = Field(description="Cost per 1K input tokens in USD")
    output_cost_per_1k_tokens: float = Field(description="Cost per 1K output tokens in USD")
    max_tokens: int = Field(default=4096, description="Maximum tokens per request")
    supports_vision: bool = Field(default=False, description="Supports image inputs")
    quality_score: float = Field(default=0.8, description="Quality score 0-1")
    speed_score: float = Field(default=0.8, description="Speed score 0-1")
    description: str = Field(default="", description="Model description")


class OCRTaskType(str, Enum):
    """Types of OCR tasks."""
    BASIC_TEXT_EXTRACTION = "basic_text"
    STRUCTURED_DATA_EXTRACTION = "structured_data"
    HANDWRITING_RECOGNITION = "handwriting"
    TABLE_EXTRACTION = "table_extraction"
    DOCUMENT_ANALYSIS = "document_analysis"


class LLMOCRConfig(BaseModel):
    """Configuration for LLM-based OCR."""
    
    enabled_models: List[str] = Field(default_factory=list)
    default_model: str = Field(default="gpt-4o-mini")
    fallback_model: str = Field(default="gpt-4o-mini")
    
    # Task-specific model mapping
    task_model_mapping: Dict[OCRTaskType, str] = Field(default_factory=dict)
    
    # Quality thresholds
    min_confidence_threshold: float = Field(default=0.7)
    use_ensemble_for_critical: bool = Field(default=True)
    
    # Cost controls
    max_cost_per_document: float = Field(default=0.50, description="Max cost in USD per document")
    enable_cost_optimization: bool = Field(default=True)
    prefer_cheaper_models: bool = Field(default=True)
    
    # Performance settings
    max_retry_attempts: int = Field(default=3)
    timeout_seconds: int = Field(default=60)
    batch_processing: bool = Field(default=True)


class LLMModelRegistry:
    """Registry of available LLM models with pricing and capabilities."""
    
    MODELS = {
        # OpenAI Models
        "gpt-3.5-turbo": LLMModel(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo",
            input_cost_per_1k_tokens=0.0005,
            output_cost_per_1k_tokens=0.0015,
            max_tokens=4096,
            supports_vision=False,
            quality_score=0.85,
            speed_score=0.9,
            description="Fast and cost-effective for most OCR tasks"
        ),
        "gpt-4o-mini": LLMModel(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            display_name="GPT-4o Mini",
            input_cost_per_1k_tokens=0.00015,
            output_cost_per_1k_tokens=0.0006,
            max_tokens=128000,
            supports_vision=True,
            quality_score=0.88,
            speed_score=0.85,
            description="Most cost-effective vision model for OCR"
        ),
        # Note: gpt-4-vision-preview has been deprecated by OpenAI
        # Using gpt-4o-mini as the primary vision model instead
        
        # Anthropic Models
        "claude-3-haiku": LLMModel(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-haiku-20240307",
            display_name="Claude 3 Haiku",
            input_cost_per_1k_tokens=0.00025,
            output_cost_per_1k_tokens=0.00125,
            max_tokens=4096,
            supports_vision=True,
            quality_score=0.82,
            speed_score=0.95,
            description="Ultra-fast and affordable vision model"
        ),
        "claude-3-sonnet": LLMModel(
            provider=LLMProvider.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            display_name="Claude 3 Sonnet",
            input_cost_per_1k_tokens=0.003,
            output_cost_per_1k_tokens=0.015,
            max_tokens=4096,
            supports_vision=True,
            quality_score=0.92,
            speed_score=0.8,
            description="Balanced performance and cost for complex OCR"
        ),
        
        # Google Models
        "gemini-1.5-flash": LLMModel(
            provider=LLMProvider.GOOGLE,
            model_name="gemini-1.5-flash",
            display_name="Gemini 1.5 Flash",
            input_cost_per_1k_tokens=0.000075,
            output_cost_per_1k_tokens=0.0003,
            max_tokens=8192,
            supports_vision=True,
            quality_score=0.83,
            speed_score=0.92,
            description="Extremely cost-effective with good quality"
        ),
        "gemini-1.5-pro": LLMModel(
            provider=LLMProvider.GOOGLE,
            model_name="gemini-1.5-pro",
            display_name="Gemini 1.5 Pro",
            input_cost_per_1k_tokens=0.00125,
            output_cost_per_1k_tokens=0.00375,
            max_tokens=8192,
            supports_vision=True,
            quality_score=0.9,
            speed_score=0.75,
            description="High-quality multimodal processing"
        ),
        
        # DeepSeek Models
        "deepseek-chat": LLMModel(
            provider=LLMProvider.DEEPSEEK,
            model_name="deepseek-chat",
            display_name="DeepSeek Chat",
            input_cost_per_1k_tokens=0.00014,
            output_cost_per_1k_tokens=0.00028,
            max_tokens=4096,
            supports_vision=False,
            quality_score=0.8,
            speed_score=0.85,
            description="Very affordable text processing"
        ),
        
        # NVIDIA Models
        "nvidia/llama-3.1-nemotron-70b": LLMModel(
            provider=LLMProvider.NVIDIA,
            model_name="nvidia/llama-3.1-nemotron-70b-instruct",
            display_name="Nemotron 70B",
            input_cost_per_1k_tokens=0.0004,
            output_cost_per_1k_tokens=0.0004,
            max_tokens=4096,
            supports_vision=False,
            quality_score=0.87,
            speed_score=0.8,
            description="High-quality instruction following"
        )
    }
    
    @classmethod
    def get_model(cls, model_name: str) -> Optional[LLMModel]:
        """Get model configuration by name."""
        return cls.MODELS.get(model_name)
    
    @classmethod
    def get_models_by_provider(cls, provider: LLMProvider) -> List[LLMModel]:
        """Get all models from a specific provider."""
        return [model for model in cls.MODELS.values() if model.provider == provider]
    
    @classmethod
    def get_vision_models(cls) -> List[LLMModel]:
        """Get all models that support vision."""
        return [model for model in cls.MODELS.values() if model.supports_vision]
    
    @classmethod
    def get_cheapest_models(cls, limit: int = 5, vision_only: bool = False) -> List[LLMModel]:
        """Get the cheapest models sorted by input cost."""
        models = cls.get_vision_models() if vision_only else list(cls.MODELS.values())
        return sorted(models, key=lambda m: m.input_cost_per_1k_tokens)[:limit]
    
    @classmethod
    def get_best_value_models(cls, vision_only: bool = False) -> List[LLMModel]:
        """Get models with the best quality/cost ratio."""
        models = cls.get_vision_models() if vision_only else list(cls.MODELS.values())
        
        # Calculate value score (quality / average_cost)
        def value_score(model: LLMModel) -> float:
            avg_cost = (model.input_cost_per_1k_tokens + model.output_cost_per_1k_tokens) / 2
            return model.quality_score / max(avg_cost, 0.00001)  # Avoid division by zero
        
        return sorted(models, key=value_score, reverse=True)


class CostCalculator:
    """Calculate costs for LLM-based OCR operations."""
    
    @staticmethod
    def estimate_tokens_for_image(width: int, height: int, has_text: bool = True) -> int:
        """Estimate tokens needed for image processing."""
        # Base tokens for image processing
        base_tokens = 85  # OpenAI's base cost for images
        
        # Additional tokens based on image complexity
        pixel_factor = (width * height) / (1024 * 1024)  # Normalize to 1MP
        complexity_tokens = int(pixel_factor * 50)
        
        # Text density factor
        text_factor = 1.5 if has_text else 1.0
        
        return int((base_tokens + complexity_tokens) * text_factor)
    
    @staticmethod
    def estimate_output_tokens(task_type: OCRTaskType, input_length: int) -> int:
        """Estimate output tokens based on task type and input."""
        multipliers = {
            OCRTaskType.BASIC_TEXT_EXTRACTION: 0.8,
            OCRTaskType.STRUCTURED_DATA_EXTRACTION: 1.2,
            OCRTaskType.HANDWRITING_RECOGNITION: 0.9,
            OCRTaskType.TABLE_EXTRACTION: 1.5,
            OCRTaskType.DOCUMENT_ANALYSIS: 2.0
        }
        
        base_output = max(50, int(input_length * multipliers.get(task_type, 1.0)))
        return min(base_output, 2000)  # Cap at reasonable maximum
    
    @staticmethod
    def calculate_cost(model: LLMModel, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for a request."""
        input_cost = (input_tokens / 1000) * model.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000) * model.output_cost_per_1k_tokens
        return input_cost + output_cost
    
    @staticmethod
    def estimate_document_cost(
        model_name: str,
        task_type: OCRTaskType,
        image_width: int = 1024,
        image_height: int = 1024,
        has_complex_text: bool = True
    ) -> Dict[str, Union[float, int]]:
        """Estimate cost for processing a document."""
        model = LLMModelRegistry.get_model(model_name)
        if not model:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Estimate tokens
        input_tokens = CostCalculator.estimate_tokens_for_image(
            image_width, image_height, has_complex_text
        )
        output_tokens = CostCalculator.estimate_output_tokens(task_type, input_tokens)
        
        # Calculate cost
        total_cost = CostCalculator.calculate_cost(model, input_tokens, output_tokens)
        
        return {
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": round(total_cost, 6),
            "cost_breakdown": {
                "input_cost": round((input_tokens / 1000) * model.input_cost_per_1k_tokens, 6),
                "output_cost": round((output_tokens / 1000) * model.output_cost_per_1k_tokens, 6)
            }
        }


class LLMCostTracker:
    """Track LLM usage costs."""
    
    def __init__(self):
        self.usage_history: List[Dict] = []
    
    def record_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        task_type: str,
        timestamp: Optional[datetime] = None
    ):
        """Record LLM usage for cost tracking."""
        self.usage_history.append({
            "timestamp": timestamp or datetime.now(),
            "model_name": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": cost,
            "task_type": task_type
        })
    
    def get_daily_costs(self, date: datetime = None) -> float:
        """Get total costs for a specific day."""
        target_date = date or datetime.now()
        daily_usage = [
            usage for usage in self.usage_history
            if usage["timestamp"].date() == target_date.date()
        ]
        return sum(usage["total_cost"] for usage in daily_usage)
    
    def get_monthly_costs(self, year: int = None, month: int = None) -> float:
        """Get total costs for a specific month."""
        now = datetime.now()
        target_year = year or now.year
        target_month = month or now.month
        
        monthly_usage = [
            usage for usage in self.usage_history
            if usage["timestamp"].year == target_year and usage["timestamp"].month == target_month
        ]
        return sum(usage["total_cost"] for usage in monthly_usage)
    
    def get_model_usage_stats(self) -> Dict[str, Dict]:
        """Get usage statistics by model."""
        stats = {}
        for usage in self.usage_history:
            model = usage["model_name"]
            if model not in stats:
                stats[model] = {
                    "total_requests": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cost": 0.0
                }
            
            stats[model]["total_requests"] += 1
            stats[model]["total_input_tokens"] += usage["input_tokens"]
            stats[model]["total_output_tokens"] += usage["output_tokens"]
            stats[model]["total_cost"] += usage["total_cost"]
        
        return stats


# Global configuration instance
llm_config = LLMOCRConfig()
cost_tracker = LLMCostTracker()