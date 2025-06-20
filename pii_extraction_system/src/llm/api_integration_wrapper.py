"""
Comprehensive Multi-LLM API Integration Wrapper

This module provides a unified interface for integrating multiple LLM APIs
with conditional API key handling, cost tracking, and error management.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import json

from .multimodal_llm_service import MultimodalLLMService, llm_service
from .api_key_manager import APIKeyManager, api_key_manager
from .cost_tracker import CostTracker, default_cost_tracker, TokenUsageMonitor, token_monitor

logger = logging.getLogger(__name__)


@dataclass
class LLMModelInfo:
    """Detailed information about an LLM model"""
    provider: str
    model_name: str
    display_name: str
    supports_vision: bool
    supports_json: bool
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    is_available: bool
    status_message: str
    recommended_for: List[str]
    max_context_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'provider': self.provider,
            'model_name': self.model_name,
            'display_name': self.display_name,
            'supports_vision': self.supports_vision,
            'supports_json': self.supports_json,
            'cost_per_1k_input_tokens': self.cost_per_1k_input_tokens,
            'cost_per_1k_output_tokens': self.cost_per_1k_output_tokens,
            'is_available': self.is_available,
            'status_message': self.status_message,
            'recommended_for': self.recommended_for,
            'max_context_length': self.max_context_length
        }


class MultiLLMIntegrationWrapper:
    """Main wrapper for multi-LLM API integration with full feature support"""
    
    # Model metadata and recommendations
    MODEL_METADATA = {
        'openai/gpt-4o': {
            'display_name': 'GPT-4o (Omni)',
            'recommended_for': ['complex_documents', 'high_accuracy', 'multimodal'],
            'max_context_length': 128000
        },
        'openai/gpt-4o-mini': {
            'display_name': 'GPT-4o Mini',
            'recommended_for': ['cost_effective', 'simple_documents', 'high_speed'],
            'max_context_length': 128000
        },
        'openai/gpt-4-turbo': {
            'display_name': 'GPT-4 Turbo',
            'recommended_for': ['complex_reasoning', 'detailed_analysis'],
            'max_context_length': 128000
        },
        'openai/gpt-4': {
            'display_name': 'GPT-4',
            'recommended_for': ['legacy_support', 'proven_reliability'],
            'max_context_length': 8192
        },
        'anthropic/claude-3-5-sonnet-20241022': {
            'display_name': 'Claude 3.5 Sonnet',
            'recommended_for': ['best_overall', 'complex_documents', 'high_accuracy'],
            'max_context_length': 200000
        },
        'anthropic/claude-3-5-haiku-20241022': {
            'display_name': 'Claude 3.5 Haiku',
            'recommended_for': ['cost_effective', 'fast_processing', 'simple_documents'],
            'max_context_length': 200000
        },
        'anthropic/claude-3-opus-20240229': {
            'display_name': 'Claude 3 Opus',
            'recommended_for': ['premium_quality', 'complex_reasoning'],
            'max_context_length': 200000
        },
        'google/gemini-1.5-pro': {
            'display_name': 'Gemini 1.5 Pro',
            'recommended_for': ['long_context', 'multimodal', 'cost_effective'],
            'max_context_length': 1048576
        },
        'google/gemini-1.5-flash': {
            'display_name': 'Gemini 1.5 Flash',
            'recommended_for': ['ultra_fast', 'cost_effective', 'simple_tasks'],
            'max_context_length': 1048576
        },
        'mistral/mistral-large': {
            'display_name': 'Mistral Large',
            'recommended_for': ['text_only', 'european_languages', 'cost_effective'],
            'max_context_length': 32000
        },
        'mistral/mistral-medium': {
            'display_name': 'Mistral Medium',
            'recommended_for': ['text_only', 'balanced_performance'],
            'max_context_length': 32000
        },
        'mistral/mistral-small': {
            'display_name': 'Mistral Small',
            'recommended_for': ['text_only', 'cost_effective', 'fast'],
            'max_context_length': 32000
        },
        'mistral/mistral-tiny': {
            'display_name': 'Mistral Tiny',
            'recommended_for': ['text_only', 'ultra_cost_effective'],
            'max_context_length': 32000
        }
    }
    
    def __init__(
        self,
        env_file_path: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_cost_tracking: bool = True,
        enable_monitoring: bool = True,
        budget_config: Optional[Any] = None
    ):
        """
        Initialize the multi-LLM integration wrapper
        
        Args:
            env_file_path: Path to .env file for API keys
            session_id: Unique session identifier for cost tracking
            enable_cost_tracking: Enable cost tracking functionality
            enable_monitoring: Enable real-time monitoring
            budget_config: Budget configuration object
        """
        # Initialize components
        self.api_key_manager = api_key_manager if not env_file_path else APIKeyManager(env_file_path)
        self.cost_tracker = default_cost_tracker if enable_cost_tracking else None
        self.token_monitor = token_monitor if enable_monitoring else None
        
        # Budget configuration
        if budget_config is None:
            try:
                from ..core.config import settings
                self.budget_config = settings.budget
            except ImportError:
                self.budget_config = None
                logger.warning("Budget configuration not available")
        else:
            self.budget_config = budget_config
        
        # Initialize LLM service with budget support
        self.llm_service = MultimodalLLMService(
            cost_tracker=self.cost_tracker,
            budget_config=self.budget_config
        )
        
        # Session management
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if self.cost_tracker:
            self.cost_tracker.session_id = self.session_id
        
        # Initialize model availability
        self._available_models: Dict[str, LLMModelInfo] = {}
        self._refresh_available_models()
        
        logger.info(f"Multi-LLM Integration initialized with {len(self._available_models)} available models")
        logger.info(f"Budget enforcement: {'enabled' if self.budget_config else 'disabled'}")
    
    def _refresh_available_models(self):
        """Refresh the list of available models based on API keys"""
        self._available_models = {}
        
        # Get available models from LLM service
        available_model_keys = self.llm_service.get_available_models()
        
        for model_key in available_model_keys:
            provider, model_name = model_key.split('/', 1)
            
            # Check if API key is available
            if not self.api_key_manager.is_available(provider):
                continue
            
            # Get model info from LLM service
            model_info = self.llm_service.get_model_info(model_key)
            
            # Get metadata
            metadata = self.MODEL_METADATA.get(model_key, {})
            
            # Create model info object
            llm_model_info = LLMModelInfo(
                provider=provider,
                model_name=model_name,
                display_name=metadata.get('display_name', f"{provider.title()} {model_name}"),
                supports_vision=model_info.get('supports_images', True),
                supports_json=model_info.get('supports_json', True),
                cost_per_1k_input_tokens=model_info.get('cost_per_1k_input_tokens', 0),
                cost_per_1k_output_tokens=model_info.get('cost_per_1k_output_tokens', 0),
                is_available=model_info.get('available', False),
                status_message="Available" if model_info.get('available') else "Not available",
                recommended_for=metadata.get('recommended_for', []),
                max_context_length=metadata.get('max_context_length', 4096)
            )
            
            self._available_models[model_key] = llm_model_info
    
    def get_available_models(self) -> Dict[str, LLMModelInfo]:
        """Get all available models with their detailed information"""
        return self._available_models.copy()
    
    def get_models_by_capability(self, capability: str) -> List[LLMModelInfo]:
        """
        Get models that have a specific capability
        
        Args:
            capability: Capability to filter by (e.g., 'complex_documents', 'cost_effective')
        
        Returns:
            List of models with the specified capability
        """
        return [
            model for model in self._available_models.values()
            if capability in model.recommended_for
        ]
    
    def get_best_model_for_task(self, task_type: str) -> Optional[str]:
        """
        Get the best available model for a specific task type
        
        Args:
            task_type: Type of task (e.g., 'complex_documents', 'cost_effective')
        
        Returns:
            Model key of the best available model, or None if no suitable model
        """
        suitable_models = self.get_models_by_capability(task_type)
        
        if not suitable_models:
            # Fall back to any available model
            if self._available_models:
                return list(self._available_models.keys())[0]
            return None
        
        # Prioritize based on recommendation order and cost
        if task_type == 'best_overall':
            # Claude 3.5 Sonnet is marked as best overall
            for model_key, model_info in self._available_models.items():
                if 'best_overall' in model_info.recommended_for:
                    return model_key
        
        # Return the first suitable model
        return f"{suitable_models[0].provider}/{suitable_models[0].model_name}"
    
    def extract_pii_with_tracking(
        self,
        image_data: str,
        model_key: str,
        document_type: str = "document",
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        allow_auto_fallback: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract PII from image with full tracking and error handling
        
        Args:
            image_data: Base64 encoded image
            model_key: Model identifier (e.g., "openai/gpt-4o")
            document_type: Type of document
            user_id: User identifier for tracking
            document_id: Document identifier for tracking
            allow_auto_fallback: Allow automatic fallback to cheaper models
            **kwargs: Additional parameters for the LLM
        
        Returns:
            Extraction results with tracking information
        """
        original_model = model_key
        
        # Validate model availability
        if model_key not in self._available_models:
            return {
                "success": False,
                "error": f"Model {model_key} is not available. Available models: {list(self._available_models.keys())}",
                "available_models": list(self._available_models.keys())
            }
        
        # Extract provider and model
        provider, model_name = model_key.split('/', 1)
        
        # Perform extraction
        try:
            result = self.llm_service.extract_pii_from_image(
                image_data=image_data,
                model_key=model_key,
                document_type=document_type,
                **kwargs
            )
            
            # Handle budget constraint with automatic fallback
            if not result.get("success") and "Budget limit exceeded" in result.get("error", ""):
                if allow_auto_fallback:
                    alternative_model = self.suggest_alternative_model(
                        preferred_model=model_key,
                        task_requirements={'requires_vision': True, 'requires_json': True}
                    )
                    
                    if alternative_model:
                        logger.info(f"Falling back to cheaper model {alternative_model} due to budget constraints")
                        result = self.llm_service.extract_pii_from_image(
                            image_data=image_data,
                            model_key=alternative_model,
                            document_type=document_type,
                            **kwargs
                        )
                        result["fallback_used"] = True
                        result["original_model"] = original_model
                        result["fallback_model"] = alternative_model
                        model_key = alternative_model  # Update for tracking
                        provider, model_name = model_key.split('/', 1)
                    else:
                        # Add budget status to error response
                        result["budget_status"] = self.get_budget_status()
                        return result
                else:
                    # Add budget status to error response
                    result["budget_status"] = self.get_budget_status()
                    return result
            
            # Track usage if cost tracking is enabled
            if self.cost_tracker and result.get("success") and "usage" in result:
                usage = result["usage"]
                self.cost_tracker.record_usage(
                    provider=provider,
                    model=model_name,
                    task_type="pii_extraction",
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    estimated_cost=usage.get("estimated_cost", 0),
                    processing_time=result.get("processing_time"),
                    success=True,
                    user_id=user_id,
                    document_id=document_id
                )
                
                # Check usage limits if monitoring is enabled
                if self.token_monitor:
                    limit_check = self.token_monitor.check_limits(provider)
                    if limit_check.get("alerts"):
                        result["usage_alerts"] = limit_check["alerts"]
            
            # Add model info to result
            result["model_info"] = self._available_models[model_key].to_dict()
            result["budget_status_summary"] = self.get_budget_status(provider)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in PII extraction with {model_key}: {str(e)}")
            
            # Track failed usage if cost tracking is enabled
            if self.cost_tracker:
                self.cost_tracker.record_usage(
                    provider=provider,
                    model=model_name,
                    task_type="pii_extraction",
                    input_tokens=0,
                    output_tokens=0,
                    estimated_cost=0,
                    success=False,
                    error_message=str(e),
                    user_id=user_id,
                    document_id=document_id
                )
            
            return {
                "success": False,
                "error": str(e),
                "model": model_key,
                "provider": provider,
                "budget_status": self.get_budget_status()
            }
    
    def batch_extract_with_fallback(
        self,
        image_data_list: List[str],
        primary_model: str,
        fallback_models: List[str],
        document_types: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch extract with automatic fallback to other models on failure
        
        Args:
            image_data_list: List of base64 encoded images
            primary_model: Primary model to use
            fallback_models: List of fallback models
            document_types: List of document types
            progress_callback: Progress callback function
        
        Returns:
            List of extraction results
        """
        results = []
        total_cost = 0.0
        
        for i, image_data in enumerate(image_data_list):
            doc_type = document_types[i] if document_types and i < len(document_types) else "document"
            
            # Try primary model first
            result = self.extract_pii_with_tracking(image_data, primary_model, doc_type)
            
            # If primary model fails, try fallbacks
            if not result.get("success"):
                for fallback_model in fallback_models:
                    if fallback_model in self._available_models:
                        logger.info(f"Falling back to {fallback_model} after {primary_model} failed")
                        result = self.extract_pii_with_tracking(image_data, fallback_model, doc_type)
                        if result.get("success"):
                            result["fallback_used"] = True
                            result["original_model"] = primary_model
                            result["fallback_model"] = fallback_model
                            break
            
            results.append(result)
            
            if result.get("success") and "usage" in result:
                total_cost += result["usage"].get("estimated_cost", 0)
            
            if progress_callback:
                progress_callback(i + 1, len(image_data_list), total_cost)
        
        return results
    
    def get_cost_summary(self, time_period: str = "session") -> Dict[str, Any]:
        """
        Get cost summary for specified time period
        
        Args:
            time_period: 'session', 'daily', 'monthly', or 'all'
        
        Returns:
            Cost summary with breakdown by provider and model
        """
        if not self.cost_tracker:
            return {"error": "Cost tracking is not enabled"}
        
        if time_period == "session":
            return self.cost_tracker.get_session_stats(self.session_id)
        elif time_period == "daily":
            return self.cost_tracker.get_daily_costs()
        elif time_period == "monthly":
            return self.cost_tracker.get_monthly_costs()
        elif time_period == "all":
            return self.cost_tracker.get_cost_analysis(days=30)
        else:
            return {"error": f"Invalid time period: {time_period}"}
    
    def set_usage_limits(self, provider: str, daily_limit: float, monthly_limit: float):
        """
        Set usage limits for a provider
        
        Args:
            provider: Provider name
            daily_limit: Daily cost limit in USD
            monthly_limit: Monthly cost limit in USD
        """
        if self.token_monitor:
            self.token_monitor.set_daily_limit(provider, daily_limit)
            self.token_monitor.set_monthly_limit(provider, monthly_limit)
            logger.info(f"Set usage limits for {provider}: Daily=${daily_limit}, Monthly=${monthly_limit}")
    
    def get_api_key_status(self) -> Dict[str, Any]:
        """Get comprehensive API key status report"""
        return self.api_key_manager.get_usage_summary()
    
    def validate_all_api_keys(self, online: bool = False) -> Dict[str, bool]:
        """
        Validate all configured API keys
        
        Args:
            online: Whether to perform online validation
        
        Returns:
            Dictionary of provider -> validation status
        """
        results = {}
        
        for provider in self.api_key_manager.get_available_providers():
            if online:
                results[provider] = self.api_key_manager.validate_api_key_online(provider)
            else:
                results[provider] = self.api_key_manager.is_available(provider)
        
        return results
    
    def get_model_recommendations(self, requirements: Dict[str, Any]) -> List[str]:
        """
        Get model recommendations based on requirements
        
        Args:
            requirements: Dictionary of requirements (e.g., {'cost': 'low', 'accuracy': 'high'})
        
        Returns:
            List of recommended model keys
        """
        recommendations = []
        
        # Map requirements to capabilities
        if requirements.get('cost') == 'low':
            recommendations.extend(self.get_models_by_capability('cost_effective'))
        
        if requirements.get('accuracy') == 'high':
            recommendations.extend(self.get_models_by_capability('high_accuracy'))
        
        if requirements.get('speed') == 'fast':
            recommendations.extend(self.get_models_by_capability('high_speed'))
        
        if requirements.get('complex_documents'):
            recommendations.extend(self.get_models_by_capability('complex_documents'))
        
        # Remove duplicates and return model keys
        seen = set()
        unique_recommendations = []
        for model in recommendations:
            model_key = f"{model.provider}/{model.model_name}"
            if model_key not in seen:
                seen.add(model_key)
                unique_recommendations.append(model_key)
        
        return unique_recommendations
    
    def get_budget_status(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive budget status for all or specific provider
        
        Args:
            provider: Specific provider to check, or None for all providers
        
        Returns:
            Budget status information
        """
        if not self.cost_tracker or not self.budget_config:
            return {
                "error": "Budget tracking not configured",
                "budget_enforcement": False
            }
        
        providers_to_check = [provider] if provider else ['openai', 'anthropic', 'google', 'mistral']
        status = {
            "timestamp": datetime.now().isoformat(),
            "budget_enforcement": getattr(self.budget_config, 'strict_budget_enforcement', True),
            "providers": {}
        }
        
        for prov in providers_to_check:
            budget_info = self.cost_tracker.get_remaining_budget(prov, self.budget_config)
            emergency_stop = self.cost_tracker.check_emergency_stop(prov, self.budget_config)
            
            # Calculate usage percentages
            daily_percentage = (budget_info['daily_usage'] / budget_info['daily_limit']) * 100 if budget_info['daily_limit'] > 0 else 0
            monthly_percentage = (budget_info['monthly_usage'] / budget_info['monthly_limit']) * 100 if budget_info['monthly_limit'] > 0 else 0
            
            # Determine status
            if emergency_stop:
                prov_status = "emergency_stop"
            elif budget_info['remaining_daily'] <= 0 or budget_info['remaining_monthly'] <= 0:
                prov_status = "budget_exceeded"
            elif daily_percentage >= 80 or monthly_percentage >= 80:
                prov_status = "warning"
            else:
                prov_status = "healthy"
            
            status["providers"][prov] = {
                "status": prov_status,
                "daily": {
                    "usage": budget_info['daily_usage'],
                    "limit": budget_info['daily_limit'],
                    "remaining": budget_info['remaining_daily'],
                    "percentage_used": daily_percentage
                },
                "monthly": {
                    "usage": budget_info['monthly_usage'],
                    "limit": budget_info['monthly_limit'],
                    "remaining": budget_info['remaining_monthly'],
                    "percentage_used": monthly_percentage
                },
                "emergency_stop": emergency_stop
            }
        
        return status
    
    def suggest_alternative_model(self, preferred_model: str, task_requirements: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Suggest an alternative model when the preferred model exceeds budget
        
        Args:
            preferred_model: The originally requested model
            task_requirements: Optional task requirements for better suggestions
        
        Returns:
            Alternative model key or None if no suitable alternative
        """
        if not self.budget_config or not getattr(self.budget_config, 'auto_switch_to_cheaper_model', False):
            return None
        
        # Parse preferred model
        try:
            preferred_provider, preferred_model_name = preferred_model.split('/', 1)
        except ValueError:
            return None
        
        # Get all available models sorted by cost
        available_models = []
        for model_key, model_info in self._available_models.items():
            provider, model_name = model_key.split('/', 1)
            
            # Check if this provider is within budget
            budget_status = self.get_budget_status(provider)
            if budget_status.get('providers', {}).get(provider, {}).get('status') in ['healthy', 'warning']:
                available_models.append((
                    model_key,
                    model_info.cost_per_1k_input_tokens + model_info.cost_per_1k_output_tokens,
                    model_info
                ))
        
        # Sort by cost (cheapest first)
        available_models.sort(key=lambda x: x[1])
        
        # Find the cheapest alternative that meets task requirements
        for model_key, cost, model_info in available_models:
            if model_key == preferred_model:
                continue
            
            # Check if this model can handle the task
            if task_requirements:
                if task_requirements.get('requires_vision', True) and not model_info.supports_vision:
                    continue
                if task_requirements.get('requires_json', True) and not model_info.supports_json:
                    continue
            
            logger.info(f"Suggesting cheaper alternative: {model_key} (cost: {cost:.6f}) instead of {preferred_model}")
            return model_key
        
        return None
    
    def export_usage_report(self, output_path: str, format: str = 'json') -> bool:
        """
        Export comprehensive usage report
        
        Args:
            output_path: Path to save the report
            format: Export format ('json' or 'csv')
        
        Returns:
            True if successful
        """
        try:
            if self.cost_tracker:
                self.cost_tracker.export_usage_data(
                    output_path=output_path,
                    format=format,
                    session_id=self.session_id
                )
            
            # Also export API key status
            api_status_path = output_path.replace('.json', '_api_status.txt').replace('.csv', '_api_status.txt')
            api_report = self.api_key_manager.export_status_report(api_status_path)
            
            logger.info(f"Usage report exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export usage report: {e}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        health = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "api_keys": {
                "total": len(self.api_key_manager.SUPPORTED_PROVIDERS),
                "available": len(self.api_key_manager.get_available_providers()),
                "status": "healthy" if len(self.api_key_manager.get_available_providers()) > 0 else "critical"
            },
            "models": {
                "total_available": len(self._available_models),
                "by_provider": {}
            },
            "cost_tracking": {
                "enabled": self.cost_tracker is not None,
                "session_stats": self.cost_tracker.get_real_time_stats() if self.cost_tracker else None
            }
        }
        
        # Count models by provider
        for model_key, model_info in self._available_models.items():
            provider = model_info.provider
            if provider not in health["models"]["by_provider"]:
                health["models"]["by_provider"][provider] = 0
            health["models"]["by_provider"][provider] += 1
        
        # Overall health status
        if health["api_keys"]["available"] == 0:
            health["overall_status"] = "critical"
            health["status_message"] = "No API keys configured"
        elif health["api_keys"]["available"] < 2:
            health["overall_status"] = "warning"
            health["status_message"] = "Limited API providers available"
        else:
            health["overall_status"] = "healthy"
            health["status_message"] = f"{health['api_keys']['available']} providers available"
        
        return health


# Global instance for easy access
try:
    from ..core.config import settings
    budget_config = settings.budget
except ImportError:
    budget_config = None

llm_integration = MultiLLMIntegrationWrapper(budget_config=budget_config)