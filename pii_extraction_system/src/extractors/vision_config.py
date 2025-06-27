"""
Configuration System for Vision-LLM PII Extraction

This module provides comprehensive configuration management for all
vision-based PII extraction components and model settings.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from .prompt_router import RoutingStrategy

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Supported configuration formats"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"


@dataclass
class ModelConfig:
    """Configuration for individual models"""
    name: str
    provider: str
    enabled: bool = True
    priority: int = 1
    max_tokens: int = 4000
    temperature: float = 0.1
    timeout: float = 120.0
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: List[str] = field(default_factory=lambda: ["vision", "text"])
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationConfig:
    """Configuration for document classification"""
    enabled: bool = True
    preferred_models: List[str] = field(default_factory=lambda: ["gpt-4o-mini", "claude-3-5-haiku-20241022"])
    confidence_threshold: float = 0.7
    max_retries: int = 2
    cache_classifications: bool = True
    cache_ttl: int = 3600  # seconds


@dataclass
class RoutingConfig:
    """Configuration for prompt routing"""
    enabled: bool = True
    default_strategy: str = "balanced"
    enable_learning: bool = True
    max_history: int = 1000
    model_preferences: Dict[str, List[str]] = field(default_factory=dict)
    domain_preferences: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class LocalModelConfig:
    """Configuration for local models"""
    enabled: bool = False
    models_dir: Optional[str] = None
    enable_gpu: bool = True
    max_memory_usage: int = 8192  # MB
    enable_auto_download: bool = False
    preferred_frameworks: List[str] = field(default_factory=lambda: ["ollama", "transformers"])
    fallback_to_cloud: bool = True


@dataclass
class QualityConfig:
    """Configuration for quality assessment"""
    enabled: bool = True
    min_confidence_threshold: float = 0.7
    enable_cross_validation: bool = True
    enable_format_validation: bool = True
    enable_context_analysis: bool = True
    auto_flag_threshold: float = 0.5
    require_human_review_threshold: float = 0.6


@dataclass
class SecurityConfig:
    """Configuration for security and privacy"""
    enable_role_based_filtering: bool = True
    default_user_role: str = "general_user"
    enable_redaction: bool = True
    default_redaction_level: str = "partial"
    enable_audit_logging: bool = True
    mask_sensitive_logs: bool = True
    compliance_requirements: List[str] = field(default_factory=lambda: ["GDPR", "CCPA"])


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    max_concurrent_requests: int = 5
    request_timeout: float = 300.0
    enable_caching: bool = True
    cache_ttl: int = 1800  # seconds
    enable_compression: bool = True
    max_image_size: int = 10485760  # bytes (10MB)
    enable_async_processing: bool = False


@dataclass
class IntegrationConfig:
    """Configuration for pipeline integration"""
    enable_vision_extraction: bool = True
    enable_fallback: bool = True
    fallback_extractors: List[str] = field(default_factory=lambda: ["rule_based", "ner"])
    combine_results: bool = True
    prefer_vision_results: bool = True
    confidence_weighting: Dict[str, float] = field(default_factory=lambda: {
        "vision": 0.8,
        "traditional": 0.6
    })


@dataclass
class VisionExtractionConfig:
    """Complete configuration for vision-based PII extraction"""
    models: List[ModelConfig] = field(default_factory=list)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    local_models: LocalModelConfig = field(default_factory=LocalModelConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    def __post_init__(self):
        """Initialize default models if none provided"""
        if not self.models:
            self.models = self._get_default_models()
    
    def _get_default_models(self) -> List[ModelConfig]:
        """Get default model configurations"""
        return [
            # OpenAI Models
            ModelConfig(
                name="gpt-4o",
                provider="openai",
                priority=1,
                cost_per_1k_input=2.5,
                cost_per_1k_output=10.0,
                capabilities=["vision", "text", "high_accuracy"]
            ),
            ModelConfig(
                name="gpt-4o-mini",
                provider="openai",
                priority=2,
                cost_per_1k_input=0.15,
                cost_per_1k_output=0.6,
                capabilities=["vision", "text", "fast", "cost_effective"]
            ),
            
            # Anthropic Models
            ModelConfig(
                name="claude-3-5-sonnet-20241022",
                provider="anthropic",
                priority=1,
                cost_per_1k_input=3.0,
                cost_per_1k_output=15.0,
                capabilities=["vision", "text", "high_accuracy", "reasoning"]
            ),
            ModelConfig(
                name="claude-3-5-haiku-20241022",
                provider="anthropic",
                priority=2,
                cost_per_1k_input=1.0,
                cost_per_1k_output=5.0,
                capabilities=["vision", "text", "fast", "balanced"]
            ),
            
            # Google Models
            ModelConfig(
                name="gemini-1.5-pro",
                provider="google",
                priority=1,
                cost_per_1k_input=2.5,
                cost_per_1k_output=7.5,
                capabilities=["vision", "text", "large_context"]
            ),
            ModelConfig(
                name="gemini-1.5-flash",
                provider="google",
                priority=2,
                cost_per_1k_input=0.075,
                cost_per_1k_output=0.3,
                capabilities=["vision", "text", "very_fast", "cost_effective"]
            )
        ]


class VisionConfigManager:
    """
    Configuration manager for Vision-LLM PII extraction system
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 config_format: ConfigFormat = ConfigFormat.JSON,
                 auto_save: bool = True):
        """
        Initialize configuration manager
        
        Args:
            config_file: Path to configuration file
            config_format: Configuration file format
            auto_save: Automatically save changes
        """
        self.config_file = config_file
        self.config_format = config_format
        self.auto_save = auto_save
        
        # Default configuration
        self.config = VisionExtractionConfig()
        
        # Load configuration if file provided
        if config_file:
            self.load_config(config_file, config_format)
        else:
            # Load from environment variables
            self._load_from_environment()
        
        logger.info(f"VisionConfigManager initialized with {len(self.config.models)} models")
    
    def load_config(self, 
                   config_file: str,
                   config_format: Optional[ConfigFormat] = None) -> bool:
        """
        Load configuration from file
        
        Args:
            config_file: Path to configuration file
            config_format: Configuration format (auto-detected if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            config_path = Path(config_file)
            
            if not config_path.exists():
                logger.warning(f"Configuration file not found: {config_file}")
                return False
            
            # Auto-detect format if not specified
            if config_format is None:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_format = ConfigFormat.YAML
                elif config_path.suffix.lower() == '.json':
                    config_format = ConfigFormat.JSON
                else:
                    config_format = ConfigFormat.JSON
            
            # Load configuration
            with open(config_path, 'r') as f:
                if config_format == ConfigFormat.JSON:
                    config_data = json.load(f)
                elif config_format == ConfigFormat.YAML:
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_format}")
            
            # Parse configuration
            self.config = self._parse_config_data(config_data)
            self.config_file = config_file
            self.config_format = config_format
            
            logger.info(f"Configuration loaded from {config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            return False
    
    def save_config(self, 
                   config_file: Optional[str] = None,
                   config_format: Optional[ConfigFormat] = None) -> bool:
        """
        Save configuration to file
        
        Args:
            config_file: Path to save configuration (uses current file if None)
            config_format: Configuration format (uses current format if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use current file and format if not specified
            file_path = config_file or self.config_file
            format_type = config_format or self.config_format
            
            if not file_path:
                raise ValueError("No configuration file specified")
            
            # Convert config to dictionary
            config_data = asdict(self.config)
            
            # Create directory if needed
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save configuration
            with open(file_path, 'w') as f:
                if format_type == ConfigFormat.JSON:
                    json.dump(config_data, f, indent=2, default=str)
                elif format_type == ConfigFormat.YAML:
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported config format: {format_type}")
            
            logger.info(f"Configuration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        
        # Model configuration from environment
        if os.getenv('VISION_MODELS_CONFIG'):
            try:
                models_config = json.loads(os.getenv('VISION_MODELS_CONFIG'))
                self.config.models = [ModelConfig(**model) for model in models_config]
            except Exception as e:
                logger.warning(f"Failed to parse VISION_MODELS_CONFIG: {e}")
        
        # Individual settings
        env_mappings = {
            'VISION_CLASSIFICATION_ENABLED': ('classification', 'enabled', bool),
            'VISION_ROUTING_ENABLED': ('routing', 'enabled', bool),
            'VISION_ROUTING_STRATEGY': ('routing', 'default_strategy', str),
            'VISION_LOCAL_MODELS_ENABLED': ('local_models', 'enabled', bool),
            'VISION_LOCAL_MODELS_DIR': ('local_models', 'models_dir', str),
            'VISION_QUALITY_ENABLED': ('quality', 'enabled', bool),
            'VISION_CONFIDENCE_THRESHOLD': ('quality', 'min_confidence_threshold', float),
            'VISION_SECURITY_ROLE_FILTERING': ('security', 'enable_role_based_filtering', bool),
            'VISION_DEFAULT_USER_ROLE': ('security', 'default_user_role', str),
            'VISION_PERFORMANCE_MAX_CONCURRENT': ('performance', 'max_concurrent_requests', int),
            'VISION_INTEGRATION_FALLBACK': ('integration', 'enable_fallback', bool)
        }
        
        for env_var, (section, key, value_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert value to appropriate type
                    if value_type == bool:
                        parsed_value = value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == int:
                        parsed_value = int(value)
                    elif value_type == float:
                        parsed_value = float(value)
                    else:
                        parsed_value = value
                    
                    # Set configuration value
                    section_obj = getattr(self.config, section)
                    setattr(section_obj, key, parsed_value)
                    
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to parse environment variable {env_var}: {e}")
        
        logger.info("Configuration loaded from environment variables")
    
    def _parse_config_data(self, config_data: Dict[str, Any]) -> VisionExtractionConfig:
        """Parse configuration data into config objects"""
        
        # Parse models
        models = []
        if 'models' in config_data:
            for model_data in config_data['models']:
                models.append(ModelConfig(**model_data))
        
        # Parse other sections
        classification = ClassificationConfig(**config_data.get('classification', {}))
        routing = RoutingConfig(**config_data.get('routing', {}))
        local_models = LocalModelConfig(**config_data.get('local_models', {}))
        quality = QualityConfig(**config_data.get('quality', {}))
        security = SecurityConfig(**config_data.get('security', {}))
        performance = PerformanceConfig(**config_data.get('performance', {}))
        integration = IntegrationConfig(**config_data.get('integration', {}))
        
        return VisionExtractionConfig(
            models=models,
            classification=classification,
            routing=routing,
            local_models=local_models,
            quality=quality,
            security=security,
            performance=performance,
            integration=integration
        )
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for specific model"""
        for model in self.config.models:
            if model.name == model_name:
                return model
        return None
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models"""
        return [model for model in self.config.models if model.enabled]
    
    def get_models_by_provider(self, provider: str) -> List[ModelConfig]:
        """Get models by provider"""
        return [model for model in self.config.models if model.provider == provider]
    
    def get_models_by_capability(self, capability: str) -> List[ModelConfig]:
        """Get models by capability"""
        return [model for model in self.config.models if capability in model.capabilities]
    
    def update_model_config(self, model_name: str, updates: Dict[str, Any]) -> bool:
        """Update configuration for specific model"""
        for model in self.config.models:
            if model.name == model_name:
                for key, value in updates.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                
                if self.auto_save and self.config_file:
                    self.save_config()
                
                return True
        return False
    
    def add_model_config(self, model_config: ModelConfig) -> bool:
        """Add new model configuration"""
        # Check if model already exists
        if any(model.name == model_config.name for model in self.config.models):
            logger.warning(f"Model {model_config.name} already exists")
            return False
        
        self.config.models.append(model_config)
        
        if self.auto_save and self.config_file:
            self.save_config()
        
        return True
    
    def remove_model_config(self, model_name: str) -> bool:
        """Remove model configuration"""
        original_count = len(self.config.models)
        self.config.models = [model for model in self.config.models if model.name != model_name]
        
        if len(self.config.models) < original_count:
            if self.auto_save and self.config_file:
                self.save_config()
            return True
        
        return False
    
    def update_section_config(self, section: str, updates: Dict[str, Any]) -> bool:
        """Update configuration for specific section"""
        try:
            if hasattr(self.config, section):
                section_obj = getattr(self.config, section)
                
                for key, value in updates.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                    else:
                        logger.warning(f"Unknown configuration key: {section}.{key}")
                
                if self.auto_save and self.config_file:
                    self.save_config()
                
                return True
            else:
                logger.error(f"Unknown configuration section: {section}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to update section {section}: {e}")
            return False
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues"""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Validate models
        if not self.config.models:
            issues['warnings'].append("No models configured")
        
        enabled_models = self.get_enabled_models()
        if not enabled_models:
            issues['errors'].append("No enabled models found")
        
        # Validate routing strategy
        try:
            RoutingStrategy(self.config.routing.default_strategy)
        except ValueError:
            issues['errors'].append(f"Invalid routing strategy: {self.config.routing.default_strategy}")
        
        # Validate confidence thresholds
        if not 0.0 <= self.config.quality.min_confidence_threshold <= 1.0:
            issues['errors'].append("Quality confidence threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.config.classification.confidence_threshold <= 1.0:
            issues['errors'].append("Classification confidence threshold must be between 0.0 and 1.0")
        
        # Validate performance settings
        if self.config.performance.max_concurrent_requests <= 0:
            issues['errors'].append("Max concurrent requests must be positive")
        
        if self.config.performance.request_timeout <= 0:
            issues['errors'].append("Request timeout must be positive")
        
        # Validate local models settings
        if self.config.local_models.enabled and not self.config.local_models.models_dir:
            issues['warnings'].append("Local models enabled but no models directory specified")
        
        return issues
    
    def export_config_template(self, file_path: str) -> bool:
        """Export configuration template with all available options"""
        try:
            template_config = VisionExtractionConfig()
            config_data = asdict(template_config)
            
            # Add documentation
            config_data['_documentation'] = {
                'models': 'List of available vision models with their configurations',
                'classification': 'Document classification settings',
                'routing': 'Intelligent model routing configuration',
                'local_models': 'Local model management settings',
                'quality': 'Quality assessment and confidence scoring',
                'security': 'Security, privacy, and role-based filtering',
                'performance': 'Performance optimization settings',
                'integration': 'Pipeline integration configuration'
            }
            
            # Save template
            with open(file_path, 'w') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2, default=str)
            
            logger.info(f"Configuration template exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration template: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        enabled_models = self.get_enabled_models()
        
        return {
            'total_models': len(self.config.models),
            'enabled_models': len(enabled_models),
            'providers': list(set(model.provider for model in enabled_models)),
            'capabilities': list(set(cap for model in enabled_models for cap in model.capabilities)),
            'features_enabled': {
                'classification': self.config.classification.enabled,
                'routing': self.config.routing.enabled,
                'local_models': self.config.local_models.enabled,
                'quality_assessment': self.config.quality.enabled,
                'role_based_filtering': self.config.security.enable_role_based_filtering,
                'fallback': self.config.integration.enable_fallback
            },
            'performance_settings': {
                'max_concurrent': self.config.performance.max_concurrent_requests,
                'timeout': self.config.performance.request_timeout,
                'caching': self.config.performance.enable_caching
            }
        }