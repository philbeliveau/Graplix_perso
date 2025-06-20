"""Configuration management for the PII extraction system."""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DataSourceConfig(BaseModel):
    """Data source configuration."""
    
    source_type: str = Field(default="local", description="Data source: 'local' or 's3'")
    local_path: str = Field(default="../data", description="Local data directory")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket name")
    s3_region: str = Field(default="us-west-2", description="S3 region")


class MLModelConfig(BaseModel):
    """ML model configuration."""
    
    enabled_models: List[str] = Field(
        default=["rule_based", "ner", "layout_aware"],
        description="List of enabled extraction models"
    )
    huggingface_token: Optional[str] = Field(default=None, description="HuggingFace API token")
    mlflow_uri: str = Field(default="sqlite:///mlflow.db", description="MLflow tracking URI")
    model_cache_dir: str = Field(default="./data/models", description="Model cache directory")


class ProcessingConfig(BaseModel):
    """Document processing configuration."""
    
    max_file_size_mb: int = Field(default=50, description="Maximum file size in MB")
    concurrent_jobs: int = Field(default=4, description="Number of concurrent processing jobs")
    timeout_seconds: int = Field(default=300, description="Processing timeout in seconds")
    ocr_languages: str = Field(default="fra", description="OCR languages for Tesseract")
    tesseract_cmd: str = Field(default="/opt/homebrew/bin/tesseract", description="Tesseract command path")
    ocr_engine: str = Field(default="tesseract", description="OCR engine: 'tesseract', 'easyocr', or 'both'")
    easyocr_use_gpu: bool = Field(default=False, description="Use GPU for EasyOCR (requires CUDA)")
    
    # LLM OCR settings
    enable_llm_ocr: bool = Field(default=True, description="Enable LLM-based OCR for enhanced accuracy")
    llm_ocr_model: str = Field(default="gpt-4o-mini", description="Default LLM model for OCR")
    llm_ocr_fallback_model: str = Field(default="gpt-3.5-turbo", description="Fallback LLM model for OCR")
    max_llm_cost_per_document: float = Field(default=0.10, description="Maximum cost per document for LLM OCR")
    llm_confidence_threshold: float = Field(default=0.8, description="Minimum confidence to use LLM result")


class PrivacyConfig(BaseModel):
    """Privacy and compliance configuration."""
    
    enable_redaction: bool = Field(default=True, description="Enable PII redaction")
    redaction_character: str = Field(default="*", description="Character for redaction")
    gdpr_compliance: bool = Field(default=True, description="GDPR compliance mode")
    law25_compliance: bool = Field(default=True, description="Quebec Law 25 compliance")
    audit_logging: bool = Field(default=True, description="Enable audit logging")
    data_retention_days: int = Field(default=90, description="Data retention period")


class BudgetConfig(BaseModel):
    """Budget enforcement configuration."""
    
    # Global budget enforcement settings
    strict_budget_enforcement: bool = Field(default=True, description="Prevent API calls when budgets would be exceeded")
    auto_switch_to_cheaper_model: bool = Field(default=False, description="Automatically switch to cheaper models when budget constraints are hit")
    budget_warning_threshold: float = Field(default=0.8, description="Warn when usage reaches this percentage of limit (0.8 = 80%)")
    
    # Daily budget limits per provider (in USD)
    daily_budget_openai: float = Field(default=10.0, description="Daily budget limit for OpenAI API")
    daily_budget_anthropic: float = Field(default=10.0, description="Daily budget limit for Anthropic API")
    daily_budget_google: float = Field(default=10.0, description="Daily budget limit for Google API")
    daily_budget_mistral: float = Field(default=10.0, description="Daily budget limit for Mistral API")
    
    # Monthly budget limits per provider (in USD)
    monthly_budget_openai: float = Field(default=100.0, description="Monthly budget limit for OpenAI API")
    monthly_budget_anthropic: float = Field(default=100.0, description="Monthly budget limit for Anthropic API")
    monthly_budget_google: float = Field(default=100.0, description="Monthly budget limit for Google API")
    monthly_budget_mistral: float = Field(default=100.0, description="Monthly budget limit for Mistral API")
    
    # Emergency settings
    enable_emergency_stop: bool = Field(default=True, description="Enable emergency stop when critical budget exceeded")
    emergency_stop_multiplier: float = Field(default=1.2, description="Emergency stop when usage exceeds budget by this multiplier")
    
    # Model cost estimation parameters
    fallback_token_estimate: int = Field(default=1000, description="Fallback token estimate for pre-flight cost calculation")
    safety_margin_multiplier: float = Field(default=1.1, description="Safety margin for cost estimation (10% buffer)")
    
    def get_daily_limit(self, provider: str) -> float:
        """Get daily budget limit for a provider."""
        provider_map = {
            'openai': self.daily_budget_openai,
            'anthropic': self.daily_budget_anthropic,
            'google': self.daily_budget_google,
            'mistral': self.daily_budget_mistral
        }
        return provider_map.get(provider.lower(), 5.0)  # Default $5 for unknown providers
    
    def get_monthly_limit(self, provider: str) -> float:
        """Get monthly budget limit for a provider."""
        provider_map = {
            'openai': self.monthly_budget_openai,
            'anthropic': self.monthly_budget_anthropic,
            'google': self.monthly_budget_google,
            'mistral': self.monthly_budget_mistral
        }
        return provider_map.get(provider.lower(), 50.0)  # Default $50 for unknown providers


class SecurityConfig(BaseModel):
    """Security configuration."""
    
    secret_key: str = Field(description="Secret key for encryption")
    encryption_key: str = Field(description="Encryption key for data protection")
    enable_auth: bool = Field(default=True, description="Enable authentication")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: str = Field(default="development", description="Environment: dev/staging/prod")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str = Field(default="logs/pii_extraction.log", description="Log file path")
    
    # Database
    database_url: str = Field(
        default="sqlite:///pii_extraction.db",
        description="Database connection URL"
    )
    
    # Dashboard
    streamlit_port: int = Field(default=8501, description="Streamlit server port")
    streamlit_host: str = Field(default="localhost", description="Streamlit server host")
    
    # AWS
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret key")
    aws_region: str = Field(default="us-west-2", description="AWS region")
    
    # Configuration objects
    data_source: DataSourceConfig = Field(default_factory=DataSourceConfig)
    ml_models: MLModelConfig = Field(default_factory=MLModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig(
        secret_key=os.getenv("SECRET_KEY", "dev-secret-key"),
        encryption_key=os.getenv("ENCRYPTION_KEY", "dev-encryption-key")
    ))
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
        # Nested configuration
        env_nested_delimiter = "__"
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> any:
            """Parse environment variables with special handling."""
            if field_name.endswith("_list") and isinstance(raw_val, str):
                return [item.strip() for item in raw_val.split(",")]
            return raw_val


# Global settings instance
settings = Settings()

# Ensure required directories exist
def ensure_directories() -> None:
    """Ensure all required directories exist."""
    dirs = [
        Path(settings.log_file).parent,
        Path(settings.ml_models.model_cache_dir),
        Path(settings.data_source.local_path),
        Path("data/raw"),
        Path("data/processed"),
        Path("data/models"),
        Path("logs"),
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


# Initialize directories on import
ensure_directories()