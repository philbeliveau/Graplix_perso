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
    ocr_languages: str = Field(default="eng+fra", description="OCR languages")
    tesseract_cmd: str = Field(default="/opt/homebrew/bin/tesseract", description="Tesseract command path")


class PrivacyConfig(BaseModel):
    """Privacy and compliance configuration."""
    
    enable_redaction: bool = Field(default=True, description="Enable PII redaction")
    redaction_character: str = Field(default="*", description="Character for redaction")
    gdpr_compliance: bool = Field(default=True, description="GDPR compliance mode")
    law25_compliance: bool = Field(default=True, description="Quebec Law 25 compliance")
    audit_logging: bool = Field(default=True, description="Enable audit logging")
    data_retention_days: int = Field(default=90, description="Data retention period")


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