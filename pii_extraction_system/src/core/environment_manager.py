"""Environment configuration manager for the PII extraction system."""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from loguru import logger
from pydantic import BaseModel, Field


@dataclass
class EnvironmentInfo:
    """Environment information."""
    name: str
    is_production: bool
    debug_enabled: bool
    monitoring_enabled: bool
    created_at: datetime


class EnvironmentValidator(BaseModel):
    """Environment configuration validator."""
    
    required_vars: List[str] = Field(default_factory=list)
    optional_vars: List[str] = Field(default_factory=list)
    sensitive_vars: List[str] = Field(default_factory=list)
    
    def validate_environment(self, env_vars: Dict[str, str]) -> Dict[str, Any]:
        """Validate environment variables."""
        validation_result = {
            "valid": True,
            "missing_required": [],
            "warnings": [],
            "sensitive_exposed": []
        }
        
        # Check required variables
        for var in self.required_vars:
            if var not in env_vars or not env_vars[var]:
                validation_result["missing_required"].append(var)
                validation_result["valid"] = False
        
        # Check for sensitive variables in logs
        for var in self.sensitive_vars:
            if var in env_vars and env_vars[var]:
                # Don't log sensitive values
                logger.debug(f"Sensitive variable {var} is configured")
            else:
                validation_result["warnings"].append(f"Sensitive variable {var} not set")
        
        return validation_result


class EnvironmentManager:
    """Environment configuration manager."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize environment manager."""
        self.base_path = base_path or Path.cwd()
        self.current_env: Optional[str] = None
        self.env_info: Optional[EnvironmentInfo] = None
        self.validators: Dict[str, EnvironmentValidator] = {}
        
        # Setup environment validators
        self._setup_validators()
        
        # Load current environment
        self._load_current_environment()
    
    def _setup_validators(self) -> None:
        """Setup environment validators for different environments."""
        
        # Development validator
        self.validators["development"] = EnvironmentValidator(
            required_vars=[
                "ENVIRONMENT",
                "DATABASE_URL",
                "LOG_LEVEL",
                "STREAMLIT_PORT"
            ],
            optional_vars=[
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "HUGGINGFACE_TOKEN"
            ],
            sensitive_vars=[
                "SECRET_KEY",
                "ENCRYPTION_KEY",
                "AWS_SECRET_ACCESS_KEY",
                "HUGGINGFACE_TOKEN"
            ]
        )
        
        # Staging validator
        self.validators["staging"] = EnvironmentValidator(
            required_vars=[
                "ENVIRONMENT",
                "DATABASE_URL",
                "LOG_LEVEL",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "SECRET_KEY",
                "ENCRYPTION_KEY"
            ],
            optional_vars=[
                "HUGGINGFACE_TOKEN",
                "SLACK_WEBHOOK_URL"
            ],
            sensitive_vars=[
                "SECRET_KEY",
                "ENCRYPTION_KEY",
                "AWS_SECRET_ACCESS_KEY",
                "DB_PASSWORD",
                "HUGGINGFACE_TOKEN"
            ]
        )
        
        # Production validator
        self.validators["production"] = EnvironmentValidator(
            required_vars=[
                "ENVIRONMENT",
                "DATABASE_URL",
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "SECRET_KEY",
                "ENCRYPTION_KEY",
                "DB_PASSWORD"
            ],
            optional_vars=[
                "SLACK_WEBHOOK_URL",
                "EMAIL_SMTP_SERVER"
            ],
            sensitive_vars=[
                "SECRET_KEY",
                "ENCRYPTION_KEY",
                "AWS_SECRET_ACCESS_KEY",
                "DB_PASSWORD",
                "HUGGINGFACE_TOKEN",
                "SLACK_WEBHOOK_URL"
            ]
        )
    
    def _load_current_environment(self) -> None:
        """Load current environment configuration."""
        env_name = os.getenv("ENVIRONMENT", "development")
        self.current_env = env_name
        
        # Load environment-specific .env file
        env_file = self.base_path / f".env.{env_name}"
        if env_file.exists():
            self._load_env_file(env_file)
        
        # Create environment info
        self.env_info = EnvironmentInfo(
            name=env_name,
            is_production=env_name == "production",
            debug_enabled=os.getenv("DEBUG", "false").lower() == "true",
            monitoring_enabled=os.getenv("MONITORING__ENABLE_METRICS", "false").lower() == "true",
            created_at=datetime.now()
        )
        
        logger.info(f"Loaded environment: {env_name}")
    
    def _load_env_file(self, env_file: Path) -> None:
        """Load environment variables from file."""
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Only set if not already set
                        if key not in os.environ:
                            os.environ[key] = value
        except Exception as e:
            logger.warning(f"Failed to load environment file {env_file}: {e}")
    
    def get_current_environment(self) -> str:
        """Get current environment name."""
        return self.current_env or "development"
    
    def get_environment_info(self) -> EnvironmentInfo:
        """Get environment information."""
        if not self.env_info:
            self._load_current_environment()
        return self.env_info
    
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.get_current_environment() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.get_current_environment() == "development"
    
    def is_staging(self) -> bool:
        """Check if running in staging."""
        return self.get_current_environment() == "staging"
    
    def validate_current_environment(self) -> Dict[str, Any]:
        """Validate current environment configuration."""
        env_name = self.get_current_environment()
        
        if env_name not in self.validators:
            return {
                "valid": False,
                "error": f"No validator configured for environment: {env_name}"
            }
        
        # Get current environment variables
        env_vars = dict(os.environ)
        
        # Validate
        validator = self.validators[env_name]
        result = validator.validate_environment(env_vars)
        
        # Log validation results
        if result["valid"]:
            logger.info(f"Environment {env_name} validation passed")
        else:
            logger.error(f"Environment {env_name} validation failed: {result['missing_required']}")
        
        if result["warnings"]:
            for warning in result["warnings"]:
                logger.warning(warning)
        
        return result
    
    def get_config_value(self, key: str, default: Any = None, cast_type: type = str) -> Any:
        """Get configuration value with type casting."""
        value = os.getenv(key, default)
        
        if value is None:
            return default
        
        try:
            if cast_type == bool:
                return str(value).lower() in ('true', '1', 'yes', 'on')
            elif cast_type == int:
                return int(value)
            elif cast_type == float:
                return float(value)
            elif cast_type == list:
                return [item.strip() for item in str(value).split(',') if item.strip()]
            else:
                return cast_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast {key}={value} to {cast_type}: {e}")
            return default
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "url": self.get_config_value("DATABASE_URL"),
            "pool_size": self.get_config_value("DB_POOL_SIZE", 10, int),
            "max_overflow": self.get_config_value("DB_MAX_OVERFLOW", 20, int),
            "echo": self.get_config_value("DB_ECHO", False, bool)
        }
    
    def get_aws_config(self) -> Dict[str, Any]:
        """Get AWS configuration."""
        return {
            "access_key_id": self.get_config_value("AWS_ACCESS_KEY_ID"),
            "secret_access_key": self.get_config_value("AWS_SECRET_ACCESS_KEY"),
            "region": self.get_config_value("AWS_REGION", "us-west-2"),
            "s3_bucket": self.get_config_value("DATA_SOURCE__S3_BUCKET")
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            "enabled": self.get_config_value("MONITORING__ENABLE_METRICS", False, bool),
            "prometheus_port": self.get_config_value("MONITORING__PROMETHEUS_PORT", 9090, int),
            "health_check_interval": self.get_config_value("MONITORING__HEALTH_CHECK_INTERVAL", 30, int),
            "alert_on_errors": self.get_config_value("MONITORING__ALERT_ON_ERRORS", False, bool),
            "performance_threshold_ms": self.get_config_value("MONITORING__PERFORMANCE_THRESHOLD_MS", 5000, int)
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            "secret_key": self.get_config_value("SECURITY__SECRET_KEY"),
            "encryption_key": self.get_config_value("SECURITY__ENCRYPTION_KEY"),
            "enable_auth": self.get_config_value("SECURITY__ENABLE_AUTH", True, bool),
            "session_timeout": self.get_config_value("SECURITY__SESSION_TIMEOUT", 3600, int)
        }
    
    def export_environment_info(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Export environment information."""
        info = {
            "environment": self.get_current_environment(),
            "info": self.get_environment_info().__dict__ if self.env_info else None,
            "validation": self.validate_current_environment(),
            "configs": {
                "database": self.get_database_config(),
                "aws": {k: "***" if "secret" in k.lower() or "key" in k.lower() else v 
                       for k, v in self.get_aws_config().items()},
                "monitoring": self.get_monitoring_config(),
                "security": {k: "***" if "secret" in k.lower() or "key" in k.lower() else v 
                           for k, v in self.get_security_config().items()}
            },
            "exported_at": datetime.now().isoformat()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(info, f, indent=2, default=str)
            logger.info(f"Environment info exported to {output_file}")
        
        return info
    
    def switch_environment(self, new_env: str) -> None:
        """Switch to a different environment."""
        if new_env not in ["development", "staging", "production"]:
            raise ValueError(f"Invalid environment: {new_env}")
        
        logger.info(f"Switching from {self.current_env} to {new_env}")
        
        # Update environment variable
        os.environ["ENVIRONMENT"] = new_env
        
        # Reload environment
        self._load_current_environment()
        
        # Validate new environment
        validation = self.validate_current_environment()
        if not validation["valid"]:
            logger.error(f"Environment switch validation failed: {validation}")
            raise RuntimeError(f"Invalid environment configuration: {validation['missing_required']}")


# Global environment manager instance
env_manager = EnvironmentManager()


def get_env_manager() -> EnvironmentManager:
    """Get the global environment manager instance."""
    return env_manager


def get_current_environment() -> str:
    """Get current environment name."""
    return env_manager.get_current_environment()


def is_production() -> bool:
    """Check if running in production."""
    return env_manager.is_production()


def is_development() -> bool:
    """Check if running in development."""
    return env_manager.is_development()


def get_config_value(key: str, default: Any = None, cast_type: type = str) -> Any:
    """Get configuration value with type casting."""
    return env_manager.get_config_value(key, default, cast_type)