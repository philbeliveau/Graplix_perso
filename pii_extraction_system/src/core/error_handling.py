"""Comprehensive error handling system for the PII extraction system."""

import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json

from loguru import logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    CONFIGURATION = "configuration"
    DATA_PROCESSING = "data_processing"
    ML_MODEL = "ml_model"
    SECURITY = "security"
    PRIVACY = "privacy"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL_API = "external_api"
    USER_INPUT = "user_input"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Error context information."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    environment: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class PIIExtractionError(Exception):
    """Base exception for PII extraction system."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        component: str = "unknown",
        operation: str = "unknown",
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Initialize PII extraction error."""
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.component = component
        self.operation = operation
        self.error_code = error_code or self._generate_error_code()
        self.context = context or {}
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.now()
    
    def _generate_error_code(self) -> str:
        """Generate error code based on category and component."""
        return f"{self.category.value.upper()}_{self.component.upper()}_{self.severity.value.upper()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "component": self.component,
            "operation": self.operation,
            "error_code": self.error_code,
            "context": self.context,
            "traceback": traceback.format_exc()
        }


class ConfigurationError(PIIExtractionError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            component="config",
            **kwargs
        )
        if config_key:
            self.context["config_key"] = config_key


class DataProcessingError(PIIExtractionError):
    """Data processing errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA_PROCESSING,
            component="processor",
            **kwargs
        )
        if file_path:
            self.context["file_path"] = file_path


class MLModelError(PIIExtractionError):
    """ML model errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.ML_MODEL,
            component="ml_model",
            **kwargs
        )
        if model_name:
            self.context["model_name"] = model_name


class SecurityError(PIIExtractionError):
    """Security-related errors."""
    
    def __init__(self, message: str, security_context: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.SECURITY,
            component="security",
            **kwargs
        )
        if security_context:
            self.context["security_context"] = security_context


class PrivacyError(PIIExtractionError):
    """Privacy compliance errors."""
    
    def __init__(self, message: str, privacy_rule: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.PRIVACY,
            component="privacy",
            **kwargs
        )
        if privacy_rule:
            self.context["privacy_rule"] = privacy_rule


class InfrastructureError(PIIExtractionError):
    """Infrastructure-related errors."""
    
    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INFRASTRUCTURE,
            component="infrastructure",
            **kwargs
        )
        if service:
            self.context["service"] = service


class ExternalAPIError(PIIExtractionError):
    """External API errors."""
    
    def __init__(self, message: str, api_name: Optional[str] = None, status_code: Optional[int] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.EXTERNAL_API,
            component="external_api",
            **kwargs
        )
        if api_name:
            self.context["api_name"] = api_name
        if status_code:
            self.context["status_code"] = status_code


class UserInputError(PIIExtractionError):
    """User input validation errors."""
    
    def __init__(self, message: str, field_name: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.USER_INPUT,
            component="validation",
            **kwargs
        )
        if field_name:
            self.context["field_name"] = field_name


class ErrorHandler:
    """Centralized error handling system."""
    
    def __init__(self, log_file: Optional[Path] = None):
        """Initialize error handler."""
        self.log_file = log_file or Path("logs/errors.json")
        self.error_counts: Dict[str, int] = {}
        self.recent_errors: List[Dict[str, Any]] = []
        self.max_recent_errors = 100
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def handle_error(
        self,
        error: Union[Exception, PIIExtractionError],
        context: Optional[Dict[str, Any]] = None,
        notify: bool = True
    ) -> str:
        """Handle an error and return error ID."""
        
        # Convert to PIIExtractionError if needed
        if not isinstance(error, PIIExtractionError):
            error = PIIExtractionError(
                message=str(error),
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.SYSTEM,
                component="unknown",
                operation="unknown",
                context=context
            )
        
        # Update error counts
        error_key = f"{error.category.value}_{error.component}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error
        self._log_error(error)
        
        # Store in recent errors
        error_dict = error.to_dict()
        self.recent_errors.append(error_dict)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        # Save to file
        self._save_error_to_file(error_dict)
        
        # Send notifications if needed
        if notify and error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self._send_error_notification(error)
        
        return error.error_id
    
    def _log_error(self, error: PIIExtractionError) -> None:
        """Log error with appropriate level."""
        log_data = {
            "error_id": error.error_id,
            "error_code": error.error_code,
            "component": error.component,
            "operation": error.operation,
            "context": error.context
        }
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"{error.message}", **log_data)
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"{error.message}", **log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"{error.message}", **log_data)
        else:
            logger.info(f"{error.message}", **log_data)
    
    def _save_error_to_file(self, error_dict: Dict[str, Any]) -> None:
        """Save error to JSON file."""
        try:
            # Read existing errors
            errors = []
            if self.log_file.exists():
                try:
                    with open(self.log_file, 'r') as f:
                        errors = json.load(f)
                except json.JSONDecodeError:
                    errors = []
            
            # Add new error
            errors.append(error_dict)
            
            # Keep only last 1000 errors
            if len(errors) > 1000:
                errors = errors[-1000:]
            
            # Save back to file
            with open(self.log_file, 'w') as f:
                json.dump(errors, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save error to file: {e}")
    
    def _send_error_notification(self, error: PIIExtractionError) -> None:
        """Send error notification."""
        try:
            # TODO: Implement notification logic (Slack, email, etc.)
            logger.info(f"Error notification would be sent for {error.error_id}")
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts_by_type": self.error_counts.copy(),
            "recent_errors_count": len(self.recent_errors),
            "severe_errors_count": len([
                e for e in self.recent_errors 
                if e.get("severity") in ["high", "critical"]
            ])
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent errors."""
        return self.recent_errors[-limit:]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[Dict[str, Any]]:
        """Get errors by category."""
        return [
            e for e in self.recent_errors 
            if e.get("category") == category.value
        ]
    
    def clear_error_history(self) -> None:
        """Clear error history."""
        self.recent_errors.clear()
        self.error_counts.clear()
        
        if self.log_file.exists():
            self.log_file.unlink()
    
    def export_error_report(self, output_file: Path) -> None:
        """Export comprehensive error report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": self.get_error_stats(),
            "recent_errors": self.recent_errors,
            "error_counts": self.error_counts
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Error report exported to {output_file}")


# Global error handler instance
error_handler = ErrorHandler()


def handle_error(
    error: Union[Exception, PIIExtractionError],
    context: Optional[Dict[str, Any]] = None,
    notify: bool = True
) -> str:
    """Global error handling function."""
    return error_handler.handle_error(error, context, notify)


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance."""
    return error_handler


# Decorator for error handling
def handle_exceptions(
    component: str = "unknown",
    operation: str = "unknown",
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.SYSTEM,
    reraise: bool = True
):
    """Decorator for automatic error handling."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except PIIExtractionError:
                # Re-raise PIIExtractionError as-is
                raise
            except Exception as e:
                # Convert to PIIExtractionError
                error = PIIExtractionError(
                    message=str(e),
                    severity=severity,
                    category=category,
                    component=component,
                    operation=operation or func.__name__
                )
                
                handle_error(error)
                
                if reraise:
                    raise error
                return None
        
        return wrapper
    return decorator


# Context manager for error handling
class ErrorContext:
    """Context manager for error handling."""
    
    def __init__(
        self,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM
    ):
        self.component = component
        self.operation = operation
        self.context = context or {}
        self.severity = severity
        self.category = category
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if not isinstance(exc_val, PIIExtractionError):
                error = PIIExtractionError(
                    message=str(exc_val),
                    severity=self.severity,
                    category=self.category,
                    component=self.component,
                    operation=self.operation,
                    context=self.context
                )
            else:
                error = exc_val
            
            handle_error(error)
        
        return False  # Don't suppress the exception