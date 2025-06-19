"""
Comprehensive Error Handling Framework for Multi-LLM Integration

This module provides robust error handling, retry logic, fallback mechanisms,
and monitoring for all LLM API interactions.
"""

import logging
import time
import random
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import traceback


logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Classification of different error types"""
    API_KEY_MISSING = "api_key_missing"
    API_KEY_INVALID = "api_key_invalid"
    RATE_LIMIT = "rate_limit"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_AVAILABLE = "model_not_available"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    PROVIDER_ERROR = "provider_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorDetails:
    """Detailed error information"""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    provider: str
    model: str
    timestamp: datetime
    request_details: Optional[Dict[str, Any]] = None
    response_details: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    retry_count: int = 0
    is_retryable: bool = False
    suggested_action: Optional[str] = None


class RetryConfig:
    """Configuration for retry logic"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class LLMErrorHandler:
    """Comprehensive error handler for LLM API calls"""
    
    # Error patterns for different providers
    ERROR_PATTERNS = {
        'openai': {
            'api_key_missing': ['No API key provided', 'API key not found'],
            'api_key_invalid': ['Invalid API key', 'Incorrect API key'],
            'rate_limit': ['Rate limit exceeded', 'Too many requests'],
            'quota_exceeded': ['Quota exceeded', 'Usage limit exceeded'],
            'insufficient_funds': ['Insufficient funds', 'Billing issue'],
            'model_not_available': ['Model not found', 'Model not available'],
            'timeout': ['Request timeout', 'Connection timeout'],
            'network_error': ['Connection error', 'Network error']
        },
        'anthropic': {
            'api_key_missing': ['No API key provided', 'Missing API key'],
            'api_key_invalid': ['Invalid API key', 'Authentication failed'],
            'rate_limit': ['Rate limited', 'Too many requests'],
            'quota_exceeded': ['Usage quota exceeded', 'Monthly limit exceeded'],
            'model_not_available': ['Model not available', 'Unknown model'],
            'timeout': ['Request timeout', 'Timeout'],
            'network_error': ['Connection failed', 'Network error']
        },
        'mistral': {
            'api_key_missing': ['API key required', 'Missing API key'],
            'api_key_invalid': ['Invalid API key', 'Unauthorized'],
            'rate_limit': ['Rate limit', 'Too many requests'],
            'quota_exceeded': ['Quota exceeded', 'Limit exceeded'],
            'model_not_available': ['Model not found', 'Invalid model'],
            'timeout': ['Timeout', 'Request timeout'],
            'network_error': ['Connection error', 'Network issue']
        },
        'google': {
            'api_key_missing': ['API key not provided', 'Missing API key'],
            'api_key_invalid': ['Invalid API key', 'API key error'],
            'rate_limit': ['Quota exceeded', 'Rate limit exceeded'],
            'quota_exceeded': ['Daily quota exceeded', 'Usage limit'],
            'model_not_available': ['Model not found', 'Unsupported model'],
            'timeout': ['Deadline exceeded', 'Timeout'],
            'network_error': ['Connection error', 'Network failure']
        }
    }
    
    # Retry configuration for different error types
    RETRY_CONFIGS = {
        ErrorType.RATE_LIMIT: RetryConfig(max_retries=5, base_delay=5.0, max_delay=120.0),
        ErrorType.TIMEOUT: RetryConfig(max_retries=3, base_delay=2.0, max_delay=30.0),
        ErrorType.NETWORK_ERROR: RetryConfig(max_retries=3, base_delay=1.0, max_delay=15.0),
        ErrorType.PROVIDER_ERROR: RetryConfig(max_retries=2, base_delay=3.0, max_delay=30.0),
        ErrorType.UNKNOWN_ERROR: RetryConfig(max_retries=1, base_delay=1.0, max_delay=5.0)
    }
    
    def __init__(self):
        self.error_history: List[ErrorDetails] = []
        self.error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, datetime] = {}
        self.circuit_breakers: Dict[str, bool] = {}
        self.circuit_breaker_reset_time: Dict[str, datetime] = {}
    
    def classify_error(self, error: Exception, provider: str, model: str) -> ErrorDetails:
        """
        Classify and analyze an error
        
        Args:
            error: The exception that occurred
            provider: API provider name
            model: Model name
            
        Returns:
            ErrorDetails object with classification and analysis
        """
        error_message = str(error).lower()
        error_type = ErrorType.UNKNOWN_ERROR
        severity = ErrorSeverity.MEDIUM
        is_retryable = False
        suggested_action = None
        
        # Check provider-specific error patterns
        provider_patterns = self.ERROR_PATTERNS.get(provider, {})
        
        for err_type, patterns in provider_patterns.items():
            if any(pattern.lower() in error_message for pattern in patterns):
                error_type = ErrorType(err_type)
                break
        
        # Determine severity and retry behavior
        if error_type == ErrorType.API_KEY_MISSING:
            severity = ErrorSeverity.CRITICAL
            suggested_action = f"Configure {provider.upper()}_API_KEY in environment variables"
        
        elif error_type == ErrorType.API_KEY_INVALID:
            severity = ErrorSeverity.CRITICAL
            suggested_action = f"Verify {provider.upper()}_API_KEY is correct and active"
        
        elif error_type == ErrorType.RATE_LIMIT:
            severity = ErrorSeverity.MEDIUM
            is_retryable = True
            suggested_action = "Implement rate limiting or switch to a different model"
        
        elif error_type == ErrorType.QUOTA_EXCEEDED:
            severity = ErrorSeverity.HIGH
            suggested_action = "Check billing/usage limits or upgrade plan"
        
        elif error_type == ErrorType.TIMEOUT:
            severity = ErrorSeverity.LOW
            is_retryable = True
            suggested_action = "Retry with shorter timeout or smaller request"
        
        elif error_type == ErrorType.NETWORK_ERROR:
            severity = ErrorSeverity.MEDIUM
            is_retryable = True
            suggested_action = "Check network connectivity and retry"
        
        elif error_type == ErrorType.MODEL_NOT_AVAILABLE:
            severity = ErrorSeverity.HIGH
            suggested_action = f"Check if {model} is available for {provider} or use a different model"
        
        elif error_type == ErrorType.INSUFFICIENT_FUNDS:
            severity = ErrorSeverity.CRITICAL
            suggested_action = f"Add funds to {provider} account or check billing settings"
        
        else:
            is_retryable = True
            suggested_action = "Review error details and retry with different parameters"
        
        error_details = ErrorDetails(
            error_type=error_type,
            severity=severity,
            message=str(error),
            provider=provider,
            model=model,
            timestamp=datetime.now(),
            stack_trace=traceback.format_exc(),
            is_retryable=is_retryable,
            suggested_action=suggested_action
        )
        
        # Record error
        self._record_error(error_details)
        
        return error_details
    
    def _record_error(self, error_details: ErrorDetails):
        """Record error for analysis and monitoring"""
        self.error_history.append(error_details)
        
        # Update error counts
        key = f"{error_details.provider}/{error_details.model}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        self.last_error_time[key] = error_details.timestamp
        
        # Circuit breaker logic
        if error_details.severity == ErrorSeverity.CRITICAL:
            self._trigger_circuit_breaker(key)
        
        # Log error
        logger.error(
            f"LLM Error [{error_details.error_type.value}]: {error_details.message} "
            f"({error_details.provider}/{error_details.model})"
        )
    
    def _trigger_circuit_breaker(self, provider_model_key: str, duration_minutes: int = 10):
        """Trigger circuit breaker for a provider/model combination"""
        self.circuit_breakers[provider_model_key] = True
        self.circuit_breaker_reset_time[provider_model_key] = datetime.now() + timedelta(minutes=duration_minutes)
        
        logger.warning(f"Circuit breaker triggered for {provider_model_key} for {duration_minutes} minutes")
    
    def is_circuit_breaker_open(self, provider: str, model: str) -> bool:
        """Check if circuit breaker is open for a provider/model"""
        key = f"{provider}/{model}"
        
        if key not in self.circuit_breakers:
            return False
        
        # Check if reset time has passed
        if datetime.now() >= self.circuit_breaker_reset_time.get(key, datetime.now()):
            self.circuit_breakers[key] = False
            del self.circuit_breaker_reset_time[key]
            logger.info(f"Circuit breaker reset for {key}")
            return False
        
        return self.circuit_breakers[key]
    
    def should_retry(self, error_details: ErrorDetails) -> bool:
        """Determine if an error should be retried"""
        # Don't retry if circuit breaker is open
        if self.is_circuit_breaker_open(error_details.provider, error_details.model):
            return False
        
        # Don't retry critical errors that aren't retryable
        if error_details.severity == ErrorSeverity.CRITICAL and not error_details.is_retryable:
            return False
        
        # Check if max retries reached
        retry_config = self.RETRY_CONFIGS.get(error_details.error_type)
        if retry_config and error_details.retry_count >= retry_config.max_retries:
            return False
        
        return error_details.is_retryable
    
    def get_retry_delay(self, error_details: ErrorDetails) -> float:
        """Get delay before retry"""
        retry_config = self.RETRY_CONFIGS.get(error_details.error_type, RetryConfig())
        return retry_config.get_delay(error_details.retry_count)
    
    def handle_with_retry(
        self,
        func: Callable,
        provider: str,
        model: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic and error handling
        
        Args:
            func: Function to execute
            provider: API provider name
            model: Model name
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or raises final error
        """
        last_error_details = None
        
        # Check circuit breaker
        if self.is_circuit_breaker_open(provider, model):
            raise Exception(f"Circuit breaker open for {provider}/{model}")
        
        for attempt in range(5):  # Max 5 attempts
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                error_details = self.classify_error(e, provider, model)
                error_details.retry_count = attempt
                last_error_details = error_details
                
                if not self.should_retry(error_details):
                    logger.error(f"Not retrying error: {error_details.error_type.value}")
                    raise e
                
                delay = self.get_retry_delay(error_details)
                logger.warning(
                    f"Retrying in {delay:.1f}s (attempt {attempt + 1}): {error_details.message}"
                )
                time.sleep(delay)
        
        # If we get here, all retries failed
        if last_error_details:
            raise Exception(f"All retries failed: {last_error_details.message}")
        else:
            raise Exception("Unknown error occurred")
    
    def get_fallback_models(self, provider: str, model: str) -> List[str]:
        """
        Get fallback models for a failed provider/model combination
        
        Args:
            provider: Failed provider
            model: Failed model
            
        Returns:
            List of alternative provider/model combinations
        """
        fallbacks = []
        
        # Define fallback mappings
        fallback_mapping = {
            'openai/gpt-4o': ['openai/gpt-4o-mini', 'anthropic/claude-3-5-sonnet-20241022', 'google/gemini-1.5-pro'],
            'openai/gpt-4o-mini': ['openai/gpt-4-turbo', 'google/gemini-1.5-flash', 'anthropic/claude-3-5-haiku-20241022'],
            'anthropic/claude-3-5-sonnet-20241022': ['openai/gpt-4o', 'google/gemini-1.5-pro', 'anthropic/claude-3-opus-20240229'],
            'anthropic/claude-3-5-haiku-20241022': ['openai/gpt-4o-mini', 'google/gemini-1.5-flash'],
            'google/gemini-1.5-pro': ['openai/gpt-4o', 'anthropic/claude-3-5-sonnet-20241022'],
            'google/gemini-1.5-flash': ['openai/gpt-4o-mini', 'anthropic/claude-3-5-haiku-20241022']
        }
        
        original_key = f"{provider}/{model}"
        fallbacks = fallback_mapping.get(original_key, [])
        
        # Filter out models with open circuit breakers
        available_fallbacks = []
        for fallback in fallbacks:
            fb_provider, fb_model = fallback.split('/', 1)
            if not self.is_circuit_breaker_open(fb_provider, fb_model):
                available_fallbacks.append(fallback)
        
        return available_fallbacks
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        if not recent_errors:
            return {
                'period_hours': hours,
                'total_errors': 0,
                'error_rate': 0,
                'by_type': {},
                'by_provider': {},
                'by_severity': {},
                'top_errors': []
            }
        
        # Group by type
        by_type = {}
        for error in recent_errors:
            error_type = error.error_type.value
            by_type[error_type] = by_type.get(error_type, 0) + 1
        
        # Group by provider
        by_provider = {}
        for error in recent_errors:
            provider = error.provider
            by_provider[provider] = by_provider.get(provider, 0) + 1
        
        # Group by severity
        by_severity = {}
        for error in recent_errors:
            severity = error.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        # Top errors with details
        top_errors = sorted(recent_errors, key=lambda x: x.timestamp, reverse=True)[:10]
        
        return {
            'period_hours': hours,
            'total_errors': len(recent_errors),
            'by_type': by_type,
            'by_provider': by_provider,
            'by_severity': by_severity,
            'top_errors': [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'type': e.error_type.value,
                    'severity': e.severity.value,
                    'provider': e.provider,
                    'model': e.model,
                    'message': e.message,
                    'suggested_action': e.suggested_action
                }
                for e in top_errors
            ]
        }
    
    def export_error_report(self, output_path: str, hours: int = 24):
        """Export detailed error report"""
        summary = self.get_error_summary(hours)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'circuit_breakers': {
                key: {
                    'active': active,
                    'reset_time': self.circuit_breaker_reset_time.get(key).isoformat() if key in self.circuit_breaker_reset_time else None
                }
                for key, active in self.circuit_breakers.items()
            },
            'error_counts': self.error_counts,
            'recommendations': self._generate_recommendations(summary)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Error report exported to {output_path}")
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on error patterns"""
        recommendations = []
        
        if summary['total_errors'] == 0:
            return ["No errors detected in the specified period."]
        
        # Check for API key issues
        if 'api_key_missing' in summary['by_type'] or 'api_key_invalid' in summary['by_type']:
            recommendations.append("⚠️  API Key Issues: Review and update API keys in environment variables.")
        
        # Check for rate limiting
        if 'rate_limit' in summary['by_type']:
            recommendations.append("🚦 Rate Limiting: Implement request throttling or upgrade API plans.")
        
        # Check for quota issues
        if 'quota_exceeded' in summary['by_type']:
            recommendations.append("📊 Quota Exceeded: Monitor usage and consider upgrading plans.")
        
        # Check for network issues
        if 'network_error' in summary['by_type'] or 'timeout' in summary['by_type']:
            recommendations.append("🌐 Network Issues: Check connectivity and implement retry logic.")
        
        # Check for high error rates
        if summary['total_errors'] > 10:
            recommendations.append("🔄 High Error Rate: Consider implementing fallback models and circuit breakers.")
        
        # Provider-specific recommendations
        if 'openai' in summary['by_provider'] and summary['by_provider']['openai'] > 5:
            recommendations.append("🤖 OpenAI Issues: Consider using alternative models or check OpenAI status.")
        
        return recommendations


# Global error handler instance
error_handler = LLMErrorHandler()