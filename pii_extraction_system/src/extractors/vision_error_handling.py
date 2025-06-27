"""
Comprehensive Error Handling and Fallback Mechanisms for Vision-LLM PII Extraction

This module provides robust error handling, retry mechanisms, and fallback strategies
to ensure production reliability and graceful degradation.
"""

import os
import time
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"           # Minor issues, continue processing
    MEDIUM = "medium"     # Notable issues, may affect quality
    HIGH = "high"         # Serious issues, fallback recommended
    CRITICAL = "critical" # System failure, immediate action required


class ErrorCategory(Enum):
    """Error categories for classification"""
    MODEL_ERROR = "model_error"              # LLM/Model failures
    NETWORK_ERROR = "network_error"          # Network connectivity issues
    AUTHENTICATION_ERROR = "auth_error"      # API key/authentication failures
    QUOTA_ERROR = "quota_error"              # Rate limits/quota exceeded
    FORMAT_ERROR = "format_error"            # Data format/parsing issues
    VALIDATION_ERROR = "validation_error"    # Data validation failures
    CONFIGURATION_ERROR = "config_error"     # Configuration issues
    RESOURCE_ERROR = "resource_error"        # Memory/disk/CPU issues
    TIMEOUT_ERROR = "timeout_error"          # Request timeouts
    UNKNOWN_ERROR = "unknown_error"          # Unclassified errors


class FallbackStrategy(Enum):
    """Fallback strategies"""
    RETRY_SAME = "retry_same"                # Retry with same model
    RETRY_DIFFERENT = "retry_different"      # Try different model
    DEGRADE_QUALITY = "degrade_quality"      # Lower quality settings
    TRADITIONAL_METHODS = "traditional"      # Fall back to non-vision methods
    MANUAL_REVIEW = "manual_review"          # Flag for manual processing
    SKIP_PROCESSING = "skip_processing"      # Skip this document


@dataclass
class ErrorDetails:
    """Detailed error information"""
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    component: str
    model_used: Optional[str] = None
    document_id: Optional[str] = None
    retry_count: int = 0
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackResult:
    """Result of fallback processing"""
    success: bool
    strategy_used: FallbackStrategy
    fallback_component: str
    processing_time: float
    quality_degradation: float = 0.0  # 0-1 scale
    error_details: Optional[ErrorDetails] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    retry_on_errors: List[ErrorCategory] = field(default_factory=lambda: [
        ErrorCategory.NETWORK_ERROR,
        ErrorCategory.TIMEOUT_ERROR,
        ErrorCategory.QUOTA_ERROR
    ])


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying to close circuit
            expected_exception: Exception type to monitor
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func):
        """Decorator for circuit breaker"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call(func, *args, **kwargs)
        return wrapper
    
    def _call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class VisionErrorHandler:
    """
    Comprehensive error handling system for vision-based PII extraction
    """
    
    def __init__(self,
                 retry_config: Optional[RetryConfig] = None,
                 enable_circuit_breaker: bool = True,
                 max_error_history: int = 1000):
        """
        Initialize error handler
        
        Args:
            retry_config: Retry configuration
            enable_circuit_breaker: Enable circuit breaker pattern
            max_error_history: Maximum errors to keep in history
        """
        self.retry_config = retry_config or RetryConfig()
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Error tracking
        self.error_history = deque(maxlen=max_error_history)
        self.error_counts = defaultdict(int)
        self.component_errors = defaultdict(lambda: defaultdict(int))
        
        # Circuit breakers for different components
        self.circuit_breakers = {}
        if enable_circuit_breaker:
            self._initialize_circuit_breakers()
        
        # Fallback strategies mapping
        self.fallback_strategies = {
            ErrorCategory.MODEL_ERROR: FallbackStrategy.RETRY_DIFFERENT,
            ErrorCategory.NETWORK_ERROR: FallbackStrategy.RETRY_SAME,
            ErrorCategory.AUTHENTICATION_ERROR: FallbackStrategy.TRADITIONAL_METHODS,
            ErrorCategory.QUOTA_ERROR: FallbackStrategy.DEGRADE_QUALITY,
            ErrorCategory.FORMAT_ERROR: FallbackStrategy.TRADITIONAL_METHODS,
            ErrorCategory.VALIDATION_ERROR: FallbackStrategy.MANUAL_REVIEW,
            ErrorCategory.CONFIGURATION_ERROR: FallbackStrategy.TRADITIONAL_METHODS,
            ErrorCategory.RESOURCE_ERROR: FallbackStrategy.DEGRADE_QUALITY,
            ErrorCategory.TIMEOUT_ERROR: FallbackStrategy.RETRY_SAME,
            ErrorCategory.UNKNOWN_ERROR: FallbackStrategy.TRADITIONAL_METHODS
        }
        
        logger.info("VisionErrorHandler initialized")
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for different components"""
        components = [
            'openai_provider',
            'anthropic_provider', 
            'google_provider',
            'document_classifier',
            'prompt_router',
            'vision_extractor',
            'local_model_manager'
        ]
        
        for component in components:
            self.circuit_breakers[component] = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60.0
            )
    
    def handle_error(self,
                    error: Exception,
                    component: str,
                    context: Dict[str, Any] = None,
                    model_used: str = None,
                    document_id: str = None) -> ErrorDetails:
        """
        Handle and categorize an error
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            context: Additional context information
            model_used: Model being used when error occurred
            document_id: Document being processed when error occurred
            
        Returns:
            ErrorDetails object with classified error information
        """
        error_details = self._classify_error(
            error=error,
            component=component,
            context=context or {},
            model_used=model_used,
            document_id=document_id
        )
        
        # Record error
        self._record_error(error_details)
        
        # Log error
        self._log_error(error_details)
        
        return error_details
    
    def _classify_error(self,
                       error: Exception,
                       component: str,
                       context: Dict[str, Any],
                       model_used: str = None,
                       document_id: str = None) -> ErrorDetails:
        """Classify error into category and severity"""
        
        error_message = str(error)
        error_type = type(error).__name__
        
        # Determine category
        category = self._determine_error_category(error, error_message)
        
        # Determine severity
        severity = self._determine_error_severity(category, error, context)
        
        # Generate unique error ID
        error_id = f"{component}_{category.value}_{int(time.time())}_{hash(error_message) % 10000}"
        
        return ErrorDetails(
            error_id=error_id,
            timestamp=time.time(),
            category=category,
            severity=severity,
            message=error_message,
            component=component,
            model_used=model_used,
            document_id=document_id,
            stack_trace=traceback.format_exc(),
            context=context,
            metadata={
                'error_type': error_type,
                'component_state': self._get_component_state(component)
            }
        )
    
    def _determine_error_category(self, error: Exception, error_message: str) -> ErrorCategory:
        """Determine error category based on error type and message"""
        
        error_message_lower = error_message.lower()
        
        # Network/connection errors
        if any(keyword in error_message_lower for keyword in [
            'connection', 'network', 'timeout', 'unreachable', 'dns'
        ]):
            return ErrorCategory.NETWORK_ERROR
        
        # Authentication errors
        if any(keyword in error_message_lower for keyword in [
            'unauthorized', 'authentication', 'api key', 'invalid key', 'forbidden'
        ]):
            return ErrorCategory.AUTHENTICATION_ERROR
        
        # Quota/rate limit errors
        if any(keyword in error_message_lower for keyword in [
            'quota', 'rate limit', 'too many requests', 'limit exceeded'
        ]):
            return ErrorCategory.QUOTA_ERROR
        
        # Format/parsing errors
        if any(keyword in error_message_lower for keyword in [
            'json', 'parse', 'format', 'decode', 'invalid response'
        ]):
            return ErrorCategory.FORMAT_ERROR
        
        # Validation errors
        if any(keyword in error_message_lower for keyword in [
            'validation', 'invalid input', 'bad request', 'malformed'
        ]):
            return ErrorCategory.VALIDATION_ERROR
        
        # Configuration errors
        if any(keyword in error_message_lower for keyword in [
            'configuration', 'config', 'not found', 'missing', 'invalid setting'
        ]):
            return ErrorCategory.CONFIGURATION_ERROR
        
        # Resource errors
        if any(keyword in error_message_lower for keyword in [
            'memory', 'disk space', 'resource', 'out of memory', 'cpu'
        ]):
            return ErrorCategory.RESOURCE_ERROR
        
        # Timeout errors
        if any(keyword in error_message_lower for keyword in [
            'timeout', 'timed out', 'deadline exceeded'
        ]):
            return ErrorCategory.TIMEOUT_ERROR
        
        # Model-specific errors
        if any(keyword in error_message_lower for keyword in [
            'model', 'inference', 'generation', 'completion'
        ]):
            return ErrorCategory.MODEL_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _determine_error_severity(self,
                                 category: ErrorCategory,
                                 error: Exception,
                                 context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity"""
        
        # Critical errors that require immediate attention
        if category in [ErrorCategory.AUTHENTICATION_ERROR, ErrorCategory.CONFIGURATION_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High severity for resource and unknown errors
        if category in [ErrorCategory.RESOURCE_ERROR, ErrorCategory.UNKNOWN_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity for model and format errors
        if category in [ErrorCategory.MODEL_ERROR, ErrorCategory.FORMAT_ERROR, ErrorCategory.VALIDATION_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity for temporary network/quota issues
        if category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.QUOTA_ERROR, ErrorCategory.TIMEOUT_ERROR]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _record_error(self, error_details: ErrorDetails):
        """Record error for tracking and analysis"""
        self.error_history.append(error_details)
        self.error_counts[error_details.category] += 1
        self.component_errors[error_details.component][error_details.category] += 1
    
    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level"""
        log_message = (f"Error in {error_details.component}: "
                      f"{error_details.category.value} - {error_details.message}")
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _get_component_state(self, component: str) -> Dict[str, Any]:
        """Get current state of component for error context"""
        state = {'component': component}
        
        if component in self.circuit_breakers:
            cb = self.circuit_breakers[component]
            state['circuit_breaker'] = {
                'state': cb.state,
                'failure_count': cb.failure_count,
                'last_failure': cb.last_failure_time
            }
        
        # Component-specific error history
        if component in self.component_errors:
            state['recent_errors'] = dict(self.component_errors[component])
        
        return state
    
    def should_retry(self, error_details: ErrorDetails) -> bool:
        """Determine if operation should be retried"""
        
        # Check retry count
        if error_details.retry_count >= self.retry_config.max_retries:
            return False
        
        # Check if error category supports retry
        if error_details.category not in self.retry_config.retry_on_errors:
            return False
        
        # Check circuit breaker
        if (self.enable_circuit_breaker and 
            error_details.component in self.circuit_breakers and
            self.circuit_breakers[error_details.component].state == 'OPEN'):
            return False
        
        # Don't retry critical errors
        if error_details.severity == ErrorSeverity.CRITICAL:
            return False
        
        return True
    
    def calculate_retry_delay(self, retry_count: int) -> float:
        """Calculate delay before retry using exponential backoff"""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.backoff_factor ** retry_count),
            self.retry_config.max_delay
        )
        
        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def get_fallback_strategy(self, error_details: ErrorDetails) -> FallbackStrategy:
        """Get recommended fallback strategy for error"""
        return self.fallback_strategies.get(error_details.category, FallbackStrategy.TRADITIONAL_METHODS)
    
    def execute_with_retry(self,
                          func: Callable,
                          component: str,
                          *args,
                          **kwargs) -> Any:
        """
        Execute function with retry logic and error handling
        
        Args:
            func: Function to execute
            component: Component name for error tracking
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_error = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Apply circuit breaker if enabled
                if (self.enable_circuit_breaker and 
                    component in self.circuit_breakers):
                    cb = self.circuit_breakers[component]
                    return cb._call(func, *args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                error_details = self.handle_error(
                    error=e,
                    component=component,
                    context={'attempt': attempt + 1},
                    **kwargs.get('error_context', {})
                )
                error_details.retry_count = attempt
                
                last_error = e
                
                # Check if we should retry
                if attempt < self.retry_config.max_retries and self.should_retry(error_details):
                    delay = self.calculate_retry_delay(attempt)
                    logger.info(f"Retrying {component} after {delay:.2f}s (attempt {attempt + 2})")
                    time.sleep(delay)
                else:
                    break
        
        # All retries failed
        raise last_error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        
        recent_errors = list(self.error_history)[-100:]  # Last 100 errors
        current_time = time.time()
        
        # Calculate error rates
        error_rates = {}
        time_windows = [3600, 86400]  # 1 hour, 1 day
        
        for window in time_windows:
            window_errors = [e for e in recent_errors 
                           if current_time - e.timestamp <= window]
            error_rates[f"{window}s"] = len(window_errors)
        
        # Component health
        component_health = {}
        for component, error_counts in self.component_errors.items():
            total_errors = sum(error_counts.values())
            component_health[component] = {
                'total_errors': total_errors,
                'error_categories': dict(error_counts),
                'circuit_breaker_state': (
                    self.circuit_breakers[component].state 
                    if component in self.circuit_breakers 
                    else 'N/A'
                )
            }
        
        return {
            'total_errors': len(self.error_history),
            'error_counts_by_category': dict(self.error_counts),
            'error_rates': error_rates,
            'component_health': component_health,
            'circuit_breakers_enabled': self.enable_circuit_breaker,
            'recent_errors': [
                {
                    'timestamp': e.timestamp,
                    'component': e.component,
                    'category': e.category.value,
                    'severity': e.severity.value,
                    'message': e.message[:100]  # Truncated
                }
                for e in recent_errors[-10:]  # Last 10 errors
            ]
        }
    
    def reset_component_errors(self, component: str):
        """Reset error tracking for specific component"""
        if component in self.component_errors:
            del self.component_errors[component]
        
        if component in self.circuit_breakers:
            cb = self.circuit_breakers[component]
            cb.failure_count = 0
            cb.state = 'CLOSED'
            cb.last_failure_time = None
        
        logger.info(f"Reset error tracking for component: {component}")
    
    def add_fallback_strategy(self, category: ErrorCategory, strategy: FallbackStrategy):
        """Add or update fallback strategy for error category"""
        self.fallback_strategies[category] = strategy
        logger.info(f"Updated fallback strategy for {category.value}: {strategy.value}")


# Global error handler instance
error_handler = VisionErrorHandler()


def handle_vision_errors(component: str):
    """Decorator for automatic error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return error_handler.execute_with_retry(func, component, *args, **kwargs)
        return wrapper
    return decorator


# Utility functions for common error handling patterns

def safe_execute(func: Callable, 
                default_return: Any = None,
                component: str = "unknown",
                **kwargs) -> Any:
    """Safely execute function with error handling"""
    try:
        return func(**kwargs)
    except Exception as e:
        error_handler.handle_error(e, component, kwargs)
        return default_return


def with_fallback(primary_func: Callable,
                 fallback_func: Callable,
                 component: str = "unknown",
                 **kwargs) -> Any:
    """Execute primary function with fallback on error"""
    try:
        return primary_func(**kwargs)
    except Exception as e:
        error_details = error_handler.handle_error(e, component, kwargs)
        logger.warning(f"Primary function failed, using fallback: {error_details.message}")
        return fallback_func(**kwargs)