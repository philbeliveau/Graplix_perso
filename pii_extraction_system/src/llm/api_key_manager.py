"""
API Key Manager for Multi-LLM Integration

This module provides secure API key management with conditional loading,
validation, and availability checking for multiple LLM providers.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class APIKeyInfo:
    """Information about an API key"""
    provider: str
    key_name: str
    is_available: bool
    status: str
    last_validated: Optional[datetime] = None
    validation_error: Optional[str] = None


class APIKeyManager:
    """Manages API keys for multiple LLM providers"""
    
    # Supported API providers and their environment variable names
    SUPPORTED_PROVIDERS = {
        'openai': {
            'env_var': 'OPENAI_API_KEY',
            'required_length': 50,
            'prefix': 'sk-',
            'description': 'OpenAI API Key'
        },
        'anthropic': {
            'env_var': 'ANTHROPIC_API_KEY',
            'required_length': 40,
            'prefix': 'sk-',
            'description': 'Anthropic Claude API Key'
        },
        'mistral': {
            'env_var': 'MISTRAL_API_KEY',
            'required_length': 30,
            'prefix': 'm-',
            'description': 'Mistral API Key'
        },
        'google': {
            'env_var': 'GOOGLE_API_KEY',
            'required_length': 30,
            'prefix': 'AI',
            'description': 'Google AI API Key'
        },
        'deepseek': {
            'env_var': 'DEEPSEEK_API',
            'required_length': 30,
            'prefix': 'sk-',
            'description': 'DeepSeek API Key'
        },
        'nvidia': {
            'env_var': 'NVIDIA_KEY',
            'required_length': 30,
            'prefix': 'nvapi-',
            'description': 'NVIDIA API Key'
        },
        'huggingface': {
            'env_var': 'HUGGINGFACE_API_KEY',
            'required_length': 30,
            'prefix': 'hf_',
            'description': 'Hugging Face API Key'
        }
    }
    
    def __init__(self, env_file_path: Optional[str] = None):
        """
        Initialize API Key Manager
        
        Args:
            env_file_path: Path to .env file (optional)
        """
        self.env_file_path = env_file_path
        self._api_keys: Dict[str, str] = {}
        self._key_info: Dict[str, APIKeyInfo] = {}
        self._validation_cache: Dict[str, Tuple[bool, datetime]] = {}
        
        # Load environment variables
        self._load_environment_variables()
        
        # Check API key availability
        self._check_api_key_availability()
    
    def _load_environment_variables(self):
        """Load environment variables from .env file if specified"""
        if self.env_file_path and os.path.exists(self.env_file_path):
            try:
                from dotenv import load_dotenv
                load_dotenv(self.env_file_path)
                logger.info(f"Loaded environment variables from {self.env_file_path}")
            except ImportError:
                logger.warning("python-dotenv not installed, skipping .env file loading")
            except Exception as e:
                logger.warning(f"Failed to load .env file: {e}")
    
    def _check_api_key_availability(self):
        """Check availability of all API keys"""
        for provider, config in self.SUPPORTED_PROVIDERS.items():
            env_var = config['env_var']
            api_key = os.getenv(env_var)
            
            if api_key and api_key.strip():
                # Basic validation
                is_valid, error = self._validate_api_key(provider, api_key)
                
                if is_valid:
                    self._api_keys[provider] = api_key
                    self._key_info[provider] = APIKeyInfo(
                        provider=provider,
                        key_name=env_var,
                        is_available=True,
                        status="Available",
                        last_validated=datetime.now()
                    )
                    logger.info(f"✓ {config['description']} is available")
                else:
                    self._key_info[provider] = APIKeyInfo(
                        provider=provider,
                        key_name=env_var,
                        is_available=False,
                        status="Invalid",
                        validation_error=error
                    )
                    logger.warning(f"✗ {config['description']} is invalid: {error}")
            else:
                self._key_info[provider] = APIKeyInfo(
                    provider=provider,
                    key_name=env_var,
                    is_available=False,
                    status="Missing",
                    validation_error=f"Environment variable {env_var} not set"
                )
                logger.info(f"- {config['description']} is not configured")
    
    def _validate_api_key(self, provider: str, api_key: str) -> Tuple[bool, Optional[str]]:
        """
        Validate API key format
        
        Args:
            provider: Provider name
            api_key: API key to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if provider not in self.SUPPORTED_PROVIDERS:
            return False, f"Unsupported provider: {provider}"
        
        config = self.SUPPORTED_PROVIDERS[provider]
        
        # Check length
        if len(api_key) < config['required_length']:
            return False, f"API key too short (minimum {config['required_length']} characters)"
        
        # Check prefix (if specified)
        if config.get('prefix') and not api_key.startswith(config['prefix']):
            return False, f"API key should start with '{config['prefix']}'"
        
        # Check for obviously invalid patterns
        if api_key.count(' ') > 0:
            return False, "API key contains spaces"
        
        if api_key in ['your-api-key', 'sk-your-key', 'placeholder']:
            return False, "API key appears to be a placeholder"
        
        return True, None
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider
        
        Args:
            provider: Provider name
            
        Returns:
            API key if available, None otherwise
        """
        return self._api_keys.get(provider)
    
    def is_available(self, provider: str) -> bool:
        """
        Check if API key is available for a provider
        
        Args:
            provider: Provider name
            
        Returns:
            True if API key is available and valid
        """
        return provider in self._api_keys
    
    def get_available_providers(self) -> List[str]:
        """Get list of providers with valid API keys"""
        return list(self._api_keys.keys())
    
    def get_provider_status(self, provider: str) -> Optional[APIKeyInfo]:
        """
        Get detailed status information for a provider
        
        Args:
            provider: Provider name
            
        Returns:
            APIKeyInfo object with status details
        """
        return self._key_info.get(provider)
    
    def get_all_provider_status(self) -> Dict[str, APIKeyInfo]:
        """Get status for all providers"""
        return self._key_info.copy()
    
    def validate_api_key_online(self, provider: str, force_refresh: bool = False) -> bool:
        """
        Validate API key by making a test request (online validation)
        
        Args:
            provider: Provider name
            force_refresh: Force refresh even if cached
            
        Returns:
            True if API key is valid online
        """
        if not self.is_available(provider):
            return False
        
        # Check cache first (valid for 1 hour)
        if not force_refresh and provider in self._validation_cache:
            is_valid, timestamp = self._validation_cache[provider]
            if datetime.now() - timestamp < timedelta(hours=1):
                return is_valid
        
        # Perform online validation
        api_key = self.get_api_key(provider)
        is_valid = self._test_api_key_online(provider, api_key)
        
        # Update cache
        self._validation_cache[provider] = (is_valid, datetime.now())
        
        # Update key info
        if provider in self._key_info:
            self._key_info[provider].last_validated = datetime.now()
            if is_valid:
                self._key_info[provider].status = "Validated Online"
                self._key_info[provider].validation_error = None
            else:
                self._key_info[provider].status = "Online Validation Failed"
                self._key_info[provider].validation_error = "API key rejected by provider"
        
        return is_valid
    
    def _test_api_key_online(self, provider: str, api_key: str) -> bool:
        """
        Test API key with actual API call
        
        Args:
            provider: Provider name
            api_key: API key to test
            
        Returns:
            True if API key works
        """
        try:
            if provider == 'openai':
                import openai
                client = openai.OpenAI(api_key=api_key)
                # Test with a minimal request
                client.models.list()
                return True
                
            elif provider == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                # Test with a minimal request
                client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                return True
                
            elif provider == 'mistral':
                from mistralai.client import MistralClient
                client = MistralClient(api_key=api_key)
                # Test by listing models
                client.list_models()
                return True
                
            elif provider == 'google':
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                # Test by listing models
                list(genai.list_models())
                return True
                
            else:
                # For other providers, assume valid if key format is correct
                return True
                
        except Exception as e:
            logger.debug(f"Online validation failed for {provider}: {e}")
            return False
    
    def get_usage_summary(self) -> Dict[str, any]:
        """Get summary of API key usage and availability"""
        available_count = len(self._api_keys)
        total_count = len(self.SUPPORTED_PROVIDERS)
        
        return {
            "total_providers": total_count,
            "available_providers": available_count,
            "availability_percentage": (available_count / total_count) * 100,
            "available_list": list(self._api_keys.keys()),
            "missing_list": [p for p in self.SUPPORTED_PROVIDERS if p not in self._api_keys],
            "provider_details": {
                provider: {
                    "available": info.is_available,
                    "status": info.status,
                    "last_validated": info.last_validated.isoformat() if info.last_validated else None,
                    "error": info.validation_error
                }
                for provider, info in self._key_info.items()
            }
        }
    
    def export_status_report(self, output_path: Optional[str] = None) -> str:
        """
        Export detailed status report
        
        Args:
            output_path: Path to save report (optional)
            
        Returns:
            Report as string
        """
        summary = self.get_usage_summary()
        
        report = f"""
# API Key Manager Status Report
Generated: {datetime.now().isoformat()}

## Summary
- Total Providers: {summary['total_providers']}
- Available Providers: {summary['available_providers']}
- Availability: {summary['availability_percentage']:.1f}%

## Available Providers
"""
        
        for provider in summary['available_list']:
            info = self._key_info[provider]
            report += f"- ✓ {provider.upper()}: {info.status}\n"
        
        report += "\n## Missing/Invalid Providers\n"
        
        for provider in self.SUPPORTED_PROVIDERS:
            if provider not in summary['available_list']:
                info = self._key_info.get(provider)
                if info:
                    report += f"- ✗ {provider.upper()}: {info.status}"
                    if info.validation_error:
                        report += f" ({info.validation_error})"
                    report += "\n"
        
        report += f"\n## Setup Instructions\n"
        report += "To enable missing providers, add these to your .env file:\n\n"
        
        for provider in summary['missing_list']:
            config = self.SUPPORTED_PROVIDERS[provider]
            report += f"# {config['description']}\n"
            report += f"{config['env_var']}=your-{provider}-api-key-here\n\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Status report saved to {output_path}")
        
        return report


# Global instance
api_key_manager = APIKeyManager()