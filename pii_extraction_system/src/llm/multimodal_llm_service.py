"""
Multimodal LLM Service for PII Extraction

This module provides a unified interface for multiple multimodal LLMs
that can perform OCR and PII extraction from images.
"""

import os
import json
import base64
import time
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from io import BytesIO
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def extract_pii(self, image_data: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Extract PII from image using LLM"""
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> Dict[str, float]:
        """Get cost per input/output token"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass

class OpenAIProvider(LLMProvider):
    """OpenAI GPT-4o and GPT-4o-mini provider"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                logger.warning("OpenAI library not installed")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
    
    def extract_pii(self, image_data: str, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.is_available():
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                max_tokens=kwargs.get('max_tokens', 4000),
                temperature=kwargs.get('temperature', 0.0)
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost
            cost_per_token = self.get_cost_per_token()
            total_cost = (usage.prompt_tokens * cost_per_token['input'] + 
                         usage.completion_tokens * cost_per_token['output'])
            
            return {
                "success": True,
                "content": content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "estimated_cost": total_cost
                },
                "model": self.model,
                "provider": "openai"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "provider": "openai"
            }
    
    def get_cost_per_token(self) -> Dict[str, float]:
        costs = {
            "gpt-4o": {"input": 0.0025 / 1000, "output": 0.01 / 1000},
            "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
            "gpt-4-turbo": {"input": 0.01 / 1000, "output": 0.03 / 1000},
            "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000}
        }
        return costs.get(self.model, costs["gpt-4o"])
    
    def is_available(self) -> bool:
        available = bool(self.api_key and self.client)
        if available:
            logger.debug(f"OpenAI provider available for model {self.model}")
        return available

class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider"""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        self.model = model
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("Anthropic library not installed")
                self.client = None
    
    def extract_pii(self, image_data: str, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.is_available():
            raise ValueError("Anthropic API key not configured or library not installed")
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 4000),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            
            content = response.content[0].text
            usage = response.usage
            
            # Calculate cost
            cost_per_token = self.get_cost_per_token()
            total_cost = (usage.input_tokens * cost_per_token['input'] + 
                         usage.output_tokens * cost_per_token['output'])
            
            return {
                "success": True,
                "content": content,
                "usage": {
                    "prompt_tokens": usage.input_tokens,
                    "completion_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens,
                    "estimated_cost": total_cost
                },
                "model": self.model,
                "provider": "anthropic"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "provider": "anthropic"
            }
    
    def get_cost_per_token(self) -> Dict[str, float]:
        costs = {
            "claude-3-5-sonnet-20241022": {"input": 0.003 / 1000, "output": 0.015 / 1000},
            "claude-3-5-haiku-20241022": {"input": 0.001 / 1000, "output": 0.005 / 1000},
            "claude-3-opus-20240229": {"input": 0.015 / 1000, "output": 0.075 / 1000}
        }
        return costs.get(self.model, costs["claude-3-5-sonnet-20241022"])
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.client)

class GoogleProvider(LLMProvider):
    """Google Gemini provider"""
    
    def __init__(self, model: str = "gemini-1.5-pro"):
        self.model = model
        self.api_key = os.getenv('GOOGLE_API_KEY')
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(model)
            except ImportError:
                logger.warning("Google GenerativeAI library not installed")
                self.client = None
    
    def extract_pii(self, image_data: str, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.is_available():
            raise ValueError("Google API key not configured or library not installed")
        
        try:
            # Convert base64 to PIL Image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            response = self.client.generate_content([prompt, image])
            
            # Note: Google doesn't provide detailed usage stats in the same way
            # We'll estimate based on content length
            estimated_input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            estimated_output_tokens = len(response.text.split()) * 1.3
            
            cost_per_token = self.get_cost_per_token()
            total_cost = (estimated_input_tokens * cost_per_token['input'] + 
                         estimated_output_tokens * cost_per_token['output'])
            
            return {
                "success": True,
                "content": response.text,
                "usage": {
                    "prompt_tokens": int(estimated_input_tokens),
                    "completion_tokens": int(estimated_output_tokens),
                    "total_tokens": int(estimated_input_tokens + estimated_output_tokens),
                    "estimated_cost": total_cost
                },
                "model": self.model,
                "provider": "google"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "provider": "google"
            }
    
    def get_cost_per_token(self) -> Dict[str, float]:
        costs = {
            "gemini-1.5-pro": {"input": 0.0025 / 1000, "output": 0.0075 / 1000},
            "gemini-1.5-flash": {"input": 0.000075 / 1000, "output": 0.0003 / 1000}
        }
        return costs.get(self.model, costs["gemini-1.5-pro"])
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.client)

class MistralProvider(LLMProvider):
    """Mistral AI provider"""
    
    def __init__(self, model: str = "mistral-large"):
        self.model = model
        self.api_key = os.getenv('MISTRAL_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                from mistralai.client import MistralClient
                self.client = MistralClient(api_key=self.api_key)
            except ImportError:
                logger.warning("Mistral AI library not installed")
            except Exception as e:
                logger.warning(f"Failed to initialize Mistral client: {e}")
    
    def extract_pii(self, image_data: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Note: Mistral doesn't currently support vision models for images,
        so this method will return an error for image processing.
        This is included for text-based PII extraction compatibility.
        """
        if not self.is_available():
            raise ValueError("Mistral API key not configured or library not installed")
        
        # Mistral doesn't support vision yet, so we return an informative error
        return {
            "success": False,
            "error": "Mistral models do not currently support image processing. Use for text-based PII extraction only.",
            "model": self.model,
            "provider": "mistral",
            "supports_vision": False
        }
    
    def extract_text_pii(self, text: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Extract PII from text using Mistral"""
        if not self.is_available():
            raise ValueError("Mistral API key not configured or library not installed")
        
        try:
            from mistralai.models.chat_completion import ChatMessage
            
            response = self.client.chat(
                model=self.model,
                messages=[
                    ChatMessage(role="user", content=f"{prompt}\n\nText to analyze:\n{text}")
                ],
                max_tokens=kwargs.get('max_tokens', 4000),
                temperature=kwargs.get('temperature', 0.0)
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost
            cost_per_token = self.get_cost_per_token()
            total_cost = (usage.prompt_tokens * cost_per_token['input'] + 
                         usage.completion_tokens * cost_per_token['output'])
            
            return {
                "success": True,
                "content": content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "estimated_cost": total_cost
                },
                "model": self.model,
                "provider": "mistral"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "provider": "mistral"
            }
    
    def get_cost_per_token(self) -> Dict[str, float]:
        costs = {
            "mistral-large": {"input": 0.002 / 1000, "output": 0.006 / 1000},
            "mistral-medium": {"input": 0.0015 / 1000, "output": 0.0045 / 1000},
            "mistral-small": {"input": 0.0006 / 1000, "output": 0.0018 / 1000},
            "mistral-tiny": {"input": 0.00025 / 1000, "output": 0.00025 / 1000}
        }
        return costs.get(self.model, costs["mistral-large"])
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.client)

class DeepSeekProvider(LLMProvider):
    """DeepSeek AI provider"""
    
    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        # Check multiple possible API key environment variables
        self.api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('DEEPSEEK_API')
        self.client = None
        
        if self.api_key:
            try:
                import openai
                # DeepSeek uses OpenAI-compatible API
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com/v1"
                )
            except ImportError:
                logger.warning("OpenAI library not installed (required for DeepSeek)")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepSeek client: {e}")
    
    def extract_pii(self, image_data: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Note: DeepSeek doesn't currently support vision models for images,
        so this method will return an error for image processing.
        This is included for text-based PII extraction compatibility.
        """
        if not self.is_available():
            raise ValueError("DeepSeek API key not configured or library not installed")
        
        # DeepSeek doesn't support vision yet, so we return an informative error
        return {
            "success": False,
            "error": "DeepSeek models do not currently support image processing. Use for text-based PII extraction only.",
            "model": self.model,
            "provider": "deepseek",
            "supports_vision": False
        }
    
    def extract_text_pii(self, text: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Extract PII from text using DeepSeek"""
        if not self.is_available():
            raise ValueError("DeepSeek API key not configured or library not installed")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a PII extraction specialist. Extract personally identifiable information from the provided text and return it in JSON format."},
                    {"role": "user", "content": f"{prompt}\n\nText to analyze:\n{text}"}
                ],
                max_tokens=kwargs.get('max_tokens', 4000),
                temperature=kwargs.get('temperature', 0.0)
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost
            cost_per_token = self.get_cost_per_token()
            total_cost = (usage.prompt_tokens * cost_per_token['input'] + 
                         usage.completion_tokens * cost_per_token['output'])
            
            return {
                "success": True,
                "content": content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "estimated_cost": total_cost
                },
                "model": self.model,
                "provider": "deepseek"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": self.model,
                "provider": "deepseek"
            }
    
    def get_cost_per_token(self) -> Dict[str, float]:
        costs = {
            "deepseek-chat": {"input": 0.00014 / 1000, "output": 0.00028 / 1000},
            "deepseek-coder": {"input": 0.00014 / 1000, "output": 0.00028 / 1000}
        }
        return costs.get(self.model, costs["deepseek-chat"])
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.client)

class MultimodalLLMService:
    """Main service for multimodal LLM operations"""
    
    def __init__(self, cost_tracker=None, budget_config=None):
        self.providers = {}
        self.cost_tracker = cost_tracker
        self.budget_config = budget_config
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        logger.info("Initializing LLM providers...")
        
        # OpenAI models
        openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4"]
        for model in openai_models:
            try:
                provider = OpenAIProvider(model)
                if provider.is_available():
                    self.providers[f"openai/{model}"] = provider
                    logger.info(f"Initialized OpenAI provider: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider {model}: {e}")
        
        # Anthropic models
        anthropic_models = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-opus-20240229"
        ]
        for model in anthropic_models:
            try:
                provider = AnthropicProvider(model)
                if provider.is_available():
                    self.providers[f"anthropic/{model}"] = provider
                    logger.info(f"Initialized Anthropic provider: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic provider {model}: {e}")
        
        # Google models
        google_models = ["gemini-1.5-pro", "gemini-1.5-flash"]
        for model in google_models:
            try:
                provider = GoogleProvider(model)
                if provider.is_available():
                    self.providers[f"google/{model}"] = provider
                    logger.info(f"Initialized Google provider: {model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Google provider {model}: {e}")
        
        # Mistral models (text-only for now)
        mistral_models = ["mistral-large", "mistral-medium", "mistral-small", "mistral-tiny"]
        for model in mistral_models:
            try:
                provider = MistralProvider(model)
                if provider.is_available():
                    self.providers[f"mistral/{model}"] = provider
                    logger.info(f"Initialized Mistral provider: {model} (text-only)")
            except Exception as e:
                logger.warning(f"Failed to initialize Mistral provider {model}: {e}")
        
        # DeepSeek models (text-only for now)
        deepseek_models = ["deepseek-chat", "deepseek-coder"]
        for model in deepseek_models:
            try:
                provider = DeepSeekProvider(model)
                if provider.is_available():
                    self.providers[f"deepseek/{model}"] = provider
                    logger.info(f"Initialized DeepSeek provider: {model} (text-only)")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepSeek provider {model}: {e}")
        
        logger.info(f"LLM service initialized with {len(self.providers)} available models: {list(self.providers.keys())}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.providers.keys())
    
    def normalize_model_key(self, model_key: str) -> str:
        """
        Normalize model key to provider/model format.
        Handles cases where users provide just the model name.
        """
        # If already in provider/model format, return as-is
        if '/' in model_key:
            return model_key
        
        # Try to map common model names to their provider/model format
        model_mappings = {
            # OpenAI models
            "gpt-4o": "openai/gpt-4o",
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "gpt-4-turbo": "openai/gpt-4-turbo",
            "gpt-4": "openai/gpt-4",
            
            # Anthropic models
            "claude-3-5-sonnet-20241022": "anthropic/claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022": "anthropic/claude-3-5-haiku-20241022",
            "claude-3-opus-20240229": "anthropic/claude-3-opus-20240229",
            
            # Google models
            "gemini-1.5-pro": "google/gemini-1.5-pro",
            "gemini-1.5-flash": "google/gemini-1.5-flash",
            
            # Mistral models
            "mistral-large": "mistral/mistral-large",
            "mistral-medium": "mistral/mistral-medium",
            "mistral-small": "mistral/mistral-small",
            "mistral-tiny": "mistral/mistral-tiny",
            
            # DeepSeek models
            "deepseek-chat": "deepseek/deepseek-chat",
            "deepseek-coder": "deepseek/deepseek-coder"
        }
        
        normalized = model_mappings.get(model_key)
        if normalized:
            logger.debug(f"Normalized model key '{model_key}' to '{normalized}'")
            return normalized
        
        # If no mapping found, return as-is and let the caller handle the error
        logger.warning(f"No provider mapping found for model '{model_key}'. Available models: {list(self.providers.keys())}")
        return model_key
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        # Normalize the model key first
        normalized_key = self.normalize_model_key(model_key)
        
        if normalized_key not in self.providers:
            available_models = list(self.providers.keys())
            return {
                "available": False,
                "error": f"Model '{model_key}' not available",
                "normalized_key": normalized_key,
                "available_models": available_models,
                "suggestion": self._suggest_similar_model(model_key, available_models)
            }
        
        provider = self.providers[normalized_key]
        cost_info = provider.get_cost_per_token()
        
        # Check if this provider supports vision
        supports_images = hasattr(provider, 'extract_pii') and not isinstance(provider, (MistralProvider, DeepSeekProvider))
        
        return {
            "available": True,
            "provider": normalized_key.split('/')[0],
            "model": normalized_key.split('/')[1],
            "normalized_key": normalized_key,
            "cost_per_1k_input_tokens": cost_info['input'] * 1000,
            "cost_per_1k_output_tokens": cost_info['output'] * 1000,
            "supports_images": supports_images,
            "supports_json": True
        }
    
    def _suggest_similar_model(self, requested_model: str, available_models: List[str]) -> Optional[str]:
        """Suggest a similar available model based on the requested model name"""
        requested_lower = requested_model.lower()
        
        # Simple similarity matching
        best_match = None
        best_score = 0
        
        for available in available_models:
            model_name = available.split('/')[-1].lower()
            provider_name = available.split('/')[0].lower()
            
            # Check for exact matches in model name
            if requested_lower in model_name or model_name in requested_lower:
                score = len(set(requested_lower) & set(model_name))
                if score > best_score:
                    best_score = score
                    best_match = available
            
            # Check for provider matches (e.g., if someone asks for "gpt-4" suggest openai models)
            elif any(provider in requested_lower for provider in ["gpt", "openai"]) and provider_name == "openai":
                if not best_match:
                    best_match = available
        
        return best_match
    
    def debug_model_availability(self) -> Dict[str, Any]:
        """
        Comprehensive debug information about model availability
        """
        debug_info = {
            "total_providers_initialized": len(self.providers),
            "available_models": list(self.providers.keys()),
            "provider_breakdown": {},
            "api_key_status": {},
            "model_capabilities": {},
            "initialization_warnings": []
        }
        
        # Check API key status
        api_keys = {
            "OPENAI_API_KEY": bool(os.getenv('OPENAI_API_KEY')),
            "ANTHROPIC_API_KEY": bool(os.getenv('ANTHROPIC_API_KEY')),
            "GOOGLE_API_KEY": bool(os.getenv('GOOGLE_API_KEY')),
            "MISTRAL_API_KEY": bool(os.getenv('MISTRAL_API_KEY')),
            "DEEPSEEK_API_KEY": bool(os.getenv('DEEPSEEK_API_KEY')),
            "DEEPSEEK_API": bool(os.getenv('DEEPSEEK_API'))
        }
        debug_info["api_key_status"] = api_keys
        
        # Analyze available models by provider
        for model_key, provider in self.providers.items():
            provider_name = model_key.split('/')[0]
            if provider_name not in debug_info["provider_breakdown"]:
                debug_info["provider_breakdown"][provider_name] = []
            
            debug_info["provider_breakdown"][provider_name].append(model_key.split('/')[1])
            
            # Get model capabilities
            debug_info["model_capabilities"][model_key] = {
                "supports_vision": hasattr(provider, 'extract_pii') and not isinstance(provider, (MistralProvider, DeepSeekProvider)),
                "supports_text": hasattr(provider, 'extract_text_pii') or hasattr(provider, 'extract_pii'),
                "cost_per_1k_input": provider.get_cost_per_token().get('input', 0) * 1000,
                "cost_per_1k_output": provider.get_cost_per_token().get('output', 0) * 1000,
                "provider_class": provider.__class__.__name__
            }
        
        # Log detailed debug information
        logger.info("=== LLM Service Debug Information ===")
        logger.info(f"Total models available: {len(self.providers)}")
        for provider, models in debug_info["provider_breakdown"].items():
            logger.info(f"{provider}: {models}")
        
        logger.info("API Key Status:")
        for key, status in api_keys.items():
            logger.info(f"  {key}: {'✅ Set' if status else '❌ Not set'}")
        
        return debug_info
    
    def test_model_access(self, model_key: str) -> Dict[str, Any]:
        """
        Test if a specific model can be accessed and provide detailed diagnostics
        """
        test_result = {
            "model_requested": model_key,
            "normalized_key": self.normalize_model_key(model_key),
            "available": False,
            "error": None,
            "provider_info": None,
            "suggestions": []
        }
        
        normalized_key = test_result["normalized_key"]
        
        if normalized_key in self.providers:
            test_result["available"] = True
            test_result["provider_info"] = self.get_model_info(model_key)
            logger.info(f"✅ Model '{model_key}' is available as '{normalized_key}'")
        else:
            test_result["error"] = f"Model '{model_key}' (normalized: '{normalized_key}') not found"
            test_result["suggestions"] = [
                self._suggest_similar_model(model_key, list(self.providers.keys()))
            ]
            logger.warning(f"❌ Model '{model_key}' not available")
            logger.info(f"Available models: {list(self.providers.keys())}")
            if test_result["suggestions"][0]:
                logger.info(f"Suggested alternative: {test_result['suggestions'][0]}")
        
        return test_result
    
    def create_pii_extraction_prompt(self, document_type: str = "document") -> str:
        """Create optimized prompt for PII extraction with document classification"""
        
        base_prompt = f"""
Please help me digitize this {document_type} by extracting visible information in a structured JSON format.

Return ONLY a valid JSON object with this exact structure:
{{
  "transcribed_text": "full text you can see in the document",
  "extracted_information": {{
    "names": ["list of person names you see"],
    "contact_info": {{
      "emails": ["email addresses"],
      "phone_numbers": ["phone numbers"],
      "addresses": ["physical addresses"]
    }},
    "dates": ["any dates mentioned"],
    "identification_numbers": ["ID numbers, social insurance numbers, employee IDs, etc"],
    "organizations": ["company names, institutions"],
    "other_relevant_info": ["any other structured data like titles, positions, etc"]
  }},
  "document_classification": {{
    "difficulty_level": "Easy|Medium|Hard",
    "domain": "HR|Finance|Legal|Medical|Government|Education|Other",
    "domain_detail": "specific subdomain or document type"
  }}
}}

Important guidelines:
- Only include information that is clearly visible in the document
- For names: Include full names of people (first + last name)
- For phone numbers: Include area codes and format consistently
- For addresses: Include complete addresses with city/province/postal codes
- For dates: Use consistent format (YYYY-MM-DD when possible)
- Be thorough but accurate - only extract what you can clearly see

For document classification:
- Difficulty levels:
  - Easy: Clear text, standard format, minimal PII
  - Medium: Some formatting challenges, moderate PII density
  - Hard: Poor quality, handwriting, complex layout, or dense PII
- Domain examples:
  - HR: Employment forms, absence requests, pay stubs
  - Finance: Invoices, bank statements, tax documents
  - Legal: Contracts, agreements, legal notices
  - Medical: Medical forms, prescriptions, health records
  - Government: ID documents, permits, official forms
  - Education: Transcripts, enrollment forms, certificates

Return valid JSON only, no other text or explanations
"""
        return base_prompt.strip()
    
    def _check_budget_before_call(
        self,
        model_key: str,
        estimated_input_tokens: Optional[int] = None,
        estimated_output_tokens: Optional[int] = None,
        enforce_strict_limits: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Check budget constraints before making an API call
        
        Args:
            model_key: Model identifier (e.g., "openai/gpt-4o")
            estimated_input_tokens: Estimated input tokens
            estimated_output_tokens: Estimated output tokens
            enforce_strict_limits: Override for strict enforcement
        
        Returns:
            Dict with 'allowed' boolean and budget information
        """
        # If no cost tracker or budget config, allow the call
        if not self.cost_tracker or not self.budget_config:
            return {
                'allowed': True,
                'reason': 'Budget tracking not configured',
                'budget_info': None
            }
        
        # Parse model key
        try:
            provider, model = model_key.split('/', 1)
        except ValueError:
            return {
                'allowed': False,
                'reason': f'Invalid model key format: {model_key}',
                'budget_info': None
            }
        
        # Check emergency stop first
        if self.cost_tracker.check_emergency_stop(provider, self.budget_config):
            return {
                'allowed': False,
                'reason': f'Emergency stop activated for {provider} due to critical budget overrun',
                'budget_info': None
            }
        
        # Determine enforcement mode
        if enforce_strict_limits is None:
            enforce_strict_limits = getattr(self.budget_config, 'strict_budget_enforcement', True)
        
        # Estimate cost for this call
        estimated_cost = self.cost_tracker.estimate_cost_before_call(
            provider=provider,
            model=model,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            safety_margin=getattr(self.budget_config, 'safety_margin_multiplier', 1.1)
        )
        
        # Check if we can afford this call
        budget_check = self.cost_tracker.can_afford(
            provider=provider,
            estimated_cost=estimated_cost,
            budget_config=self.budget_config,
            enforce_limits=enforce_strict_limits
        )
        
        # Return result
        return {
            'allowed': budget_check.can_afford,
            'reason': budget_check.blocking_reason or 'Budget check passed',
            'budget_info': budget_check,
            'warnings': budget_check.warning_messages
        }
    
    def extract_pii_from_text(
        self,
        text: str,
        model_key: str,
        document_type: str = "document",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract PII from text using specified LLM model
        
        Args:
            text: Plain text content
            model_key: Model identifier (e.g., "openai/gpt-4o" or just "gpt-4o-mini")
            document_type: Type of document for prompt optimization
            **kwargs: Additional parameters for the LLM
        
        Returns:
            Dict containing extraction results, usage, and cost information
        """
        start_time = time.time()
        
        # Normalize the model key
        normalized_key = self.normalize_model_key(model_key)
        
        if normalized_key not in self.providers:
            model_info = self.get_model_info(model_key)
            return {
                "success": False,
                "error": f"Model '{model_key}' not available. {model_info.get('error', '')}",
                "suggestion": model_info.get('suggestion'),
                "available_models": model_info.get('available_models', []),
                "processing_time": time.time() - start_time
            }
        
        # Pre-flight budget check
        budget_check = self._check_budget_before_call(
            model_key=normalized_key,
            estimated_input_tokens=len(text.split()) * 1.3,  # Rough estimate
            estimated_output_tokens=kwargs.get('fallback_token_estimate', 500)
        )
        
        if not budget_check['allowed']:
            return {
                "success": False,
                "error": f"Budget limit exceeded: {budget_check['reason']}",
                "budget_check": budget_check,
                "processing_time": time.time() - start_time
            }
        
        provider = self.providers[normalized_key]
        prompt = self.create_pii_extraction_prompt(document_type)
        
        # For text-based extraction, we'll use the provider's text capabilities
        # If provider doesn't support text extraction directly, we'll simulate it
        try:
            if hasattr(provider, 'extract_text_pii'):
                # Use provider's text extraction method if available
                result = provider.extract_text_pii(text, prompt, **kwargs)
            else:
                # Fallback: Create a text prompt for the provider
                text_prompt = f"{prompt}\n\nDocument Text:\n{text}"
                
                # For OpenAI, we can use completion instead of vision
                if isinstance(provider, OpenAIProvider):
                    try:
                        response = provider.client.chat.completions.create(
                            model=provider.model,
                            messages=[
                                {"role": "system", "content": "You are a PII extraction specialist. Extract personally identifiable information from the provided text and return it in JSON format."},
                                {"role": "user", "content": text_prompt}
                            ],
                            max_tokens=2000,
                            temperature=0.1
                        )
                        
                        result = {
                            "success": True,
                            "content": response.choices[0].message.content,
                            "usage": {
                                "input_tokens": response.usage.prompt_tokens,
                                "output_tokens": response.usage.completion_tokens,
                                "total_tokens": response.usage.total_tokens
                            }
                        }
                    except Exception as e:
                        result = {
                            "success": False,
                            "error": f"OpenAI text extraction failed: {str(e)}"
                        }
                else:
                    # For other providers, return a simple success with empty extraction
                    result = {
                        "success": True,
                        "content": '{"pii_entities": [], "confidence_scores": [], "document_summary": "Text-based extraction not fully supported for this provider"}',
                        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                    }
                    
        except Exception as e:
            result = {
                "success": False,
                "error": f"Text PII extraction failed: {str(e)}"
            }
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        # Add budget warnings if available
        if budget_check.get('warnings'):
            result["budget_warnings"] = budget_check['warnings']
        
        if result["success"]:
            # Parse JSON response similar to image extraction
            try:
                content = result["content"]
                
                # Clean up the content to extract JSON
                if "```json" in content:
                    json_part = content.split("```json")[1].split("```")[0]
                elif "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_part = content[start:end]
                else:
                    json_part = content
                
                parsed_data = json.loads(json_part.strip())
                
                # Standardize the response format
                pii_entities = parsed_data.get("pii_entities", [])
                confidence_scores = parsed_data.get("confidence_scores", [])
                
                # Calculate usage costs
                usage = result.get("usage", {})
                cost_per_token = provider.get_cost_per_token()
                estimated_cost = (
                    usage.get("input_tokens", 0) * cost_per_token.get("input", 0) / 1000 +
                    usage.get("output_tokens", 0) * cost_per_token.get("output", 0) / 1000
                )
                
                result.update({
                    "pii_entities": pii_entities,
                    "confidence_scores": confidence_scores,
                    "parsed_data": parsed_data,
                    "usage": {
                        **usage,
                        "estimated_cost": estimated_cost
                    }
                })
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                result.update({
                    "pii_entities": [],
                    "confidence_scores": [],
                    "parse_error": str(e),
                    "raw_content": result.get("content", "")
                })
                
        return result

    def extract_pii_from_image(
        self, 
        image_data: str, 
        model_key: str, 
        document_type: str = "document",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract PII from image using specified LLM model
        
        Args:
            image_data: Base64 encoded image
            model_key: Model identifier (e.g., "openai/gpt-4o" or just "gpt-4o-mini")
            document_type: Type of document for prompt optimization
            **kwargs: Additional parameters for the LLM
        
        Returns:
            Dict containing extraction results, usage, and cost information
        """
        start_time = time.time()
        
        # Normalize the model key
        normalized_key = self.normalize_model_key(model_key)
        
        if normalized_key not in self.providers:
            model_info = self.get_model_info(model_key)
            return {
                "success": False,
                "error": f"Model '{model_key}' not available. {model_info.get('error', '')}",
                "suggestion": model_info.get('suggestion'),
                "available_models": model_info.get('available_models', []),
                "processing_time": time.time() - start_time
            }
        
        # Pre-flight budget check
        budget_check = self._check_budget_before_call(
            model_key=normalized_key,
            estimated_input_tokens=kwargs.get('estimated_input_tokens', 1000),  # Images typically use more tokens
            estimated_output_tokens=kwargs.get('estimated_output_tokens', 500)
        )
        
        if not budget_check['allowed']:
            return {
                "success": False,
                "error": f"Budget limit exceeded: {budget_check['reason']}",
                "budget_check": budget_check,
                "processing_time": time.time() - start_time
            }
        
        provider = self.providers[normalized_key]
        prompt = self.create_pii_extraction_prompt(document_type)
        
        # Extract with LLM
        result = provider.extract_pii(image_data, prompt, **kwargs)
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        # Add budget warnings if available
        if budget_check.get('warnings'):
            result["budget_warnings"] = budget_check['warnings']
        
        if result["success"]:
            # Try to parse the JSON response
            try:
                content = result["content"]
                
                # Clean up the content to extract JSON
                if "```json" in content:
                    json_part = content.split("```json")[1].split("```")[0]
                elif "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    json_part = content[start:end]
                else:
                    json_part = content
                
                parsed_data = json.loads(json_part)
                result["structured_data"] = parsed_data
                result["extraction_method"] = "pure_llm_structured"
                
                # Extract PII entities in consistent format
                extracted_info = parsed_data.get("extracted_information", {})
                pii_entities = []
                
                # Process names
                for name in extracted_info.get("names", []):
                    pii_entities.append({
                        "type": "PERSON",
                        "text": name,
                        "confidence": 0.95,  # LLM extractions are generally high confidence
                        "source": "llm_extraction"
                    })
                
                # Process contact info
                contact_info = extracted_info.get("contact_info", {})
                for email in contact_info.get("emails", []):
                    pii_entities.append({
                        "type": "EMAIL",
                        "text": email,
                        "confidence": 0.98,
                        "source": "llm_extraction"
                    })
                
                for phone in contact_info.get("phone_numbers", []):
                    pii_entities.append({
                        "type": "PHONE",
                        "text": phone,
                        "confidence": 0.95,
                        "source": "llm_extraction"
                    })
                
                for address in contact_info.get("addresses", []):
                    pii_entities.append({
                        "type": "ADDRESS",
                        "text": address,
                        "confidence": 0.90,
                        "source": "llm_extraction"
                    })
                
                # Process dates
                for date in extracted_info.get("dates", []):
                    pii_entities.append({
                        "type": "DATE",
                        "text": date,
                        "confidence": 0.85,
                        "source": "llm_extraction"
                    })
                
                # Process identification numbers
                for id_num in extracted_info.get("identification_numbers", []):
                    pii_entities.append({
                        "type": "ID_NUMBER",
                        "text": id_num,
                        "confidence": 0.90,
                        "source": "llm_extraction"
                    })
                
                # Process organizations
                for org in extracted_info.get("organizations", []):
                    pii_entities.append({
                        "type": "ORGANIZATION",
                        "text": org,
                        "confidence": 0.85,
                        "source": "llm_extraction"
                    })
                
                result["pii_entities"] = pii_entities
                result["total_entities"] = len(pii_entities)
                result["transcribed_text"] = parsed_data.get("transcribed_text", "")
                
            except json.JSONDecodeError as e:
                # Fallback: treat as unstructured text
                result["extraction_method"] = "pure_llm_unstructured"
                result["transcribed_text"] = result["content"]
                result["structured_data"] = None
                result["pii_entities"] = []
                result["parsing_error"] = str(e)
        
        return result
    
    def batch_extract_pii(
        self, 
        image_data_list: List[str], 
        model_key: str, 
        document_types: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract PII from multiple images
        
        Args:
            image_data_list: List of base64 encoded images
            model_key: Model identifier
            document_types: List of document types (optional)
            progress_callback: Function to call with progress updates
        
        Returns:
            List of extraction results
        """
        results = []
        total_cost = 0.0
        
        for i, image_data in enumerate(image_data_list):
            doc_type = document_types[i] if document_types and i < len(document_types) else "document"
            
            result = self.extract_pii_from_image(image_data, model_key, doc_type)
            results.append(result)
            
            if result.get("success") and "usage" in result:
                total_cost += result["usage"].get("estimated_cost", 0)
            
            if progress_callback:
                progress_callback(i + 1, len(image_data_list), total_cost)
        
        return results
    
    def compare_models(
        self, 
        image_data: str, 
        model_keys: List[str], 
        document_type: str = "document"
    ) -> Dict[str, Any]:
        """
        Compare multiple models on the same image
        
        Args:
            image_data: Base64 encoded image
            model_keys: List of model identifiers to compare
            document_type: Type of document
        
        Returns:
            Comparison results with performance metrics
        """
        results = {}
        
        for model_key in model_keys:
            if model_key in self.providers:
                result = self.extract_pii_from_image(image_data, model_key, document_type)
                results[model_key] = result
        
        # Calculate comparison metrics
        comparison = {
            "models_compared": list(results.keys()),
            "individual_results": results,
            "summary": {
                "fastest_model": min(results.keys(), 
                                   key=lambda k: results[k].get("processing_time", float('inf'))),
                "most_entities": max(results.keys(), 
                                   key=lambda k: results[k].get("total_entities", 0)),
                "lowest_cost": min(results.keys(), 
                                 key=lambda k: results[k].get("usage", {}).get("estimated_cost", float('inf'))),
                "total_cost": sum(r.get("usage", {}).get("estimated_cost", 0) for r in results.values())
            }
        }
        
        return comparison

# Global instance with error handling
try:
    # Import cost tracker and config
    from .cost_tracker import default_cost_tracker
    try:
        from ..core.config import settings
        budget_config = settings.budget
    except ImportError:
        budget_config = None
        logger.warning("Budget configuration not available")
    
    llm_service = MultimodalLLMService(
        cost_tracker=default_cost_tracker,
        budget_config=budget_config
    )
    logger.info(f"LLM service initialized with {len(llm_service.get_available_models())} available models")
    logger.info(f"Budget enforcement: {'enabled' if budget_config else 'disabled'}")
except Exception as e:
    logger.error(f"Failed to initialize LLM service: {e}")
    # Create a dummy service that shows errors
    class DummyLLMService:
        def get_available_models(self):
            return []
        def get_model_info(self, model_key):
            return {"available": False, "error": "LLM service initialization failed"}
        def extract_pii_from_image(self, *args, **kwargs):
            return {"success": False, "error": f"LLM service not available: {e}"}
        def extract_pii_from_text(self, *args, **kwargs):
            return {"success": False, "error": f"LLM service not available: {e}"}
    
    llm_service = DummyLLMService()