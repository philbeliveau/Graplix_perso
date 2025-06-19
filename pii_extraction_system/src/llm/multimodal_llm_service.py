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

class MultimodalLLMService:
    """Main service for multimodal LLM operations"""
    
    def __init__(self):
        self.providers = {}
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
        
        logger.info(f"LLM service initialized with {len(self.providers)} available models: {list(self.providers.keys())}")
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.providers.keys())
    
    def get_model_info(self, model_key: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_key not in self.providers:
            return {"available": False}
        
        provider = self.providers[model_key]
        cost_info = provider.get_cost_per_token()
        
        return {
            "available": True,
            "provider": model_key.split('/')[0],
            "model": model_key.split('/')[1],
            "cost_per_1k_input_tokens": cost_info['input'] * 1000,
            "cost_per_1k_output_tokens": cost_info['output'] * 1000,
            "supports_images": True,
            "supports_json": True
        }
    
    def create_pii_extraction_prompt(self, document_type: str = "document") -> str:
        """Create optimized prompt for PII extraction"""
        
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
  }}
}}

Important guidelines:
- Only include information that is clearly visible in the document
- For names: Include full names of people (first + last name)
- For phone numbers: Include area codes and format consistently
- For addresses: Include complete addresses with city/province/postal codes
- For dates: Use consistent format (YYYY-MM-DD when possible)
- Be thorough but accurate - only extract what you can clearly see
- Return valid JSON only, no other text or explanations
"""
        return base_prompt.strip()
    
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
            model_key: Model identifier (e.g., "openai/gpt-4o")
            document_type: Type of document for prompt optimization
            **kwargs: Additional parameters for the LLM
        
        Returns:
            Dict containing extraction results, usage, and cost information
        """
        start_time = time.time()
        
        if model_key not in self.providers:
            return {
                "success": False,
                "error": f"Model {model_key} not available",
                "processing_time": 0
            }
        
        provider = self.providers[model_key]
        prompt = self.create_pii_extraction_prompt(document_type)
        
        # Extract with LLM
        result = provider.extract_pii(image_data, prompt, **kwargs)
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
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
    llm_service = MultimodalLLMService()
    logger.info(f"LLM service initialized with {len(llm_service.get_available_models())} available models")
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
    
    llm_service = DummyLLMService()