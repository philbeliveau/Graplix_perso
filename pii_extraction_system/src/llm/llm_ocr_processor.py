"""LLM-based OCR processor for enhanced text extraction."""

import os
import base64
import json
import time
import sys
from io import BytesIO
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Load environment variables
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
env_loader_path = project_root / "load_env.py"

if env_loader_path.exists():
    sys.path.insert(0, str(project_root))
    try:
        from load_env import load_env_file
        load_env_file()
    except ImportError:
        pass

import cv2
import numpy as np
from PIL import Image
import openai
import anthropic
import google.generativeai as genai
import requests

from core.config import settings
from core.logging_config import get_logger
from .llm_config import (
    LLMModel, LLMModelRegistry, LLMProvider, OCRTaskType, 
    CostCalculator, llm_config, cost_tracker
)

logger = get_logger(__name__)


class LLMOCRProcessor:
    """LLM-based OCR processor supporting multiple providers."""
    
    def __init__(self):
        """Initialize LLM OCR processor."""
        self.clients = {}
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize API clients for different providers."""
        try:
            # OpenAI
            if os.getenv("OPENAI_API_KEY"):
                self.clients[LLMProvider.OPENAI] = openai.OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                logger.info("OpenAI client initialized")
            
            # Anthropic
            if os.getenv("ANTHROPIC_API_KEY"):
                self.clients[LLMProvider.ANTHROPIC] = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                logger.info("Anthropic client initialized")
            
            # Google
            if os.getenv("GOOGLE_API_KEY"):
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.clients[LLMProvider.GOOGLE] = genai
                logger.info("Google Gemini client initialized")
            
            # DeepSeek
            if os.getenv("DEEPSEEK_API"):
                self.clients[LLMProvider.DEEPSEEK] = {
                    "api_key": os.getenv("DEEPSEEK_API"),
                    "base_url": "https://api.deepseek.com/v1"
                }
                logger.info("DeepSeek client initialized")
            
            # NVIDIA
            if os.getenv("NVIDIA_KEY"):
                self.clients[LLMProvider.NVIDIA] = {
                    "api_key": os.getenv("NVIDIA_KEY"),
                    "base_url": "https://integrate.api.nvidia.com/v1"
                }
                logger.info("NVIDIA client initialized")
                
        except Exception as e:
            logger.error(f"Error initializing LLM clients: {e}")
    
    def process_image_with_llm(
        self,
        image: Union[np.ndarray, str, Path],
        model_name: str = "gpt-4o-mini",
        task_type: OCRTaskType = OCRTaskType.BASIC_TEXT_EXTRACTION,
        custom_prompt: Optional[str] = None,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """Process image using LLM for OCR."""
        
        try:
            # Get model configuration
            model = LLMModelRegistry.get_model(model_name)
            if not model:
                raise ValueError(f"Unknown model: {model_name}")
            
            if not model.supports_vision:
                raise ValueError(f"Model {model_name} doesn't support vision tasks")
            
            # Prepare image
            image_data = self._prepare_image(image)
            
            # Estimate costs
            input_tokens = CostCalculator.estimate_tokens_for_image(
                image_data["width"], image_data["height"], True
            )
            output_tokens = CostCalculator.estimate_output_tokens(task_type, input_tokens)
            estimated_cost = CostCalculator.calculate_cost(model, input_tokens, output_tokens)
            
            # Check cost limits
            if estimated_cost > llm_config.max_cost_per_document:
                logger.warning(f"Estimated cost ${estimated_cost:.4f} exceeds limit ${llm_config.max_cost_per_document}")
                if llm_config.enable_cost_optimization:
                    # Try cheaper alternative
                    cheaper_models = LLMModelRegistry.get_cheapest_models(vision_only=True)
                    for cheaper_model in cheaper_models:
                        cheaper_cost = CostCalculator.calculate_cost(cheaper_model, input_tokens, output_tokens)
                        if cheaper_cost <= llm_config.max_cost_per_document:
                            logger.info(f"Switching to cheaper model: {cheaper_model.model_name}")
                            model = cheaper_model
                            model_name = cheaper_model.model_name
                            estimated_cost = cheaper_cost
                            break
            
            # Generate prompt
            prompt = custom_prompt or self._generate_prompt(task_type)
            
            # Process with appropriate provider
            start_time = time.time()
            
            if model.provider == LLMProvider.OPENAI:
                result = self._process_with_openai(image_data, prompt, model_name, include_confidence)
            elif model.provider == LLMProvider.ANTHROPIC:
                result = self._process_with_anthropic(image_data, prompt, model_name, include_confidence)
            elif model.provider == LLMProvider.GOOGLE:
                result = self._process_with_google(image_data, prompt, model_name, include_confidence)
            else:
                raise ValueError(f"Provider {model.provider} not supported for vision tasks")
            
            processing_time = time.time() - start_time
            
            # Record usage for cost tracking
            actual_input_tokens = result.get("usage", {}).get("input_tokens", input_tokens)
            actual_output_tokens = result.get("usage", {}).get("output_tokens", output_tokens)
            actual_cost = CostCalculator.calculate_cost(model, actual_input_tokens, actual_output_tokens)
            
            cost_tracker.record_usage(
                model_name=model_name,
                input_tokens=actual_input_tokens,
                output_tokens=actual_output_tokens,
                cost=actual_cost,
                task_type=task_type.value
            )
            
            # Format result
            return {
                "text": result.get("text", ""),
                "structured_data": result.get("structured_data", {}),
                "confidence": result.get("confidence", 0.8),
                "bounding_boxes": result.get("bounding_boxes", []),
                "metadata": {
                    "model_used": model_name,
                    "provider": model.provider.value,
                    "task_type": task_type.value,
                    "processing_time": round(processing_time, 2),
                    "cost_info": {
                        "estimated_cost": round(estimated_cost, 6),
                        "actual_cost": round(actual_cost, 6),
                        "input_tokens": actual_input_tokens,
                        "output_tokens": actual_output_tokens
                    },
                    "image_info": {
                        "width": image_data["width"],
                        "height": image_data["height"],
                        "format": image_data["format"]
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"LLM OCR processing failed: {e}")
            raise
    
    def _prepare_image(self, image: Union[np.ndarray, str, Path]) -> Dict[str, Any]:
        """Prepare image for LLM processing."""
        try:
            if isinstance(image, (str, Path)):
                # Load from file
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                # Convert from OpenCV format
                if len(image.shape) == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                pil_image = Image.fromarray(image_rgb)
            else:
                raise ValueError("Unsupported image format")
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Optimize image size for API limits while preserving quality
            max_size = 2048  # Max dimension for most LLM APIs
            if max(pil_image.size) > max_size:
                # Use high-quality resampling to preserve text clarity
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Enhance image for better OCR (optional preprocessing)
            # Convert to high contrast if image is low quality
            if hasattr(pil_image, 'mode') and pil_image.mode in ['L', 'P']:
                pil_image = pil_image.convert('RGB')
            
            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG', optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "base64": image_base64,
                "width": pil_image.size[0],
                "height": pil_image.size[1],
                "format": "PNG"
            }
            
        except Exception as e:
            logger.error(f"Image preparation failed: {e}")
            raise
    
    def _generate_prompt(self, task_type: OCRTaskType) -> str:
        """Generate appropriate prompt based on task type."""
        prompts = {
            OCRTaskType.BASIC_TEXT_EXTRACTION: """
You are an expert OCR system. Extract ALL text from this image with perfect accuracy. 

CRITICAL REQUIREMENTS:
- Extract text EXACTLY as written, character by character
- Preserve ALL accents, special characters, and diacritics (é, è, à, ç, etc.)
- Maintain original spacing, line breaks, and formatting
- Handle multiple languages (French, English, etc.) correctly
- Pay special attention to French text and proper accent marks
- If text is unclear, use [UNCLEAR] but try your best first
- Do NOT interpret or translate - extract literally
- Return ONLY the extracted text, no explanations

Text to extract:""",
            OCRTaskType.STRUCTURED_DATA_EXTRACTION: """
Extract all text and structured data from this image. Identify and extract:
- Headers and titles
- Key-value pairs
- Lists and bullet points
- Tables (if any)
- Form fields and their values

Return the information in a structured JSON format that preserves the document's organization.
""",
            OCRTaskType.HANDWRITING_RECOGNITION: """
This image contains handwritten text. Please transcribe all handwritten content as accurately as possible.
Pay special attention to:
- Cursive writing
- Mixed print and cursive
- Numbers and special characters
- Crossed-out or corrected text

Indicate your confidence level for unclear sections.
""",
            OCRTaskType.TABLE_EXTRACTION: """
This image contains one or more tables. Please extract the table data and return it in a structured format.
For each table:
- Identify headers and column names
- Extract all row data
- Preserve cell relationships
- Handle merged cells appropriately

Return the data in CSV or JSON format that maintains the table structure.
""",
            OCRTaskType.DOCUMENT_ANALYSIS: """
Analyze this document image and provide a comprehensive extraction including:
- Document type and layout
- All text content with hierarchical structure
- Key information and data points
- Any forms, fields, or structured elements
- Visual elements that provide context

Provide both the raw text and a structured analysis of the document's content and purpose.
"""
        }
        
        return prompts.get(task_type, prompts[OCRTaskType.BASIC_TEXT_EXTRACTION])
    
    def _process_with_openai(self, image_data: Dict, prompt: str, model_name: str, include_confidence: bool) -> Dict:
        """Process image with OpenAI models."""
        try:
            client = self.clients[LLMProvider.OPENAI]
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data['base64']}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.0
            )
            
            text_content = response.choices[0].message.content
            
            # Try to parse structured data if JSON format is returned
            structured_data = {}
            try:
                if text_content.strip().startswith('{') or text_content.strip().startswith('['):
                    structured_data = json.loads(text_content)
            except json.JSONDecodeError:
                pass
            
            return {
                "text": text_content,
                "structured_data": structured_data,
                "confidence": 0.85,  # Default confidence for OpenAI
                "usage": {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI processing failed: {e}")
            raise
    
    def _process_with_anthropic(self, image_data: Dict, prompt: str, model_name: str, include_confidence: bool) -> Dict:
        """Process image with Anthropic Claude models."""
        try:
            client = self.clients[LLMProvider.ANTHROPIC]
            
            response = client.messages.create(
                model=model_name,
                max_tokens=4000,
                temperature=0.0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_data['base64']
                                }
                            },
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            
            text_content = response.content[0].text
            
            # Try to parse structured data
            structured_data = {}
            try:
                if text_content.strip().startswith('{') or text_content.strip().startswith('['):
                    structured_data = json.loads(text_content)
            except json.JSONDecodeError:
                pass
            
            return {
                "text": text_content,
                "structured_data": structured_data,
                "confidence": 0.88,  # Default confidence for Claude
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Anthropic processing failed: {e}")
            raise
    
    def _process_with_google(self, image_data: Dict, prompt: str, model_name: str, include_confidence: bool) -> Dict:
        """Process image with Google Gemini models."""
        try:
            # Convert base64 back to PIL Image for Gemini
            image_bytes = base64.b64decode(image_data['base64'])
            pil_image = Image.open(BytesIO(image_bytes))
            
            model = genai.GenerativeModel(model_name)
            response = model.generate_content([prompt, pil_image])
            
            text_content = response.text
            
            # Try to parse structured data
            structured_data = {}
            try:
                if text_content.strip().startswith('{') or text_content.strip().startswith('['):
                    structured_data = json.loads(text_content)
            except json.JSONDecodeError:
                pass
            
            # Estimate token usage (Gemini doesn't always provide exact counts)
            estimated_input = CostCalculator.estimate_tokens_for_image(
                image_data["width"], image_data["height"], True
            )
            estimated_output = len(text_content.split()) * 1.3  # Rough estimation
            
            return {
                "text": text_content,
                "structured_data": structured_data,
                "confidence": 0.83,  # Default confidence for Gemini
                "usage": {
                    "input_tokens": int(estimated_input),
                    "output_tokens": int(estimated_output)
                }
            }
            
        except Exception as e:
            logger.error(f"Google processing failed: {e}")
            raise
    
    def get_best_model_for_task(
        self, 
        task_type: OCRTaskType, 
        prioritize_cost: bool = True,
        max_cost: Optional[float] = None
    ) -> str:
        """Get the best model recommendation for a specific task."""
        
        # Get available vision models
        vision_models = LLMModelRegistry.get_vision_models()
        
        # Filter by available clients
        available_models = [
            model for model in vision_models 
            if model.provider in self.clients
        ]
        
        if not available_models:
            raise ValueError("No LLM clients available for vision tasks")
        
        # Apply cost filter if specified
        if max_cost:
            available_models = [
                model for model in available_models
                if model.input_cost_per_1k_tokens <= max_cost / 1000
            ]
        
        if not available_models:
            raise ValueError(f"No models available within cost limit ${max_cost}")
        
        # Task-specific recommendations (updated to remove deprecated models)
        task_preferences = {
            OCRTaskType.BASIC_TEXT_EXTRACTION: ["gpt-4o-mini", "claude-3-haiku", "gemini-1.5-flash"],
            OCRTaskType.STRUCTURED_DATA_EXTRACTION: ["gpt-4o-mini", "claude-3-sonnet", "gemini-1.5-pro"],
            OCRTaskType.HANDWRITING_RECOGNITION: ["gpt-4o-mini", "claude-3-sonnet"],
            OCRTaskType.TABLE_EXTRACTION: ["gpt-4o-mini", "claude-3-sonnet", "gemini-1.5-pro"],
            OCRTaskType.DOCUMENT_ANALYSIS: ["gpt-4o-mini", "claude-3-sonnet"]
        }
        
        preferred_models = task_preferences.get(task_type, ["gpt-4o-mini"])
        
        # Find best available model
        for preferred in preferred_models:
            for model in available_models:
                if model.model_name == preferred:
                    return preferred
        
        # Fallback: sort by cost or quality
        if prioritize_cost:
            available_models.sort(key=lambda m: m.input_cost_per_1k_tokens)
        else:
            available_models.sort(key=lambda m: m.quality_score, reverse=True)
        
        return available_models[0].model_name
    
    def process_with_fallback(
        self,
        image: Union[np.ndarray, str, Path],
        task_type: OCRTaskType = OCRTaskType.BASIC_TEXT_EXTRACTION,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process image with automatic fallback to cheaper/alternative models."""
        
        # Get primary model recommendation
        try:
            primary_model = self.get_best_model_for_task(task_type, prioritize_cost=llm_config.prefer_cheaper_models)
        except ValueError as e:
            logger.error(f"No suitable models available: {e}")
            raise
        
        # Try primary model
        for attempt in range(llm_config.max_retry_attempts):
            try:
                result = self.process_image_with_llm(
                    image=image,
                    model_name=primary_model,
                    task_type=task_type,
                    custom_prompt=custom_prompt
                )
                
                # Check confidence threshold
                if result["confidence"] >= llm_config.min_confidence_threshold:
                    return result
                else:
                    logger.warning(f"Low confidence result from {primary_model}: {result['confidence']}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed with {primary_model}: {e}")
                
                if attempt < llm_config.max_retry_attempts - 1:
                    # Try fallback model
                    if llm_config.fallback_model and llm_config.fallback_model != primary_model:
                        try:
                            result = self.process_image_with_llm(
                                image=image,
                                model_name=llm_config.fallback_model,
                                task_type=task_type,
                                custom_prompt=custom_prompt
                            )
                            return result
                        except Exception as fallback_error:
                            logger.warning(f"Fallback model also failed: {fallback_error}")
                    
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"All attempts failed for LLM OCR processing")


# Global processor instance
llm_ocr_processor = LLMOCRProcessor()