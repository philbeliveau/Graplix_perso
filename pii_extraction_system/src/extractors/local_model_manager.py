"""
Local Model Manager for Vision-LLM PII Extraction

This module provides support for local vision models including fallback mechanisms,
local inference capabilities, and hybrid cloud-local processing strategies.
"""

import os
import json
import time
import logging
import subprocess
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile
import shutil

logger = logging.getLogger(__name__)


class LocalModelType(Enum):
    """Types of local models supported"""
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers" 
    LLAMACPP = "llamacpp"
    ONNX = "onnx"
    CUSTOM = "custom"


class ModelSize(Enum):
    """Model size categories"""
    TINY = "tiny"      # < 1GB
    SMALL = "small"    # 1-4GB
    MEDIUM = "medium"  # 4-8GB
    LARGE = "large"    # 8-16GB
    XLARGE = "xlarge"  # > 16GB


@dataclass
class LocalModelInfo:
    """Information about a local model"""
    name: str
    model_type: LocalModelType
    size: ModelSize
    capabilities: List[str]
    file_path: Optional[str] = None
    config_path: Optional[str] = None
    is_available: bool = False
    is_loaded: bool = False
    memory_usage: int = 0  # MB
    load_time: float = 0.0  # seconds
    inference_time: float = 0.0  # average seconds per request
    accuracy_score: float = 0.0  # 0-1
    supported_formats: List[str] = field(default_factory=lambda: ['png', 'jpg', 'jpeg', 'pdf'])
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalInferenceResult:
    """Result from local model inference"""
    success: bool
    content: str
    confidence: float
    processing_time: float
    model_used: str
    memory_usage: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalModelManager:
    """
    Manager for local vision models with support for multiple inference engines,
    automatic fallback mechanisms, and hybrid processing strategies.
    """
    
    def __init__(self,
                 models_dir: Optional[str] = None,
                 enable_gpu: bool = True,
                 max_memory_usage: int = 8192,  # MB
                 enable_auto_download: bool = False):
        """
        Initialize Local Model Manager
        
        Args:
            models_dir: Directory for storing local models
            enable_gpu: Enable GPU acceleration if available
            max_memory_usage: Maximum memory usage in MB
            enable_auto_download: Enable automatic model downloading
        """
        self.models_dir = Path(models_dir or os.path.expanduser("~/.cache/pii_extraction/models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_gpu = enable_gpu
        self.max_memory_usage = max_memory_usage
        self.enable_auto_download = enable_auto_download
        
        # Available models registry
        self.available_models: Dict[str, LocalModelInfo] = {}
        self.loaded_models: Dict[str, Any] = {}
        
        # Model configurations
        self.model_configs = {
            "llava-7b": {
                "type": LocalModelType.OLLAMA,
                "size": ModelSize.MEDIUM,
                "capabilities": ["vision", "text", "pii_extraction"],
                "download_url": "ollama pull llava:7b",
                "memory_req": 4096
            },
            "llava-13b": {
                "type": LocalModelType.OLLAMA,
                "size": ModelSize.LARGE,
                "capabilities": ["vision", "text", "pii_extraction"],
                "download_url": "ollama pull llava:13b",
                "memory_req": 8192
            },
            "moondream2": {
                "type": LocalModelType.TRANSFORMERS,
                "size": ModelSize.SMALL,
                "capabilities": ["vision", "text"],
                "model_id": "vikhyatk/moondream2",
                "memory_req": 2048
            },
            "instructblip": {
                "type": LocalModelType.TRANSFORMERS,
                "size": ModelSize.MEDIUM,
                "capabilities": ["vision", "text", "instruction_following"],
                "model_id": "Salesforce/instructblip-vicuna-7b",
                "memory_req": 4096
            }
        }
        
        # Performance tracking
        self.performance_stats = {}
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"LocalModelManager initialized with {len(self.available_models)} models")
    
    def _initialize_models(self):
        """Initialize and discover available local models"""
        
        # Check for Ollama models
        self._discover_ollama_models()
        
        # Check for Transformers models
        self._discover_transformers_models()
        
        # Check for custom models
        self._discover_custom_models()
        
        logger.info(f"Discovered {len(self.available_models)} local models")
    
    def _discover_ollama_models(self):
        """Discover available Ollama models"""
        try:
            # Check if Ollama is installed
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 1:
                            model_name = parts[0]
                            
                            # Check if it's a vision model
                            if any(keyword in model_name.lower() for keyword in ['llava', 'vision', 'clip']):
                                self.available_models[f"ollama/{model_name}"] = LocalModelInfo(
                                    name=f"ollama/{model_name}",
                                    model_type=LocalModelType.OLLAMA,
                                    size=self._estimate_model_size(model_name),
                                    capabilities=["vision", "text", "pii_extraction"],
                                    is_available=True,
                                    metadata={"ollama_model": model_name}
                                )
                                
                logger.info(f"Found {len([m for m in self.available_models if m.startswith('ollama/')])} Ollama vision models")
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("Ollama not available or no models found")
    
    def _discover_transformers_models(self):
        """Discover available Transformers models"""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            # Check for pre-configured models
            for model_name, config in self.model_configs.items():
                if config["type"] == LocalModelType.TRANSFORMERS:
                    try:
                        # Try to load processor to check if model is available
                        processor = AutoProcessor.from_pretrained(
                            config["model_id"],
                            cache_dir=str(self.models_dir / "transformers")
                        )
                        
                        self.available_models[model_name] = LocalModelInfo(
                            name=model_name,
                            model_type=LocalModelType.TRANSFORMERS,
                            size=config["size"],
                            capabilities=config["capabilities"],
                            is_available=True,
                            metadata={
                                "model_id": config["model_id"],
                                "memory_req": config["memory_req"]
                            }
                        )
                        
                    except Exception as e:
                        logger.debug(f"Transformers model {model_name} not available: {e}")
                        
            logger.info(f"Found {len([m for m in self.available_models if self.available_models[m].model_type == LocalModelType.TRANSFORMERS])} Transformers models")
            
        except ImportError:
            logger.debug("Transformers library not available")
    
    def _discover_custom_models(self):
        """Discover custom local models"""
        custom_models_dir = self.models_dir / "custom"
        
        if custom_models_dir.exists():
            for model_path in custom_models_dir.iterdir():
                if model_path.is_dir():
                    config_file = model_path / "config.json"
                    
                    if config_file.exists():
                        try:
                            with open(config_file, 'r') as f:
                                config = json.load(f)
                            
                            self.available_models[f"custom/{model_path.name}"] = LocalModelInfo(
                                name=f"custom/{model_path.name}",
                                model_type=LocalModelType.CUSTOM,
                                size=ModelSize(config.get("size", "medium")),
                                capabilities=config.get("capabilities", ["vision", "text"]),
                                file_path=str(model_path),
                                config_path=str(config_file),
                                is_available=True,
                                metadata=config
                            )
                            
                        except Exception as e:
                            logger.warning(f"Failed to load custom model {model_path.name}: {e}")
    
    def _estimate_model_size(self, model_name: str) -> ModelSize:
        """Estimate model size based on name"""
        name_lower = model_name.lower()
        
        if any(size in name_lower for size in ['1b', '3b', 'tiny', 'mini']):
            return ModelSize.SMALL
        elif any(size in name_lower for size in ['7b', 'small']):
            return ModelSize.MEDIUM
        elif any(size in name_lower for size in ['13b', '15b', 'large']):
            return ModelSize.LARGE
        elif any(size in name_lower for size in ['30b', '65b', 'xl', 'xxl']):
            return ModelSize.XLARGE
        else:
            return ModelSize.MEDIUM
    
    def get_available_models(self) -> List[str]:
        """Get list of available local models"""
        return list(self.available_models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[LocalModelInfo]:
        """Get information about a specific model"""
        return self.available_models.get(model_name)
    
    def load_model(self, model_name: str) -> bool:
        """Load a specific model into memory"""
        if model_name not in self.available_models:
            logger.error(f"Model {model_name} not available")
            return False
        
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        model_info = self.available_models[model_name]
        
        try:
            start_time = time.time()
            
            if model_info.model_type == LocalModelType.OLLAMA:
                # Ollama models are loaded on-demand
                self.loaded_models[model_name] = {"type": "ollama", "ready": True}
                
            elif model_info.model_type == LocalModelType.TRANSFORMERS:
                model_data = self._load_transformers_model(model_info)
                if model_data:
                    self.loaded_models[model_name] = model_data
                else:
                    return False
                    
            elif model_info.model_type == LocalModelType.CUSTOM:
                model_data = self._load_custom_model(model_info)
                if model_data:
                    self.loaded_models[model_name] = model_data
                else:
                    return False
            
            load_time = time.time() - start_time
            model_info.load_time = load_time
            model_info.is_loaded = True
            
            logger.info(f"Loaded model {model_name} in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def _load_transformers_model(self, model_info: LocalModelInfo) -> Optional[Dict[str, Any]]:
        """Load a Transformers model"""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForVision2Seq
            from PIL import Image
            
            model_id = model_info.metadata.get("model_id")
            if not model_id:
                return None
            
            # Load processor and model
            processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=str(self.models_dir / "transformers")
            )
            
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                cache_dir=str(self.models_dir / "transformers"),
                torch_dtype=torch.float16 if self.enable_gpu else torch.float32,
                device_map="auto" if self.enable_gpu else None
            )
            
            return {
                "type": "transformers",
                "processor": processor,
                "model": model,
                "ready": True
            }
            
        except Exception as e:
            logger.error(f"Failed to load Transformers model: {e}")
            return None
    
    def _load_custom_model(self, model_info: LocalModelInfo) -> Optional[Dict[str, Any]]:
        """Load a custom model"""
        # Custom model loading logic would go here
        # This is a placeholder for extensibility
        logger.warning(f"Custom model loading not implemented for {model_info.name}")
        return None
    
    def infer(self, 
             model_name: str,
             image_data: str,
             prompt: str,
             max_tokens: int = 1000,
             temperature: float = 0.1) -> LocalInferenceResult:
        """
        Perform inference using a local model
        
        Args:
            model_name: Name of the model to use
            image_data: Base64 encoded image data
            prompt: Text prompt for the model
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            LocalInferenceResult with the inference output
        """
        start_time = time.time()
        
        if model_name not in self.available_models:
            return LocalInferenceResult(
                success=False,
                content="",
                confidence=0.0,
                processing_time=0.0,
                model_used=model_name,
                error=f"Model {model_name} not available"
            )
        
        # Load model if not already loaded
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                return LocalInferenceResult(
                    success=False,
                    content="",
                    confidence=0.0,
                    processing_time=time.time() - start_time,
                    model_used=model_name,
                    error=f"Failed to load model {model_name}"
                )
        
        model_info = self.available_models[model_name]
        
        try:
            if model_info.model_type == LocalModelType.OLLAMA:
                result = self._infer_ollama(model_name, image_data, prompt, max_tokens, temperature)
            elif model_info.model_type == LocalModelType.TRANSFORMERS:
                result = self._infer_transformers(model_name, image_data, prompt, max_tokens, temperature)
            elif model_info.model_type == LocalModelType.CUSTOM:
                result = self._infer_custom(model_name, image_data, prompt, max_tokens, temperature)
            else:
                raise ValueError(f"Unsupported model type: {model_info.model_type}")
            
            result.processing_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance_stats(model_name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {e}")
            return LocalInferenceResult(
                success=False,
                content="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=model_name,
                error=str(e)
            )
    
    def _infer_ollama(self, 
                     model_name: str,
                     image_data: str,
                     prompt: str,
                     max_tokens: int,
                     temperature: float) -> LocalInferenceResult:
        """Perform inference using Ollama"""
        try:
            import base64
            import requests
            import tempfile
            
            # Save image to temporary file
            image_bytes = base64.b64decode(image_data)
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_file.flush()
                
                # Prepare Ollama request
                ollama_model = self.available_models[model_name].metadata.get("ollama_model", model_name.split('/')[-1])
                
                # Use Ollama API
                payload = {
                    "model": ollama_model,
                    "prompt": prompt,
                    "images": [base64.b64encode(image_bytes).decode('utf-8')],
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    # Parse streaming response
                    content = ""
                    for line in response.text.strip().split('\n'):
                        if line:
                            try:
                                data = json.loads(line)
                                if 'response' in data:
                                    content += data['response']
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    
                    return LocalInferenceResult(
                        success=True,
                        content=content,
                        confidence=0.8,  # Default confidence for local models
                        processing_time=0.0,  # Will be set by caller
                        model_used=model_name,
                        metadata={"ollama_model": ollama_model}
                    )
                else:
                    raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
            
        except Exception as e:
            logger.error(f"Ollama inference error: {e}")
            return LocalInferenceResult(
                success=False,
                content="",
                confidence=0.0,
                processing_time=0.0,
                model_used=model_name,
                error=str(e)
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file.name)
            except:
                pass
    
    def _infer_transformers(self,
                          model_name: str,
                          image_data: str,
                          prompt: str,
                          max_tokens: int,
                          temperature: float) -> LocalInferenceResult:
        """Perform inference using Transformers"""
        try:
            import torch
            import base64
            from PIL import Image
            from io import BytesIO
            
            model_data = self.loaded_models[model_name]
            processor = model_data["processor"]
            model = model_data["model"]
            
            # Decode image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            
            # Prepare inputs
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Remove input prompt from output
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            return LocalInferenceResult(
                success=True,
                content=generated_text,
                confidence=0.75,  # Default confidence for local models
                processing_time=0.0,  # Will be set by caller
                model_used=model_name,
                metadata={"framework": "transformers"}
            )
            
        except Exception as e:
            logger.error(f"Transformers inference error: {e}")
            return LocalInferenceResult(
                success=False,
                content="",
                confidence=0.0,
                processing_time=0.0,
                model_used=model_name,
                error=str(e)
            )
    
    def _infer_custom(self,
                     model_name: str,
                     image_data: str,
                     prompt: str,
                     max_tokens: int,
                     temperature: float) -> LocalInferenceResult:
        """Perform inference using custom model"""
        # Placeholder for custom model inference
        return LocalInferenceResult(
            success=False,
            content="",
            confidence=0.0,
            processing_time=0.0,
            model_used=model_name,
            error="Custom model inference not implemented"
        )
    
    def _update_performance_stats(self, model_name: str, result: LocalInferenceResult):
        """Update performance statistics for a model"""
        if model_name not in self.performance_stats:
            self.performance_stats[model_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_time": 0.0,
                "avg_confidence": 0.0
            }
        
        stats = self.performance_stats[model_name]
        stats["total_requests"] += 1
        
        if result.success:
            stats["successful_requests"] += 1
            stats["total_time"] += result.processing_time
            
            # Update average confidence
            alpha = 0.1  # Learning rate
            stats["avg_confidence"] = (1 - alpha) * stats["avg_confidence"] + alpha * result.confidence
    
    def get_best_available_model(self, 
                               capabilities: List[str] = None,
                               max_size: ModelSize = ModelSize.LARGE) -> Optional[str]:
        """
        Get the best available model based on criteria
        
        Args:
            capabilities: Required capabilities
            max_size: Maximum acceptable model size
            
        Returns:
            Name of the best model or None if no suitable model found
        """
        capabilities = capabilities or ["vision", "text"]
        
        candidates = []
        
        for model_name, model_info in self.available_models.items():
            # Check if model is available
            if not model_info.is_available:
                continue
            
            # Check size constraint
            size_order = [ModelSize.TINY, ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE, ModelSize.XLARGE]
            if size_order.index(model_info.size) > size_order.index(max_size):
                continue
            
            # Check capabilities
            if not all(cap in model_info.capabilities for cap in capabilities):
                continue
            
            # Calculate score
            score = 0.0
            
            # Prefer loaded models
            if model_info.is_loaded:
                score += 0.3
            
            # Prefer smaller models for faster inference
            size_score = (len(size_order) - size_order.index(model_info.size)) / len(size_order)
            score += 0.2 * size_score
            
            # Consider performance history
            if model_name in self.performance_stats:
                stats = self.performance_stats[model_name]
                if stats["total_requests"] > 0:
                    success_rate = stats["successful_requests"] / stats["total_requests"]
                    score += 0.3 * success_rate
                    score += 0.2 * stats["avg_confidence"]
            
            candidates.append((model_name, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            try:
                del self.loaded_models[model_name]
                
                if model_name in self.available_models:
                    self.available_models[model_name].is_loaded = False
                
                logger.info(f"Unloaded model {model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unload model {model_name}: {e}")
                return False
        
        return True
    
    def cleanup(self):
        """Clean up loaded models and resources"""
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        
        logger.info("LocalModelManager cleanup completed")
    
    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get manager statistics"""
        return {
            "total_models": len(self.available_models),
            "loaded_models": len(self.loaded_models),
            "available_by_type": {
                model_type.value: len([m for m in self.available_models.values() if m.model_type == model_type])
                for model_type in LocalModelType
            },
            "performance_stats": self.performance_stats,
            "memory_usage": sum(info.memory_usage for info in self.available_models.values() if info.is_loaded),
            "models_dir": str(self.models_dir),
            "gpu_enabled": self.enable_gpu
        }