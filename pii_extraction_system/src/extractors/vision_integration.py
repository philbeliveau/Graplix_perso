"""
Vision-LLM Integration with Existing PII Extraction Pipeline

This module provides seamless integration of the Vision-LLM PII extraction system
with the existing pipeline while maintaining backward compatibility.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from .base import PIIExtractorBase, PIIExtractionResult, PIIEntity
from .vision_document_classifier import VisionDocumentClassifier
from .prompt_router import PromptRouter, RoutingStrategy
from .vision_pii_extractor import VisionPIIExtractor, UserRole
from .local_model_manager import LocalModelManager
from .confidence_assessor import ConfidenceAssessor, QualityAssessment
from ..llm.multimodal_llm_service import llm_service

logger = logging.getLogger(__name__)


class VisionPipelineIntegration(PIIExtractorBase):
    """
    Integrated Vision-LLM PII extraction system that works seamlessly
    with the existing pipeline infrastructure.
    """
    
    def __init__(self,
                 enable_vision_extraction: bool = True,
                 enable_local_models: bool = False,
                 enable_quality_assessment: bool = True,
                 fallback_to_traditional: bool = True,
                 vision_config: Optional[Dict[str, Any]] = None):
        """
        Initialize Vision Pipeline Integration
        
        Args:
            enable_vision_extraction: Enable vision-based extraction
            enable_local_models: Enable local model support
            enable_quality_assessment: Enable quality assessment
            fallback_to_traditional: Fall back to traditional extractors on failure
            vision_config: Configuration for vision components
        """
        super().__init__("vision_pipeline_integration")
        
        self.enable_vision_extraction = enable_vision_extraction
        self.enable_local_models = enable_local_models
        self.enable_quality_assessment = enable_quality_assessment
        self.fallback_to_traditional = fallback_to_traditional
        
        # Apply configuration
        self.config = vision_config or {}
        self._apply_configuration()
        
        # Initialize components
        self.components = {}
        self._initialize_components()
        
        # Performance tracking
        self.performance_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'vision_extractions': 0,
            'fallback_extractions': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        logger.info(f"VisionPipelineIntegration initialized - "
                   f"Vision: {enable_vision_extraction}, "
                   f"Local: {enable_local_models}, "
                   f"Quality: {enable_quality_assessment}")
    
    def _apply_configuration(self):
        """Apply configuration settings"""
        # Vision extraction settings
        self.vision_settings = self.config.get('vision', {})
        self.routing_strategy = RoutingStrategy(
            self.vision_settings.get('routing_strategy', 'balanced')
        )
        self.confidence_threshold = self.vision_settings.get('confidence_threshold', 0.7)
        self.max_retries = self.vision_settings.get('max_retries', 2)
        
        # Local model settings
        self.local_settings = self.config.get('local_models', {})
        self.local_models_dir = self.local_settings.get('models_dir')
        self.enable_gpu = self.local_settings.get('enable_gpu', True)
        
        # Quality assessment settings
        self.quality_settings = self.config.get('quality_assessment', {})
        self.min_quality_threshold = self.quality_settings.get('min_threshold', 0.7)
        
        # Fallback settings
        self.fallback_settings = self.config.get('fallback', {})
        self.fallback_extractors = self.fallback_settings.get('extractors', ['rule_based', 'ner'])
    
    def _initialize_components(self):
        """Initialize all vision components"""
        try:
            # Document Classifier
            if self.vision_settings.get('enable_classification', True):
                self.components['classifier'] = VisionDocumentClassifier(
                    confidence_threshold=self.confidence_threshold
                )
                logger.info("Document classifier initialized")
            
            # Prompt Router
            if self.vision_settings.get('enable_routing', True):
                self.components['router'] = PromptRouter(
                    default_strategy=self.routing_strategy
                )
                logger.info("Prompt router initialized")
            
            # Vision PII Extractor
            if self.enable_vision_extraction:
                self.components['vision_extractor'] = VisionPIIExtractor(
                    enable_classification='classifier' in self.components,
                    enable_routing='router' in self.components,
                    default_routing_strategy=self.routing_strategy,
                    confidence_threshold=self.confidence_threshold,
                    max_retries=self.max_retries
                )
                logger.info("Vision PII extractor initialized")
            
            # Local Model Manager
            if self.enable_local_models:
                self.components['local_manager'] = LocalModelManager(
                    models_dir=self.local_models_dir,
                    enable_gpu=self.enable_gpu
                )
                logger.info("Local model manager initialized")
            
            # Confidence Assessor
            if self.enable_quality_assessment:
                self.components['assessor'] = ConfidenceAssessor(
                    min_confidence_threshold=self.min_quality_threshold
                )
                logger.info("Confidence assessor initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize vision components: {e}")
            if not self.fallback_to_traditional:
                raise
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """
        Extract PII using integrated vision-LLM system with fallback support
        
        Args:
            document: Processed document containing image data and metadata
            
        Returns:
            PIIExtractionResult with extracted PII entities
        """
        start_time = time.time()
        extraction_method = "unknown"
        
        try:
            # Update performance tracking
            self.performance_stats['total_extractions'] += 1
            
            # Check if document supports vision extraction
            if self._supports_vision_extraction(document):
                if self.enable_vision_extraction and 'vision_extractor' in self.components:
                    try:
                        # Attempt vision-based extraction
                        result = self._extract_with_vision(document)
                        extraction_method = "vision"
                        self.performance_stats['vision_extractions'] += 1
                        
                        # Quality assessment
                        if self.enable_quality_assessment and 'assessor' in self.components:
                            result = self._assess_and_enhance_result(result, document)
                        
                        # Check if result meets quality standards
                        if self._meets_quality_standards(result):
                            return self._finalize_result(result, extraction_method, start_time)
                        else:
                            logger.warning("Vision extraction result below quality threshold, trying fallback")
                    
                    except Exception as e:
                        logger.error(f"Vision extraction failed: {e}")
                        if not self.fallback_to_traditional:
                            raise
            
            # Fallback to traditional extraction methods
            if self.fallback_to_traditional:
                result = self._extract_with_fallback(document)
                extraction_method = "fallback"
                self.performance_stats['fallback_extractions'] += 1
                return self._finalize_result(result, extraction_method, start_time)
            else:
                # No fallback, return empty result
                return PIIExtractionResult(
                    pii_entities=[],
                    confidence_scores=[],
                    processing_time=time.time() - start_time,
                    error="Vision extraction failed and fallback disabled",
                    metadata={'extraction_method': 'failed'}
                )
        
        except Exception as e:
            logger.error(f"Integrated extraction failed: {e}")
            return PIIExtractionResult(
                pii_entities=[],
                confidence_scores=[],
                processing_time=time.time() - start_time,
                error=str(e),
                metadata={'extraction_method': 'error'}
            )
    
    def _supports_vision_extraction(self, document: Dict[str, Any]) -> bool:
        """Check if document supports vision-based extraction"""
        
        # Check for image data
        image_keys = ['image_data', 'base64_image', 'image', 'content']
        has_image_data = any(key in document and document[key] for key in image_keys)
        
        # Check for supported file types
        supported_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.tiff', '.bmp']
        file_path = document.get('file_path', '')
        has_supported_file = any(file_path.lower().endswith(ext) for ext in supported_extensions)
        
        return has_image_data or has_supported_file
    
    def _extract_with_vision(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """Extract PII using vision-based methods"""
        
        vision_extractor = self.components['vision_extractor']
        
        # Add integration metadata to document
        enhanced_document = document.copy()
        enhanced_document['metadata'] = enhanced_document.get('metadata', {})
        enhanced_document['metadata'].update({
            'extraction_mode': 'vision',
            'integration_enabled': True,
            'components_available': list(self.components.keys())
        })
        
        # Perform vision extraction
        result = vision_extractor.extract_pii(enhanced_document)
        
        # Add integration metadata to result
        result.metadata.update({
            'vision_extraction': True,
            'integration_version': '1.0',
            'components_used': list(self.components.keys())
        })
        
        return result
    
    def _extract_with_fallback(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """Extract PII using traditional fallback methods"""
        
        # Import traditional extractors
        fallback_results = []
        
        for extractor_name in self.fallback_extractors:
            try:
                if extractor_name == 'rule_based':
                    from .rule_based import RuleBasedExtractor
                    extractor = RuleBasedExtractor()
                elif extractor_name == 'ner':
                    from .ner_extractor import NERExtractor
                    extractor = NERExtractor()
                else:
                    logger.warning(f"Unknown fallback extractor: {extractor_name}")
                    continue
                
                result = extractor.extract_pii(document)
                fallback_results.append(result)
                
            except ImportError as e:
                logger.warning(f"Fallback extractor {extractor_name} not available: {e}")
            except Exception as e:
                logger.error(f"Fallback extractor {extractor_name} failed: {e}")
        
        # Combine fallback results
        if fallback_results:
            return self._combine_fallback_results(fallback_results, document)
        else:
            return PIIExtractionResult(
                pii_entities=[],
                confidence_scores=[],
                processing_time=0.0,
                error="All fallback extractors failed",
                metadata={'extraction_method': 'fallback_failed'}
            )
    
    def _combine_fallback_results(self, 
                                results: List[PIIExtractionResult],
                                document: Dict[str, Any]) -> PIIExtractionResult:
        """Combine results from multiple fallback extractors"""
        
        combined_entities = []
        combined_scores = []
        total_processing_time = 0.0
        
        for result in results:
            combined_entities.extend(result.pii_entities)
            combined_scores.extend(result.confidence_scores)
            total_processing_time += result.processing_time
        
        # Remove duplicates
        deduplicated_entities = self._deduplicate_entities(combined_entities)
        
        return PIIExtractionResult(
            pii_entities=deduplicated_entities,
            confidence_scores=combined_scores,
            processing_time=total_processing_time,
            metadata={
                'extraction_method': 'fallback_combined',
                'num_fallback_extractors': len(results),
                'entities_before_dedup': len(combined_entities),
                'entities_after_dedup': len(deduplicated_entities)
            }
        )
    
    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove duplicate entities"""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            # Create a key based on text and type
            key = (entity.text.lower().strip(), entity.pii_type)
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
            else:
                # Keep the entity with higher confidence
                existing_index = next(
                    i for i, e in enumerate(deduplicated)
                    if (e.text.lower().strip(), e.pii_type) == key
                )
                if entity.confidence > deduplicated[existing_index].confidence:
                    deduplicated[existing_index] = entity
        
        return deduplicated
    
    def _assess_and_enhance_result(self, 
                                 result: PIIExtractionResult,
                                 document: Dict[str, Any]) -> PIIExtractionResult:
        """Assess and enhance extraction result using confidence assessor"""
        
        assessor = self.components['assessor']
        
        # Get classification and transcribed text from result metadata
        classification = None
        transcribed_text = None
        
        if result.metadata:
            classification_data = result.metadata.get('classification')
            if classification_data:
                # Reconstruct classification object if needed
                pass
            
            transcribed_text = result.metadata.get('transcribed_text', '')
        
        # Perform quality assessment
        try:
            quality_assessment = assessor.assess_extraction_result(
                extraction_result=result,
                classification=classification,
                transcribed_text=transcribed_text
            )
            
            # Add quality assessment to metadata
            result.metadata.update({
                'quality_assessment': {
                    'overall_quality': quality_assessment.overall_quality.value,
                    'confidence_score': quality_assessment.confidence_score,
                    'requires_human_review': quality_assessment.requires_human_review,
                    'recommendations': quality_assessment.recommendations,
                    'flagged_entities': quality_assessment.flagged_entities
                }
            })
            
            logger.info(f"Quality assessment: {quality_assessment.overall_quality.value} "
                       f"(score: {quality_assessment.confidence_score:.3f})")
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            result.metadata['quality_assessment_error'] = str(e)
        
        return result
    
    def _meets_quality_standards(self, result: PIIExtractionResult) -> bool:
        """Check if extraction result meets quality standards"""
        
        if not result.pii_entities:
            return False
        
        # Check overall confidence
        if result.confidence_scores:
            avg_confidence = sum(result.confidence_scores) / len(result.confidence_scores)
            if avg_confidence < self.min_quality_threshold:
                return False
        
        # Check quality assessment if available
        quality_data = result.metadata.get('quality_assessment', {})
        if quality_data:
            quality_score = quality_data.get('confidence_score', 0.0)
            if quality_score < self.min_quality_threshold:
                return False
            
            # Check for critical quality flags
            quality_flag = quality_data.get('overall_quality', '')
            if quality_flag in ['poor', 'suspicious']:
                return False
        
        return True
    
    def _finalize_result(self, 
                        result: PIIExtractionResult,
                        extraction_method: str,
                        start_time: float) -> PIIExtractionResult:
        """Finalize extraction result with integration metadata"""
        
        total_processing_time = time.time() - start_time
        
        # Update performance statistics
        if result.pii_entities:
            self.performance_stats['successful_extractions'] += 1
            
            # Update average confidence
            if result.confidence_scores:
                avg_confidence = sum(result.confidence_scores) / len(result.confidence_scores)
                alpha = 0.1  # Learning rate
                self.performance_stats['average_confidence'] = (
                    (1 - alpha) * self.performance_stats['average_confidence'] + 
                    alpha * avg_confidence
                )
        
        # Update average processing time
        alpha = 0.1
        self.performance_stats['average_processing_time'] = (
            (1 - alpha) * self.performance_stats['average_processing_time'] + 
            alpha * total_processing_time
        )
        
        # Add final integration metadata
        result.metadata.update({
            'integration': {
                'extraction_method': extraction_method,
                'total_processing_time': total_processing_time,
                'components_enabled': {
                    'vision_extraction': self.enable_vision_extraction,
                    'local_models': self.enable_local_models,
                    'quality_assessment': self.enable_quality_assessment,
                    'fallback': self.fallback_to_traditional
                },
                'performance_stats': self.performance_stats.copy()
            }
        })
        
        # Update result timing
        result.processing_time = total_processing_time
        
        logger.info(f"Extraction completed using {extraction_method} method - "
                   f"{len(result.pii_entities)} entities found in {total_processing_time:.2f}s")
        
        return result
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all integration components"""
        
        status = {
            'integration_enabled': True,
            'components': {},
            'configuration': {
                'vision_extraction': self.enable_vision_extraction,
                'local_models': self.enable_local_models,
                'quality_assessment': self.enable_quality_assessment,
                'fallback_enabled': self.fallback_to_traditional
            },
            'performance_stats': self.performance_stats.copy()
        }
        
        # Check component status
        for component_name, component in self.components.items():
            try:
                if hasattr(component, 'get_available_models'):
                    # For services with model information
                    status['components'][component_name] = {
                        'status': 'active',
                        'available_models': component.get_available_models(),
                        'type': component.__class__.__name__
                    }
                elif hasattr(component, 'available_models'):
                    # For managers with available models
                    status['components'][component_name] = {
                        'status': 'active',
                        'available_models': list(component.available_models.keys()),
                        'type': component.__class__.__name__
                    }
                else:
                    # For other components
                    status['components'][component_name] = {
                        'status': 'active',
                        'type': component.__class__.__name__
                    }
            except Exception as e:
                status['components'][component_name] = {
                    'status': 'error',
                    'error': str(e),
                    'type': component.__class__.__name__
                }
        
        return status
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update integration configuration"""
        
        self.config.update(new_config)
        self._apply_configuration()
        
        # Reinitialize components if needed
        if new_config.get('reinitialize_components', False):
            self._initialize_components()
        
        logger.info("Integration configuration updated")
    
    def cleanup(self):
        """Clean up integration resources"""
        
        # Cleanup local model manager
        if 'local_manager' in self.components:
            self.components['local_manager'].cleanup()
        
        # Clear components
        self.components.clear()
        
        logger.info("Integration cleanup completed")