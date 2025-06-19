"""
Document Difficulty Classification System

This module provides comprehensive document difficulty assessment for multimodal processing,
helping optimize model selection and resource allocation.
"""

import numpy as np
import cv2
from PIL import Image, ImageStat, ImageFilter
from typing import Dict, List, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class DifficultyLevel(Enum):
    """Document difficulty levels"""
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    VERY_HARD = "Very Hard"

@dataclass
class DifficultyAssessment:
    """Comprehensive difficulty assessment result"""
    score: float  # 0.0 - 1.0
    level: DifficultyLevel
    confidence: float  # Confidence in assessment
    factors: Dict[str, float]  # Individual factor scores
    recommendations: Dict[str, Any]  # Processing recommendations
    metadata: Dict[str, Any]  # Additional metadata

class DifficultyFactor(ABC):
    """Base class for difficulty assessment factors"""
    
    @abstractmethod
    def assess(self, image: Image.Image, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Assess difficulty factor and return score and details"""
        pass
    
    @property
    @abstractmethod
    def weight(self) -> float:
        """Weight of this factor in overall assessment"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this factor"""
        pass

class ImageQualityFactor(DifficultyFactor):
    """Assess difficulty based on image quality"""
    
    @property
    def name(self) -> str:
        return "image_quality"
    
    @property
    def weight(self) -> float:
        return 0.25
    
    def assess(self, image: Image.Image, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Assess image quality factors"""
        try:
            # Convert to grayscale for analysis
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Resolution factor
            resolution = image.width * image.height
            resolution_score = min(resolution / (1920 * 1080), 1.0)  # Normalize to 1080p
            
            # Sharpness/blur detection using Laplacian variance
            laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000, 1.0)  # Normalize
            
            # Contrast assessment
            contrast = ImageStat.Stat(gray_image).stddev[0] / 128.0  # Normalize to 0-1
            
            # Noise assessment using local variance
            noise_score = self._assess_noise(img_array)
            
            # Brightness assessment
            brightness = np.mean(img_array) / 255.0
            brightness_score = 1 - abs(brightness - 0.5) * 2  # Penalize too dark/bright
            
            # Overall quality score (higher = better quality = easier to process)
            quality_factors = {
                'resolution': resolution_score,
                'sharpness': sharpness_score,
                'contrast': contrast,
                'noise': 1 - noise_score,  # Invert noise (less noise = higher score)
                'brightness': brightness_score
            }
            
            quality_score = np.mean(list(quality_factors.values()))
            
            # Convert to difficulty (inverse of quality)
            difficulty_score = 1 - quality_score
            
            details = {
                'factors': quality_factors,
                'overall_quality': quality_score,
                'resolution_pixels': resolution,
                'laplacian_variance': laplacian_var,
                'mean_brightness': brightness
            }
            
            return difficulty_score, details
            
        except Exception as e:
            logger.warning(f"Error in image quality assessment: {e}")
            return 0.5, {'error': str(e)}
    
    def _assess_noise(self, img_array: np.ndarray) -> float:
        """Assess image noise level"""
        try:
            # Apply Gaussian blur and compare with original
            blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
            noise = np.mean(np.abs(img_array.astype(float) - blurred.astype(float)))
            return min(noise / 50, 1.0)  # Normalize
        except:
            return 0.5

class TextComplexityFactor(DifficultyFactor):
    """Assess difficulty based on text complexity"""
    
    @property
    def name(self) -> str:
        return "text_complexity"
    
    @property
    def weight(self) -> float:
        return 0.20
    
    def assess(self, image: Image.Image, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Assess text complexity factors"""
        try:
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Text density estimation using edge detection
            edges = cv2.Canny(img_array, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Text line detection using horizontal projections
            horizontal_proj = np.sum(edges, axis=1)
            text_lines = len([i for i in range(1, len(horizontal_proj)) 
                            if horizontal_proj[i-1] == 0 and horizontal_proj[i] > 0])
            
            # Estimate text complexity based on edge patterns
            # More edges generally indicate more text or complex layouts
            text_density_score = min(edge_density * 10, 1.0)
            
            # Font size estimation (approximate)
            if text_lines > 0:
                avg_line_height = image.height / text_lines
                font_size_score = 1 - min(avg_line_height / 50, 1.0)  # Smaller text = harder
            else:
                font_size_score = 0.5
            
            # Text orientation assessment
            # Apply Hough line transform to detect text orientation
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            orientation_score = 0.0
            
            if lines is not None:
                angles = []
                for line in lines[:10]:  # Analyze top 10 lines
                    angle = line[0][1] * 180 / np.pi
                    angles.append(angle)
                
                if angles:
                    angle_std = np.std(angles)
                    orientation_score = min(angle_std / 45, 1.0)  # Higher variance = more complex
            
            complexity_factors = {
                'text_density': text_density_score,
                'font_size_complexity': font_size_score,
                'orientation_complexity': orientation_score,
                'estimated_text_lines': text_lines
            }
            
            complexity_score = np.mean(list(complexity_factors.values()))
            
            details = {
                'factors': complexity_factors,
                'edge_density': edge_density,
                'text_lines_detected': text_lines
            }
            
            return complexity_score, details
            
        except Exception as e:
            logger.warning(f"Error in text complexity assessment: {e}")
            return 0.5, {'error': str(e)}

class LayoutComplexityFactor(DifficultyFactor):
    """Assess difficulty based on document layout complexity"""
    
    @property
    def name(self) -> str:
        return "layout_complexity"
    
    @property
    def weight(self) -> float:
        return 0.20
    
    def assess(self, image: Image.Image, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Assess layout complexity"""
        try:
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Detect structural elements
            edges = cv2.Canny(img_array, 50, 150)
            
            # Find contours to identify distinct regions
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Region complexity based on number of distinct areas
            region_count = len([c for c in contours if cv2.contourArea(c) > 100])
            region_complexity = min(region_count / 20, 1.0)  # Normalize
            
            # Aspect ratio complexity
            aspect_ratio = image.width / image.height
            aspect_complexity = abs(aspect_ratio - 1.4) / 2  # 1.4 is standard A4 ratio
            
            # White space analysis
            white_pixels = np.sum(img_array > 200)  # Assume white background
            white_space_ratio = white_pixels / img_array.size
            white_space_complexity = 1 - white_space_ratio  # Less white space = more complex
            
            # Column detection using vertical projections
            vertical_proj = np.sum(edges, axis=0)
            columns = len([i for i in range(1, len(vertical_proj)) 
                          if vertical_proj[i-1] == 0 and vertical_proj[i] > np.mean(vertical_proj)])
            column_complexity = min(columns / 5, 1.0)  # Normalize to max 5 columns
            
            layout_factors = {
                'region_complexity': region_complexity,
                'aspect_ratio_complexity': aspect_complexity,
                'white_space_complexity': white_space_complexity,
                'column_complexity': column_complexity
            }
            
            layout_score = np.mean(list(layout_factors.values()))
            
            details = {
                'factors': layout_factors,
                'detected_regions': region_count,
                'aspect_ratio': aspect_ratio,
                'white_space_ratio': white_space_ratio,
                'estimated_columns': columns
            }
            
            return layout_score, details
            
        except Exception as e:
            logger.warning(f"Error in layout complexity assessment: {e}")
            return 0.5, {'error': str(e)}

class ContentTypeFactor(DifficultyFactor):
    """Assess difficulty based on document content type"""
    
    @property
    def name(self) -> str:
        return "content_type"
    
    @property
    def weight(self) -> float:
        return 0.15
    
    def assess(self, image: Image.Image, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Assess content type complexity"""
        try:
            # Get file metadata if available
            file_size = metadata.get('file_size', 0)
            file_type = metadata.get('file_type', 'unknown')
            
            # Base complexity by file type
            type_complexity = {
                'pdf': 0.7,  # PDFs often have complex layouts
                'png': 0.4,  # PNGs are usually screenshots or simple images
                'jpg': 0.3,  # JPGs are often photos or scanned documents
                'jpeg': 0.3,
                'tiff': 0.6,  # TIFFs often high-resolution scanned documents
                'unknown': 0.5
            }
            
            base_complexity = type_complexity.get(file_type.lower(), 0.5)
            
            # Size-based complexity
            size_mb = file_size / (1024 * 1024) if file_size > 0 else 1
            size_complexity = min(size_mb / 10, 1.0)  # Normalize to 10MB max
            
            # Color complexity
            if image.mode == 'RGB':
                color_variance = np.var([np.var(np.array(image)[:,:,i]) for i in range(3)])
                color_complexity = min(color_variance / 10000, 1.0)
            else:
                color_complexity = 0.2  # Grayscale is simpler
            
            content_factors = {
                'type_complexity': base_complexity,
                'size_complexity': size_complexity,
                'color_complexity': color_complexity
            }
            
            content_score = np.mean(list(content_factors.values()))
            
            details = {
                'factors': content_factors,
                'file_type': file_type,
                'file_size_mb': size_mb,
                'image_mode': image.mode
            }
            
            return content_score, details
            
        except Exception as e:
            logger.warning(f"Error in content type assessment: {e}")
            return 0.5, {'error': str(e)}

class SpecialElementsFactor(DifficultyFactor):
    """Assess difficulty based on special elements (tables, forms, signatures, etc.)"""
    
    @property
    def name(self) -> str:
        return "special_elements"
    
    @property
    def weight(self) -> float:
        return 0.20
    
    def assess(self, image: Image.Image, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Assess special elements complexity"""
        try:
            gray_image = image.convert('L')
            img_array = np.array(gray_image)
            
            # Table detection using line intersections
            edges = cv2.Canny(img_array, 50, 150)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Table score based on line intersections
            table_elements = cv2.bitwise_and(horizontal_lines, vertical_lines)
            table_score = min(np.sum(table_elements > 0) / 1000, 1.0)
            
            # Form detection using rectangular regions
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rectangular_regions = 0
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area threshold
                    # Approximate contour to polygon
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Check if approximately rectangular (4 corners)
                    if len(approx) == 4:
                        rectangular_regions += 1
            
            form_score = min(rectangular_regions / 10, 1.0)
            
            # Signature/handwriting detection using texture analysis
            # Apply different filters to detect handwritten elements
            laplacian = cv2.Laplacian(img_array, cv2.CV_64F)
            signature_variance = np.var(laplacian)
            signature_score = min(signature_variance / 5000, 1.0)
            
            # Complex graphics detection using contour complexity
            complex_shapes = 0
            for contour in contours:
                if cv2.contourArea(contour) > 200:
                    perimeter = cv2.arcLength(contour, True)
                    area = cv2.contourArea(contour)
                    
                    if perimeter > 0:
                        complexity_ratio = (perimeter ** 2) / area
                        if complexity_ratio > 50:  # Threshold for complex shapes
                            complex_shapes += 1
            
            graphics_score = min(complex_shapes / 5, 1.0)
            
            special_factors = {
                'table_complexity': table_score,
                'form_complexity': form_score,
                'signature_complexity': signature_score,
                'graphics_complexity': graphics_score
            }
            
            special_score = np.mean(list(special_factors.values()))
            
            details = {
                'factors': special_factors,
                'table_elements_detected': np.sum(table_elements > 0),
                'rectangular_regions': rectangular_regions,
                'complex_shapes': complex_shapes,
                'signature_variance': signature_variance
            }
            
            return special_score, details
            
        except Exception as e:
            logger.warning(f"Error in special elements assessment: {e}")
            return 0.5, {'error': str(e)}

class DocumentDifficultyClassifier:
    """Main document difficulty classification system"""
    
    def __init__(self):
        self.factors = [
            ImageQualityFactor(),
            TextComplexityFactor(),
            LayoutComplexityFactor(),
            ContentTypeFactor(),
            SpecialElementsFactor()
        ]
        
        # Difficulty thresholds
        self.thresholds = {
            DifficultyLevel.EASY: 0.25,
            DifficultyLevel.MEDIUM: 0.5,
            DifficultyLevel.HARD: 0.75,
            DifficultyLevel.VERY_HARD: 1.0
        }
        
        # Model recommendations for each difficulty level
        self.model_recommendations = {
            DifficultyLevel.EASY: {
                'primary': ['gpt-4o-mini', 'claude-3-5-haiku', 'gemini-1.5-flash'],
                'alternative': ['gpt-4o', 'claude-3-5-sonnet'],
                'max_tokens': 2000,
                'temperature': 0.1
            },
            DifficultyLevel.MEDIUM: {
                'primary': ['gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-pro'],
                'alternative': ['gpt-4-turbo', 'claude-3-opus'],
                'max_tokens': 4000,
                'temperature': 0.0
            },
            DifficultyLevel.HARD: {
                'primary': ['claude-3-5-sonnet', 'gpt-4o', 'claude-3-opus'],
                'alternative': ['gpt-4-turbo'],
                'max_tokens': 6000,
                'temperature': 0.0
            },
            DifficultyLevel.VERY_HARD: {
                'primary': ['claude-3-opus', 'claude-3-5-sonnet', 'gpt-4o'],
                'alternative': ['gpt-4-turbo'],
                'max_tokens': 8000,
                'temperature': 0.0
            }
        }
    
    def classify_image(self, image: Image.Image, metadata: Optional[Dict[str, Any]] = None) -> DifficultyAssessment:
        """Classify document difficulty from PIL Image"""
        if metadata is None:
            metadata = {}
        
        # Assess each factor
        factor_scores = {}
        factor_details = {}
        
        for factor in self.factors:
            try:
                score, details = factor.assess(image, metadata)
                factor_scores[factor.name] = score
                factor_details[factor.name] = details
                
                logger.debug(f"Factor {factor.name}: score={score:.3f}")
                
            except Exception as e:
                logger.error(f"Error assessing factor {factor.name}: {e}")
                factor_scores[factor.name] = 0.5  # Default to medium difficulty
                factor_details[factor.name] = {'error': str(e)}
        
        # Calculate weighted overall score
        total_weight = sum(factor.weight for factor in self.factors)
        weighted_score = sum(
            factor_scores[factor.name] * factor.weight 
            for factor in self.factors
        ) / total_weight
        
        # Determine difficulty level
        difficulty_level = self._score_to_level(weighted_score)
        
        # Calculate confidence based on factor agreement
        confidence = self._calculate_confidence(factor_scores)
        
        # Get recommendations
        recommendations = self._get_recommendations(difficulty_level, factor_scores, metadata)
        
        # Additional metadata
        processing_metadata = {
            'image_size': f"{image.width}x{image.height}",
            'image_mode': image.mode,
            'assessment_timestamp': np.datetime64('now').astype(str),
            'factor_weights': {factor.name: factor.weight for factor in self.factors}
        }
        
        return DifficultyAssessment(
            score=weighted_score,
            level=difficulty_level,
            confidence=confidence,
            factors=factor_scores,
            recommendations=recommendations,
            metadata={
                'factor_details': factor_details,
                'processing_metadata': processing_metadata
            }
        )
    
    def classify_from_file(self, file_content: bytes, file_type: str = 'unknown', 
                          file_size: Optional[int] = None) -> DifficultyAssessment:
        """Classify document difficulty from file content"""
        try:
            # Convert file content to PIL Image
            image = Image.open(BytesIO(file_content))
            
            # Prepare metadata
            metadata = {
                'file_type': file_type,
                'file_size': file_size or len(file_content)
            }
            
            return self.classify_image(image, metadata)
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            # Return default assessment
            return DifficultyAssessment(
                score=0.5,
                level=DifficultyLevel.MEDIUM,
                confidence=0.0,
                factors={},
                recommendations=self.model_recommendations[DifficultyLevel.MEDIUM],
                metadata={'error': str(e)}
            )
    
    def classify_from_base64(self, base64_data: str, file_type: str = 'unknown') -> DifficultyAssessment:
        """Classify document difficulty from base64 string"""
        try:
            # Decode base64 to bytes
            file_content = base64.b64decode(base64_data)
            return self.classify_from_file(file_content, file_type, len(file_content))
            
        except Exception as e:
            logger.error(f"Error processing base64 data: {e}")
            return DifficultyAssessment(
                score=0.5,
                level=DifficultyLevel.MEDIUM,
                confidence=0.0,
                factors={},
                recommendations=self.model_recommendations[DifficultyLevel.MEDIUM],
                metadata={'error': str(e)}
            )
    
    def _score_to_level(self, score: float) -> DifficultyLevel:
        """Convert numeric score to difficulty level"""
        if score < self.thresholds[DifficultyLevel.EASY]:
            return DifficultyLevel.EASY
        elif score < self.thresholds[DifficultyLevel.MEDIUM]:
            return DifficultyLevel.MEDIUM
        elif score < self.thresholds[DifficultyLevel.HARD]:
            return DifficultyLevel.HARD
        else:
            return DifficultyLevel.VERY_HARD
    
    def _calculate_confidence(self, factor_scores: Dict[str, float]) -> float:
        """Calculate confidence in the assessment based on factor agreement"""
        if not factor_scores:
            return 0.0
        
        scores = list(factor_scores.values())
        
        # Calculate variance - lower variance means higher confidence
        variance = np.var(scores)
        
        # Convert variance to confidence (0-1 scale)
        # Lower variance = higher confidence
        confidence = max(0.0, 1.0 - variance * 4)  # Scale factor of 4
        
        return min(confidence, 1.0)
    
    def _get_recommendations(self, difficulty_level: DifficultyLevel, 
                           factor_scores: Dict[str, float], 
                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Get processing recommendations based on difficulty assessment"""
        
        base_recommendations = self.model_recommendations[difficulty_level].copy()
        
        # Adjust recommendations based on specific factors
        adjustments = []
        
        # Image quality adjustments
        if factor_scores.get('image_quality', 0.5) > 0.7:
            adjustments.append("Consider image preprocessing to improve quality")
            base_recommendations['preprocessing'] = ['enhance_contrast', 'noise_reduction']
        
        # Text complexity adjustments
        if factor_scores.get('text_complexity', 0.5) > 0.8:
            adjustments.append("Use models with strong OCR capabilities")
            base_recommendations['max_tokens'] = min(base_recommendations['max_tokens'] * 1.5, 8000)
        
        # Layout complexity adjustments
        if factor_scores.get('layout_complexity', 0.5) > 0.7:
            adjustments.append("Consider using layout-aware models")
            base_recommendations['layout_aware'] = True
        
        # Special elements adjustments
        if factor_scores.get('special_elements', 0.5) > 0.6:
            adjustments.append("Use models with strong table and form recognition")
            base_recommendations['specialized_processing'] = ['table_detection', 'form_analysis']
        
        base_recommendations['adjustments'] = adjustments
        base_recommendations['estimated_processing_time'] = self._estimate_processing_time(difficulty_level, factor_scores)
        base_recommendations['estimated_cost_range'] = self._estimate_cost_range(difficulty_level)
        
        return base_recommendations
    
    def _estimate_processing_time(self, difficulty_level: DifficultyLevel, 
                                factor_scores: Dict[str, float]) -> Dict[str, float]:
        """Estimate processing time based on difficulty"""
        base_times = {
            DifficultyLevel.EASY: {'min': 2, 'max': 8, 'avg': 5},
            DifficultyLevel.MEDIUM: {'min': 5, 'max': 15, 'avg': 10},
            DifficultyLevel.HARD: {'min': 10, 'max': 25, 'avg': 18},
            DifficultyLevel.VERY_HARD: {'min': 15, 'max': 45, 'avg': 30}
        }
        
        base_time = base_times[difficulty_level]
        
        # Adjust based on specific factors
        complexity_multiplier = 1.0
        if factor_scores.get('special_elements', 0.5) > 0.8:
            complexity_multiplier *= 1.3
        if factor_scores.get('image_quality', 0.5) > 0.8:
            complexity_multiplier *= 1.2
        
        return {
            'min': base_time['min'] * complexity_multiplier,
            'max': base_time['max'] * complexity_multiplier,
            'avg': base_time['avg'] * complexity_multiplier
        }
    
    def _estimate_cost_range(self, difficulty_level: DifficultyLevel) -> Dict[str, float]:
        """Estimate cost range based on difficulty"""
        base_costs = {
            DifficultyLevel.EASY: {'min': 0.001, 'max': 0.005},
            DifficultyLevel.MEDIUM: {'min': 0.003, 'max': 0.015},
            DifficultyLevel.HARD: {'min': 0.008, 'max': 0.030},
            DifficultyLevel.VERY_HARD: {'min': 0.015, 'max': 0.060}
        }
        
        return base_costs[difficulty_level]
    
    def batch_classify(self, images: List[Tuple[Image.Image, Dict[str, Any]]]) -> List[DifficultyAssessment]:
        """Classify multiple documents"""
        results = []
        
        for image, metadata in images:
            assessment = self.classify_image(image, metadata)
            results.append(assessment)
        
        return results
    
    def get_difficulty_statistics(self, assessments: List[DifficultyAssessment]) -> Dict[str, Any]:
        """Get statistics from multiple assessments"""
        if not assessments:
            return {}
        
        scores = [a.score for a in assessments]
        levels = [a.level.value for a in assessments]
        confidences = [a.confidence for a in assessments]
        
        # Level distribution
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Factor analysis
        factor_averages = {}
        if assessments[0].factors:
            for factor_name in assessments[0].factors.keys():
                factor_scores = [a.factors.get(factor_name, 0) for a in assessments]
                factor_averages[factor_name] = {
                    'mean': np.mean(factor_scores),
                    'std': np.std(factor_scores),
                    'min': np.min(factor_scores),
                    'max': np.max(factor_scores)
                }
        
        return {
            'total_documents': len(assessments),
            'difficulty_distribution': level_counts,
            'score_statistics': {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            },
            'confidence_statistics': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'factor_analysis': factor_averages
        }

# Global classifier instance
document_difficulty_classifier = DocumentDifficultyClassifier()