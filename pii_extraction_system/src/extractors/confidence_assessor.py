"""
Confidence Assessor for Vision-LLM PII Extraction

This module provides comprehensive confidence scoring, automatic flagging,
and quality assessment for PII extraction results.
"""

import os
import time
import json
import logging
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter, defaultdict

from .base import PIIEntity, PIIExtractionResult
from .vision_document_classifier import DocumentClassification, DocumentDifficulty, DocumentDomain

logger = logging.getLogger(__name__)


class ConfidenceLevel(Enum):
    """Confidence levels for PII entities"""
    VERY_LOW = "very_low"    # < 0.3
    LOW = "low"              # 0.3 - 0.5
    MEDIUM = "medium"        # 0.5 - 0.7
    HIGH = "high"            # 0.7 - 0.9
    VERY_HIGH = "very_high"  # > 0.9


class QualityFlag(Enum):
    """Quality flags for extraction results"""
    EXCELLENT = "excellent"
    GOOD = "good"
    QUESTIONABLE = "questionable"
    POOR = "poor"
    NEEDS_REVIEW = "needs_review"
    SUSPICIOUS = "suspicious"


class ValidationResult(Enum):
    """Validation results for PII entities"""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    FORMAT_ERROR = "format_error"
    CONTEXT_MISMATCH = "context_mismatch"


@dataclass
class ConfidenceFactors:
    """Factors contributing to confidence assessment"""
    model_confidence: float = 0.0
    format_validation: float = 0.0
    context_consistency: float = 0.0
    cross_validation: float = 0.0
    pattern_matching: float = 0.0
    domain_relevance: float = 0.0
    extraction_consistency: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_weighted_score(self) -> float:
        """Calculate weighted confidence score"""
        weights = {
            'model_confidence': 0.25,
            'format_validation': 0.20,
            'context_consistency': 0.15,
            'cross_validation': 0.15,
            'pattern_matching': 0.10,
            'domain_relevance': 0.10,
            'extraction_consistency': 0.05
        }
        
        score = 0.0
        for factor, weight in weights.items():
            score += getattr(self, factor) * weight
        
        return min(1.0, max(0.0, score))


@dataclass
class QualityAssessment:
    """Quality assessment for extraction results"""
    overall_quality: QualityFlag
    confidence_distribution: Dict[str, int]
    total_entities: int
    high_confidence_entities: int
    flagged_entities: List[str]
    validation_summary: Dict[str, int]
    recommendations: List[str]
    requires_human_review: bool
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EntityAssessment:
    """Individual entity assessment"""
    entity: PIIEntity
    confidence_factors: ConfidenceFactors
    adjusted_confidence: float
    confidence_level: ConfidenceLevel
    validation_result: ValidationResult
    flags: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfidenceAssessor:
    """
    Advanced confidence assessor for PII extraction results with
    automatic flagging, quality assessment, and validation.
    """
    
    def __init__(self,
                 min_confidence_threshold: float = 0.7,
                 enable_cross_validation: bool = True,
                 enable_format_validation: bool = True,
                 enable_context_analysis: bool = True):
        """
        Initialize Confidence Assessor
        
        Args:
            min_confidence_threshold: Minimum threshold for high confidence
            enable_cross_validation: Enable cross-validation checks
            enable_format_validation: Enable format validation
            enable_context_analysis: Enable context analysis
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.enable_cross_validation = enable_cross_validation
        self.enable_format_validation = enable_format_validation
        self.enable_context_analysis = enable_context_analysis
        
        # PII format patterns for validation
        self.format_patterns = {
            'EMAIL': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'PHONE': re.compile(r'^[\+]?[1-9][\d\s\-\(\)\.]{8,15}$'),
            'SSN': re.compile(r'^\d{3}-?\d{2}-?\d{4}$'),
            'CREDIT_CARD': re.compile(r'^(?:\d{4}[\s\-]?){3}\d{4}$'),
            'POSTAL_CODE': re.compile(r'^[A-Z]\d[A-Z]\s?\d[A-Z]\d$'),  # Canadian postal code
            'ZIP_CODE': re.compile(r'^\d{5}(-\d{4})?$'),  # US ZIP code
            'DATE': re.compile(r'^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$|^\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}$')
        }
        
        # Context keywords for domain relevance
        self.domain_keywords = {
            DocumentDomain.HR: {
                'PERSON': ['employee', 'staff', 'worker', 'personnel'],
                'EMAIL': ['contact', 'email', 'correspondence'],
                'PHONE': ['extension', 'direct', 'mobile', 'office'],
                'ADDRESS': ['residence', 'home', 'mailing']
            },
            DocumentDomain.FINANCE: {
                'PERSON': ['account holder', 'customer', 'client', 'applicant'],
                'EMAIL': ['billing', 'statement', 'notification'],
                'PHONE': ['contact', 'primary', 'alternate'],
                'BANK_ACCOUNT': ['account', 'routing', 'swift'],
                'CREDIT_CARD': ['card', 'payment', 'billing']
            },
            DocumentDomain.MEDICAL: {
                'PERSON': ['patient', 'emergency contact', 'physician', 'doctor'],
                'EMAIL': ['contact', 'notification', 'appointment'],
                'PHONE': ['emergency', 'primary', 'contact'],
                'ID_NUMBER': ['patient id', 'medical record', 'health card']
            }
        }
        
        # Common false positive patterns
        self.false_positive_patterns = {
            'EMAIL': [
                r'example@.*\.com',
                r'test@.*\.com',
                r'.*@example\.',
                r'.*@test\.',
                r'noreply@',
                r'admin@'
            ],
            'PHONE': [
                r'^555[\s\-]?555[\s\-]?\d{4}$',  # Common fake numbers
                r'^123[\s\-]?456[\s\-]?\d{4}$',
                r'^000[\s\-]?\d{3}[\s\-]?\d{4}$'
            ],
            'PERSON': [
                r'^(test|example|sample|demo)\s',
                r'\s(test|example|sample|demo)$',
                r'^(john|jane)\s(doe|smith)$'
            ]
        }
        
        logger.info("ConfidenceAssessor initialized")
    
    def assess_extraction_result(self,
                               extraction_result: PIIExtractionResult,
                               classification: Optional[DocumentClassification] = None,
                               transcribed_text: Optional[str] = None) -> QualityAssessment:
        """
        Assess the quality and confidence of extraction results
        
        Args:
            extraction_result: PII extraction result to assess
            classification: Document classification for context
            transcribed_text: Transcribed text for validation
            
        Returns:
            QualityAssessment with detailed analysis
        """
        start_time = time.time()
        
        if not extraction_result.pii_entities:
            return QualityAssessment(
                overall_quality=QualityFlag.POOR,
                confidence_distribution={},
                total_entities=0,
                high_confidence_entities=0,
                flagged_entities=[],
                validation_summary={'no_entities': 1},
                recommendations=["No PII entities found - verify document content and model performance"],
                requires_human_review=True,
                confidence_score=0.0,
                metadata={'assessment_time': time.time() - start_time}
            )
        
        # Assess individual entities
        entity_assessments = []
        for entity in extraction_result.pii_entities:
            assessment = self.assess_entity(
                entity=entity,
                classification=classification,
                transcribed_text=transcribed_text,
                all_entities=extraction_result.pii_entities
            )
            entity_assessments.append(assessment)
        
        # Calculate overall quality metrics
        quality_assessment = self._calculate_overall_quality(
            entity_assessments,
            extraction_result,
            classification
        )
        
        quality_assessment.metadata['assessment_time'] = time.time() - start_time
        quality_assessment.metadata['entity_assessments'] = [
            {
                'entity_text': a.entity.text,
                'entity_type': a.entity.pii_type,
                'adjusted_confidence': a.adjusted_confidence,
                'flags': a.flags,
                'validation_result': a.validation_result.value
            }
            for a in entity_assessments
        ]
        
        logger.info(f"Quality assessment completed: {quality_assessment.overall_quality.value} "
                   f"({quality_assessment.high_confidence_entities}/{quality_assessment.total_entities} high confidence)")
        
        return quality_assessment
    
    def assess_entity(self,
                     entity: PIIEntity,
                     classification: Optional[DocumentClassification] = None,
                     transcribed_text: Optional[str] = None,
                     all_entities: Optional[List[PIIEntity]] = None) -> EntityAssessment:
        """
        Assess confidence and quality of individual PII entity
        
        Args:
            entity: PII entity to assess
            classification: Document classification for context
            transcribed_text: Full transcribed text for validation
            all_entities: All entities for cross-validation
            
        Returns:
            EntityAssessment with detailed analysis
        """
        # Initialize confidence factors
        factors = ConfidenceFactors(model_confidence=entity.confidence)
        
        # Format validation
        if self.enable_format_validation:
            factors.format_validation = self._validate_format(entity)
        
        # Context consistency
        if self.enable_context_analysis and transcribed_text:
            factors.context_consistency = self._analyze_context_consistency(
                entity, transcribed_text, classification
            )
        
        # Cross-validation with other entities
        if self.enable_cross_validation and all_entities:
            factors.cross_validation = self._cross_validate_entity(entity, all_entities)
        
        # Pattern matching validation
        factors.pattern_matching = self._validate_patterns(entity)
        
        # Domain relevance
        if classification:
            factors.domain_relevance = self._assess_domain_relevance(entity, classification)
        
        # Extraction consistency
        factors.extraction_consistency = self._assess_extraction_consistency(entity)
        
        # Calculate adjusted confidence
        adjusted_confidence = factors.calculate_weighted_score()
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(adjusted_confidence)
        
        # Validate entity
        validation_result = self._validate_entity(entity, transcribed_text)
        
        # Generate flags and recommendations
        flags = self._generate_flags(entity, factors, validation_result)
        recommendations = self._generate_recommendations(entity, factors, validation_result)
        
        return EntityAssessment(
            entity=entity,
            confidence_factors=factors,
            adjusted_confidence=adjusted_confidence,
            confidence_level=confidence_level,
            validation_result=validation_result,
            flags=flags,
            recommendations=recommendations,
            metadata={
                'original_confidence': entity.confidence,
                'confidence_adjustment': adjusted_confidence - entity.confidence
            }
        )
    
    def _validate_format(self, entity: PIIEntity) -> float:
        """Validate entity format against known patterns"""
        pii_type = entity.pii_type
        text = entity.text.strip()
        
        if pii_type in self.format_patterns:
            pattern = self.format_patterns[pii_type]
            if pattern.match(text):
                return 1.0
            else:
                # Partial matches for common variations
                if pii_type == 'EMAIL' and '@' in text and '.' in text:
                    return 0.6
                elif pii_type == 'PHONE' and any(c.isdigit() for c in text) and len(text) >= 10:
                    return 0.7
                else:
                    return 0.2
        
        # For types without strict patterns, use heuristics
        if pii_type == 'PERSON':
            words = text.split()
            if len(words) >= 2 and all(word.isalpha() for word in words):
                return 0.9
            elif len(words) >= 2:
                return 0.6
            else:
                return 0.3
        
        elif pii_type == 'ADDRESS':
            # Basic address validation
            if any(keyword in text.lower() for keyword in ['street', 'st', 'avenue', 'ave', 'road', 'rd']):
                return 0.8
            elif len(text.split()) >= 3:
                return 0.6
            else:
                return 0.4
        
        return 0.5  # Default for unknown types
    
    def _analyze_context_consistency(self,
                                   entity: PIIEntity,
                                   transcribed_text: str,
                                   classification: Optional[DocumentClassification]) -> float:
        """Analyze context consistency of entity within document"""
        
        entity_text = entity.text.lower()
        full_text = transcribed_text.lower()
        
        # Find entity in text with surrounding context
        entity_pos = full_text.find(entity_text)
        if entity_pos == -1:
            return 0.3  # Entity not found in text is suspicious
        
        # Extract context around entity
        context_start = max(0, entity_pos - 50)
        context_end = min(len(full_text), entity_pos + len(entity_text) + 50)
        context = full_text[context_start:context_end]
        
        # Look for contextual indicators
        context_score = 0.5  # Base score
        
        # Domain-specific context keywords
        if classification and classification.domain in self.domain_keywords:
            domain_keywords = self.domain_keywords[classification.domain].get(entity.pii_type, [])
            keyword_matches = sum(1 for keyword in domain_keywords if keyword in context)
            if keyword_matches > 0:
                context_score += min(0.3, keyword_matches * 0.1)
        
        # General PII context indicators
        pii_indicators = {
            'EMAIL': ['email', 'e-mail', 'contact', '@', 'send to'],
            'PHONE': ['phone', 'telephone', 'cell', 'mobile', 'call', 'number'],
            'ADDRESS': ['address', 'located', 'residence', 'street', 'city'],
            'PERSON': ['name', 'mr.', 'mrs.', 'ms.', 'dr.', 'contact'],
            'DATE': ['date', 'born', 'birth', 'on', 'signed', 'effective']
        }
        
        indicators = pii_indicators.get(entity.pii_type, [])
        indicator_matches = sum(1 for indicator in indicators if indicator in context)
        if indicator_matches > 0:
            context_score += min(0.2, indicator_matches * 0.05)
        
        return min(1.0, context_score)
    
    def _cross_validate_entity(self, entity: PIIEntity, all_entities: List[PIIEntity]) -> float:
        """Cross-validate entity against other extracted entities"""
        
        same_type_entities = [e for e in all_entities if e.pii_type == entity.pii_type and e != entity]
        
        if not same_type_entities:
            return 0.5  # No other entities of same type to compare
        
        validation_score = 0.5
        
        # Check for duplicates (good for consistency)
        duplicates = [e for e in same_type_entities if e.text == entity.text]
        if duplicates:
            validation_score += 0.2
        
        # Check for similar patterns
        if entity.pii_type == 'EMAIL':
            domains = [e.text.split('@')[-1] for e in same_type_entities if '@' in e.text]
            entity_domain = entity.text.split('@')[-1] if '@' in entity.text else ''
            if entity_domain and entity_domain in domains:
                validation_score += 0.1
        
        elif entity.pii_type == 'PHONE':
            # Check for similar phone number patterns
            entity_digits = ''.join(c for c in entity.text if c.isdigit())
            for other in same_type_entities:
                other_digits = ''.join(c for c in other.text if c.isdigit())
                if entity_digits[:3] == other_digits[:3]:  # Same area code
                    validation_score += 0.05
                    break
        
        return min(1.0, validation_score)
    
    def _validate_patterns(self, entity: PIIEntity) -> float:
        """Validate against known patterns and false positives"""
        
        text = entity.text.lower()
        pii_type = entity.pii_type
        
        # Check for false positive patterns
        if pii_type in self.false_positive_patterns:
            for pattern in self.false_positive_patterns[pii_type]:
                if re.match(pattern, text):
                    return 0.1  # Likely false positive
        
        # Positive pattern matching
        if pii_type == 'EMAIL':
            # Common email providers
            common_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
            for domain in common_domains:
                if domain in text:
                    return 0.9
            return 0.7
        
        elif pii_type == 'PERSON':
            # Common name patterns
            if any(title in text for title in ['mr.', 'mrs.', 'ms.', 'dr.']):
                return 0.9
            words = text.split()
            if len(words) == 2 and all(word.istitle() for word in words):
                return 0.8
            return 0.6
        
        return 0.7  # Default pattern score
    
    def _assess_domain_relevance(self, entity: PIIEntity, classification: DocumentClassification) -> float:
        """Assess relevance of entity to document domain"""
        
        domain = classification.domain
        pii_type = entity.pii_type
        
        # Domain-specific relevance scores
        relevance_matrix = {
            DocumentDomain.HR: {
                'PERSON': 0.9, 'EMAIL': 0.8, 'PHONE': 0.8, 'ADDRESS': 0.7,
                'EMPLOYEE_ID': 0.9, 'SSN': 0.8, 'DATE': 0.6
            },
            DocumentDomain.FINANCE: {
                'PERSON': 0.8, 'EMAIL': 0.7, 'PHONE': 0.7, 'ADDRESS': 0.8,
                'BANK_ACCOUNT': 0.9, 'CREDIT_CARD': 0.9, 'SSN': 0.8, 'DATE': 0.7
            },
            DocumentDomain.MEDICAL: {
                'PERSON': 0.9, 'EMAIL': 0.6, 'PHONE': 0.8, 'ADDRESS': 0.7,
                'MEDICAL_RECORD': 0.9, 'DATE': 0.8, 'SSN': 0.7
            },
            DocumentDomain.LEGAL: {
                'PERSON': 0.9, 'EMAIL': 0.7, 'PHONE': 0.7, 'ADDRESS': 0.8,
                'DATE': 0.9, 'SSN': 0.8, 'ID_NUMBER': 0.8
            }
        }
        
        if domain in relevance_matrix and pii_type in relevance_matrix[domain]:
            return relevance_matrix[domain][pii_type]
        
        return 0.5  # Default relevance
    
    def _assess_extraction_consistency(self, entity: PIIEntity) -> float:
        """Assess consistency of extraction method and metadata"""
        
        consistency_score = 0.5
        
        # Check for consistent extractor information
        if entity.extractor:
            consistency_score += 0.2
        
        # Check for consistent metadata
        if entity.metadata:
            consistency_score += 0.1
            
            # Model consistency
            if 'model_used' in entity.metadata:
                consistency_score += 0.1
            
            # Source consistency
            if 'source' in entity.metadata:
                consistency_score += 0.1
        
        return min(1.0, consistency_score)
    
    def _validate_entity(self, entity: PIIEntity, transcribed_text: Optional[str]) -> ValidationResult:
        """Validate entity against multiple criteria"""
        
        # Format validation
        format_valid = self._validate_format(entity) > 0.5
        
        # Text presence validation
        text_present = True
        if transcribed_text:
            text_present = entity.text.lower() in transcribed_text.lower()
        
        # False positive check
        is_false_positive = False
        if entity.pii_type in self.false_positive_patterns:
            for pattern in self.false_positive_patterns[entity.pii_type]:
                if re.match(pattern, entity.text.lower()):
                    is_false_positive = True
                    break
        
        # Determine validation result
        if is_false_positive:
            return ValidationResult.INVALID
        elif format_valid and text_present:
            return ValidationResult.VALID
        elif format_valid or text_present:
            return ValidationResult.UNCERTAIN
        else:
            return ValidationResult.FORMAT_ERROR
    
    def _generate_flags(self,
                       entity: PIIEntity,
                       factors: ConfidenceFactors,
                       validation_result: ValidationResult) -> List[str]:
        """Generate quality flags for entity"""
        flags = []
        
        if factors.model_confidence < 0.5:
            flags.append("low_model_confidence")
        
        if factors.format_validation < 0.5:
            flags.append("format_validation_failed")
        
        if factors.context_consistency < 0.3:
            flags.append("poor_context_consistency")
        
        if validation_result == ValidationResult.INVALID:
            flags.append("validation_failed")
        elif validation_result == ValidationResult.UNCERTAIN:
            flags.append("uncertain_validation")
        
        if factors.pattern_matching < 0.3:
            flags.append("suspicious_pattern")
        
        if factors.calculate_weighted_score() < self.min_confidence_threshold:
            flags.append("below_confidence_threshold")
        
        return flags
    
    def _generate_recommendations(self,
                                entity: PIIEntity,
                                factors: ConfidenceFactors,
                                validation_result: ValidationResult) -> List[str]:
        """Generate recommendations for entity"""
        recommendations = []
        
        if factors.model_confidence < 0.6:
            recommendations.append("Consider using a different model or verification method")
        
        if factors.format_validation < 0.5:
            recommendations.append("Verify entity format against expected patterns")
        
        if factors.context_consistency < 0.4:
            recommendations.append("Review entity context within document")
        
        if validation_result == ValidationResult.INVALID:
            recommendations.append("Remove or flag as false positive")
        elif validation_result == ValidationResult.UNCERTAIN:
            recommendations.append("Requires manual verification")
        
        if not recommendations:
            if factors.calculate_weighted_score() > 0.8:
                recommendations.append("High confidence - entity appears valid")
            else:
                recommendations.append("Medium confidence - consider additional validation")
        
        return recommendations
    
    def _calculate_overall_quality(self,
                                 entity_assessments: List[EntityAssessment],
                                 extraction_result: PIIExtractionResult,
                                 classification: Optional[DocumentClassification]) -> QualityAssessment:
        """Calculate overall quality assessment"""
        
        total_entities = len(entity_assessments)
        
        # Confidence distribution
        confidence_distribution = {level.value: 0 for level in ConfidenceLevel}
        for assessment in entity_assessments:
            confidence_distribution[assessment.confidence_level.value] += 1
        
        # High confidence entities
        high_confidence_entities = sum(1 for a in entity_assessments 
                                     if a.adjusted_confidence >= self.min_confidence_threshold)
        
        # Flagged entities
        flagged_entities = [a.entity.text for a in entity_assessments if a.flags]
        
        # Validation summary
        validation_summary = {}
        for assessment in entity_assessments:
            result = assessment.validation_result.value
            validation_summary[result] = validation_summary.get(result, 0) + 1
        
        # Overall confidence score
        if entity_assessments:
            overall_confidence = statistics.mean(a.adjusted_confidence for a in entity_assessments)
        else:
            overall_confidence = 0.0
        
        # Determine overall quality
        if overall_confidence >= 0.8 and high_confidence_entities / total_entities >= 0.8:
            overall_quality = QualityFlag.EXCELLENT
        elif overall_confidence >= 0.7 and high_confidence_entities / total_entities >= 0.7:
            overall_quality = QualityFlag.GOOD
        elif overall_confidence >= 0.5 and high_confidence_entities / total_entities >= 0.5:
            overall_quality = QualityFlag.QUESTIONABLE
        elif len(flagged_entities) / total_entities > 0.5:
            overall_quality = QualityFlag.SUSPICIOUS
        else:
            overall_quality = QualityFlag.POOR
        
        # Determine if human review is needed
        requires_human_review = (
            overall_quality in [QualityFlag.QUESTIONABLE, QualityFlag.POOR, QualityFlag.SUSPICIOUS] or
            len(flagged_entities) > 0 or
            overall_confidence < self.min_confidence_threshold
        )
        
        # Generate recommendations
        recommendations = self._generate_overall_recommendations(
            entity_assessments, overall_quality, classification
        )
        
        return QualityAssessment(
            overall_quality=overall_quality,
            confidence_distribution=confidence_distribution,
            total_entities=total_entities,
            high_confidence_entities=high_confidence_entities,
            flagged_entities=flagged_entities,
            validation_summary=validation_summary,
            recommendations=recommendations,
            requires_human_review=requires_human_review,
            confidence_score=overall_confidence
        )
    
    def _generate_overall_recommendations(self,
                                        entity_assessments: List[EntityAssessment],
                                        overall_quality: QualityFlag,
                                        classification: Optional[DocumentClassification]) -> List[str]:
        """Generate overall recommendations"""
        recommendations = []
        
        if overall_quality == QualityFlag.EXCELLENT:
            recommendations.append("Extraction quality is excellent - results can be used with confidence")
        
        elif overall_quality == QualityFlag.GOOD:
            recommendations.append("Good extraction quality - minor validation recommended")
        
        elif overall_quality == QualityFlag.QUESTIONABLE:
            recommendations.append("Moderate extraction quality - human review recommended")
            recommendations.append("Consider using higher-accuracy models or additional validation")
        
        elif overall_quality == QualityFlag.POOR:
            recommendations.append("Poor extraction quality - significant issues detected")
            recommendations.append("Re-run extraction with different model or parameters")
            recommendations.append("Manual verification strongly recommended")
        
        elif overall_quality == QualityFlag.SUSPICIOUS:
            recommendations.append("Suspicious extraction results - manual review required")
            recommendations.append("Multiple validation failures detected")
        
        # Add specific recommendations based on common issues
        common_flags = Counter()
        for assessment in entity_assessments:
            common_flags.update(assessment.flags)
        
        most_common_flags = common_flags.most_common(3)
        for flag, count in most_common_flags:
            if count > len(entity_assessments) * 0.3:  # More than 30% of entities
                if flag == "low_model_confidence":
                    recommendations.append("Consider using a more accurate model")
                elif flag == "format_validation_failed":
                    recommendations.append("Review and improve format validation patterns")
                elif flag == "poor_context_consistency":
                    recommendations.append("Verify document transcription quality")
        
        return recommendations
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to confidence level"""
        if confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif confidence < 0.5:
            return ConfidenceLevel.LOW
        elif confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def get_assessor_statistics(self) -> Dict[str, Any]:
        """Get assessor statistics and configuration"""
        return {
            'assessor_name': 'ConfidenceAssessor',
            'configuration': {
                'min_confidence_threshold': self.min_confidence_threshold,
                'cross_validation_enabled': self.enable_cross_validation,
                'format_validation_enabled': self.enable_format_validation,
                'context_analysis_enabled': self.enable_context_analysis
            },
            'supported_pii_types': list(self.format_patterns.keys()),
            'confidence_levels': [level.value for level in ConfidenceLevel],
            'quality_flags': [flag.value for flag in QualityFlag],
            'validation_results': [result.value for result in ValidationResult]
        }