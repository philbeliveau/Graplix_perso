"""
Vision-based Document Classifier for PII Extraction Pipeline

This module provides advanced document classification using vision-LLM models
to determine document type, domain, and difficulty level for optimized PII extraction.
"""

import os
import json
import time
import base64
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

from .base import PIIExtractorBase, PIIExtractionResult
from ..llm.multimodal_llm_service import llm_service

logger = logging.getLogger(__name__)


class DocumentDifficulty(Enum):
    """Document processing difficulty levels"""
    EASY = "Easy"
    MEDIUM = "Medium"  
    HARD = "Hard"


class DocumentDomain(Enum):
    """Document domain classifications"""
    HR = "HR"
    FINANCE = "Finance"
    LEGAL = "Legal"
    MEDICAL = "Medical"
    GOVERNMENT = "Government"
    EDUCATION = "Education"
    INSURANCE = "Insurance"
    REAL_ESTATE = "Real Estate"
    OTHER = "Other"


@dataclass
class DocumentClassification:
    """Document classification result"""
    difficulty: DocumentDifficulty
    domain: DocumentDomain
    domain_detail: str
    confidence: float
    processing_time: float
    llm_model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'difficulty': self.difficulty.value,
            'domain': self.domain.value,
            'domain_detail': self.domain_detail,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'llm_model_used': self.llm_model_used,
            'metadata': self.metadata
        }


class VisionDocumentClassifier(PIIExtractorBase):
    """
    Vision-based document classifier using LLM models to determine
    document type, domain, and processing difficulty for optimized PII extraction.
    """
    
    def __init__(self, 
                 preferred_models: Optional[List[str]] = None,
                 confidence_threshold: float = 0.7,
                 max_retries: int = 2):
        """
        Initialize the Vision Document Classifier
        
        Args:
            preferred_models: List of preferred LLM models for classification
            confidence_threshold: Minimum confidence threshold for classification
            max_retries: Maximum number of retry attempts for failed classifications
        """
        super().__init__("vision_document_classifier")
        
        self.preferred_models = preferred_models or [
            "gpt-4o-mini",  # Fast and cost-effective for classification
            "claude-3-5-haiku-20241022",  # Good balance of speed and accuracy
            "gemini-1.5-flash"  # Google's fast model
        ]
        
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.llm_service = llm_service
        
        # Domain-specific keywords for enhanced classification
        self.domain_keywords = {
            DocumentDomain.HR: [
                'employee', 'employment', 'payroll', 'salary', 'vacation', 'absence',
                'performance', 'review', 'onboarding', 'termination', 'benefits',
                'hr', 'human resources', 'position', 'job', 'work schedule'
            ],
            DocumentDomain.FINANCE: [
                'invoice', 'payment', 'bank', 'account', 'credit', 'debit', 'financial',
                'tax', 'receipt', 'transaction', 'budget', 'expense', 'revenue',
                'accounting', 'fiscal', 'monetary', 'cost', 'price'
            ],
            DocumentDomain.LEGAL: [
                'contract', 'agreement', 'legal', 'law', 'court', 'attorney', 'lawyer',
                'litigation', 'settlement', 'clause', 'terms', 'conditions',
                'liability', 'compliance', 'regulation', 'statute'
            ],
            DocumentDomain.MEDICAL: [
                'patient', 'medical', 'health', 'doctor', 'physician', 'hospital',
                'clinic', 'diagnosis', 'treatment', 'prescription', 'medication',
                'surgery', 'therapy', 'medical record', 'healthcare'
            ],
            DocumentDomain.GOVERNMENT: [
                'government', 'official', 'permit', 'license', 'certificate',
                'public', 'municipal', 'federal', 'provincial', 'state',
                'department', 'ministry', 'agency', 'authority'
            ],
            DocumentDomain.EDUCATION: [
                'student', 'school', 'university', 'college', 'education',
                'academic', 'transcript', 'diploma', 'degree', 'course',
                'enrollment', 'tuition', 'grade', 'teacher', 'professor'
            ],
            DocumentDomain.INSURANCE: [
                'insurance', 'policy', 'claim', 'coverage', 'premium', 'deductible',
                'beneficiary', 'underwriting', 'risk', 'adjuster'
            ],
            DocumentDomain.REAL_ESTATE: [
                'property', 'real estate', 'mortgage', 'lease', 'rent', 'landlord',
                'tenant', 'deed', 'title', 'appraisal', 'listing'
            ]
        }
        
        logger.info(f"VisionDocumentClassifier initialized with models: {self.preferred_models}")
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """
        Extract PII by first classifying the document for optimized processing
        
        Args:
            document: Processed document containing image data and metadata
            
        Returns:
            PIIExtractionResult with classification metadata
        """
        start_time = time.time()
        
        try:
            # Get image data from document
            image_data = self._extract_image_data(document)
            if not image_data:
                return PIIExtractionResult(
                    pii_entities=[],
                    confidence_scores=[],
                    processing_time=time.time() - start_time,
                    error="No image data found in document",
                    metadata={'classification': None}
                )
            
            # Classify the document
            classification = self.classify_document(image_data)
            
            # Store classification in metadata for use by other extractors
            metadata = {
                'classification': classification.to_dict(),
                'recommended_processing_strategy': self._get_processing_strategy(classification),
                'estimated_pii_density': self._estimate_pii_density(classification),
                'extraction_priority': self._get_extraction_priority(classification)
            }
            
            processing_time = time.time() - start_time
            
            return PIIExtractionResult(
                pii_entities=[],  # Classification doesn't extract PII directly
                confidence_scores=[classification.confidence],
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Document classification error: {e}")
            return PIIExtractionResult(
                pii_entities=[],
                confidence_scores=[],
                processing_time=time.time() - start_time,
                error=str(e),
                metadata={'classification': None}
            )
    
    def classify_document(self, image_data: str) -> DocumentClassification:
        """
        Classify document using vision-LLM models
        
        Args:
            image_data: Base64 encoded image data
            
        Returns:
            DocumentClassification result
        """
        start_time = time.time()
        
        # Try each preferred model in order
        for model_key in self.preferred_models:
            try:
                result = self._classify_with_model(image_data, model_key)
                if result and result.confidence >= self.confidence_threshold:
                    logger.info(f"Document classified successfully with {model_key}")
                    return result
                elif result:
                    logger.warning(f"Low confidence classification ({result.confidence}) with {model_key}")
                    
            except Exception as e:
                logger.warning(f"Classification failed with {model_key}: {e}")
                continue
        
        # Fallback classification
        logger.warning("All preferred models failed, using fallback classification")
        return DocumentClassification(
            difficulty=DocumentDifficulty.MEDIUM,
            domain=DocumentDomain.OTHER,
            domain_detail="Unable to classify - using default settings",
            confidence=0.5,
            processing_time=time.time() - start_time,
            llm_model_used="fallback",
            metadata={'fallback_used': True}
        )
    
    def _classify_with_model(self, image_data: str, model_key: str) -> Optional[DocumentClassification]:
        """
        Classify document using a specific LLM model
        
        Args:
            image_data: Base64 encoded image data
            model_key: LLM model identifier
            
        Returns:
            DocumentClassification result or None if failed
        """
        start_time = time.time()
        
        prompt = self._create_classification_prompt()
        
        try:
            # Use the multimodal LLM service for classification
            result = self.llm_service.extract_pii_from_image(
                image_data=image_data,
                model_key=model_key,
                document_type="document_for_classification",
                max_tokens=1000,
                temperature=0.1
            )
            
            if not result.get('success'):
                logger.error(f"LLM classification failed: {result.get('error')}")
                return None
            
            # Parse the classification result
            classification_data = self._parse_classification_response(result.get('content', ''))
            
            if not classification_data:
                logger.error("Failed to parse classification response")
                return None
            
            # Create classification object
            classification = DocumentClassification(
                difficulty=DocumentDifficulty(classification_data.get('difficulty', 'Medium')),
                domain=DocumentDomain(classification_data.get('domain', 'Other')),
                domain_detail=classification_data.get('domain_detail', ''),
                confidence=self._calculate_classification_confidence(classification_data),
                processing_time=time.time() - start_time,
                llm_model_used=model_key,
                metadata={
                    'raw_response': result.get('content'),
                    'usage': result.get('usage', {}),
                    'llm_processing_time': result.get('processing_time', 0)
                }
            )
            
            # Enhance confidence with keyword matching
            classification.confidence = self._enhance_confidence_with_keywords(
                classification, result.get('transcribed_text', '')
            )
            
            return classification
            
        except Exception as e:
            logger.error(f"Error in classification with {model_key}: {e}")
            return None
    
    def _create_classification_prompt(self) -> str:
        """Create optimized prompt for document classification"""
        
        return """
Analyze this document image and classify it according to the following criteria.

Return ONLY a valid JSON object with this exact structure:
{
  "difficulty_level": "Easy|Medium|Hard",
  "domain": "HR|Finance|Legal|Medical|Government|Education|Insurance|Real Estate|Other",
  "domain_detail": "specific document type or subdomain",
  "confidence_indicators": {
    "text_clarity": "Clear|Moderate|Poor",
    "layout_structure": "Simple|Standard|Complex",
    "pii_density": "Low|Medium|High",
    "document_quality": "Excellent|Good|Fair|Poor"
  },
  "processing_recommendations": {
    "recommended_ocr_approach": "standard|advanced|hybrid",
    "expected_pii_types": ["list of likely PII types"],
    "special_handling_required": true|false
  }
}

Classification Guidelines:

DIFFICULTY LEVELS:
- Easy: Clear text, standard format, minimal PII, good image quality
- Medium: Some formatting challenges, moderate PII density, acceptable quality
- Hard: Poor quality, handwriting, complex layout, dense PII, or degraded images

DOMAINS:
- HR: Employment forms, payroll, absence requests, performance reviews, job applications
- Finance: Invoices, bank statements, tax documents, financial reports, receipts
- Legal: Contracts, agreements, legal notices, court documents, compliance forms
- Medical: Health records, prescriptions, patient forms, medical reports, insurance claims
- Government: ID documents, permits, licenses, official forms, tax documents
- Education: Transcripts, enrollment forms, diplomas, student records, certificates
- Insurance: Policy documents, claim forms, coverage statements, premium notices
- Real Estate: Property documents, leases, mortgage papers, deed transfers
- Other: Documents that don't fit the above categories

Focus on:
1. Visual document structure and layout complexity
2. Text clarity and readability
3. Likely presence and density of personal information
4. Document format and organization
5. Image quality and potential OCR challenges

Return valid JSON only, no other text.
"""
    
    def _parse_classification_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM classification response"""
        try:
            # Clean up response to extract JSON
            if "```json" in response:
                json_part = response.split("```json")[1].split("```")[0]
            elif "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_part = response[start:end]
            else:
                json_part = response
            
            parsed_data = json.loads(json_part.strip())
            
            # Validate required fields
            required_fields = ['difficulty_level', 'domain', 'domain_detail']
            if not all(field in parsed_data for field in required_fields):
                logger.error(f"Missing required fields in classification response: {parsed_data}")
                return None
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification JSON: {e}")
            logger.debug(f"Raw response: {response}")
            return None
    
    def _calculate_classification_confidence(self, classification_data: Dict[str, Any]) -> float:
        """Calculate confidence score for classification"""
        base_confidence = 0.8
        
        # Adjust based on confidence indicators
        indicators = classification_data.get('confidence_indicators', {})
        
        confidence_adjustments = {
            'text_clarity': {'Clear': 0.1, 'Moderate': 0.0, 'Poor': -0.15},
            'layout_structure': {'Simple': 0.05, 'Standard': 0.0, 'Complex': -0.05},
            'document_quality': {'Excellent': 0.1, 'Good': 0.05, 'Fair': 0.0, 'Poor': -0.1}
        }
        
        for indicator, value in indicators.items():
            if indicator in confidence_adjustments and value in confidence_adjustments[indicator]:
                base_confidence += confidence_adjustments[indicator][value]
        
        return max(0.1, min(1.0, base_confidence))
    
    def _enhance_confidence_with_keywords(self, 
                                        classification: DocumentClassification, 
                                        transcribed_text: str) -> float:
        """Enhance classification confidence using keyword matching"""
        if not transcribed_text:
            return classification.confidence
        
        text_lower = transcribed_text.lower()
        domain_keywords = self.domain_keywords.get(classification.domain, [])
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in domain_keywords if keyword in text_lower)
        
        if keyword_matches > 0:
            # Boost confidence based on keyword matches
            keyword_boost = min(0.2, keyword_matches * 0.05)
            enhanced_confidence = min(1.0, classification.confidence + keyword_boost)
            
            logger.debug(f"Enhanced confidence from {classification.confidence} to {enhanced_confidence} "
                        f"based on {keyword_matches} keyword matches")
            
            return enhanced_confidence
        
        return classification.confidence
    
    def _extract_image_data(self, document: Dict[str, Any]) -> Optional[str]:
        """Extract base64 image data from document"""
        # Try different possible locations for image data
        possible_keys = ['image_data', 'base64_image', 'image', 'content']
        
        for key in possible_keys:
            if key in document and document[key]:
                return document[key]
        
        # If no direct image data, check if we have a file path
        if 'file_path' in document:
            try:
                from PIL import Image
                import io
                
                # Load and convert image to base64
                with Image.open(document['file_path']) as img:
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    return img_data
                    
            except Exception as e:
                logger.error(f"Failed to load image from path {document.get('file_path')}: {e}")
        
        return None
    
    def _get_processing_strategy(self, classification: DocumentClassification) -> Dict[str, Any]:
        """Get recommended processing strategy based on classification"""
        strategies = {
            DocumentDifficulty.EASY: {
                'ocr_approach': 'standard',
                'llm_model_preference': 'fast',
                'confidence_threshold': 0.8,
                'retry_count': 1
            },
            DocumentDifficulty.MEDIUM: {
                'ocr_approach': 'hybrid',
                'llm_model_preference': 'balanced',
                'confidence_threshold': 0.7,
                'retry_count': 2
            },
            DocumentDifficulty.HARD: {
                'ocr_approach': 'advanced',
                'llm_model_preference': 'accuracy',
                'confidence_threshold': 0.6,
                'retry_count': 3
            }
        }
        
        base_strategy = strategies.get(classification.difficulty, strategies[DocumentDifficulty.MEDIUM])
        
        # Domain-specific adjustments
        domain_adjustments = {
            DocumentDomain.MEDICAL: {'privacy_level': 'high', 'accuracy_priority': True},
            DocumentDomain.LEGAL: {'accuracy_priority': True, 'confidence_threshold': 0.8},
            DocumentDomain.FINANCE: {'accuracy_priority': True, 'fraud_detection': True},
            DocumentDomain.GOVERNMENT: {'privacy_level': 'high', 'compliance_strict': True}
        }
        
        if classification.domain in domain_adjustments:
            base_strategy.update(domain_adjustments[classification.domain])
        
        return base_strategy
    
    def _estimate_pii_density(self, classification: DocumentClassification) -> str:
        """Estimate PII density based on classification"""
        high_pii_domains = [DocumentDomain.HR, DocumentDomain.MEDICAL, DocumentDomain.GOVERNMENT]
        medium_pii_domains = [DocumentDomain.FINANCE, DocumentDomain.LEGAL, DocumentDomain.EDUCATION]
        
        if classification.domain in high_pii_domains:
            return "high"
        elif classification.domain in medium_pii_domains:
            return "medium"
        else:
            return "low"
    
    def _get_extraction_priority(self, classification: DocumentClassification) -> int:
        """Get extraction priority (1-10, higher is more urgent)"""
        priority_map = {
            DocumentDomain.MEDICAL: 9,
            DocumentDomain.GOVERNMENT: 8,
            DocumentDomain.LEGAL: 7,
            DocumentDomain.FINANCE: 6,
            DocumentDomain.HR: 5,
            DocumentDomain.EDUCATION: 4,
            DocumentDomain.INSURANCE: 4,
            DocumentDomain.REAL_ESTATE: 3,
            DocumentDomain.OTHER: 2
        }
        
        base_priority = priority_map.get(classification.domain, 5)
        
        # Adjust based on difficulty
        if classification.difficulty == DocumentDifficulty.HARD:
            base_priority += 2
        elif classification.difficulty == DocumentDifficulty.EASY:
            base_priority -= 1
        
        return max(1, min(10, base_priority))
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get classification statistics for monitoring"""
        return {
            'classifier_name': self.name,
            'preferred_models': self.preferred_models,
            'confidence_threshold': self.confidence_threshold,
            'supported_domains': [domain.value for domain in DocumentDomain],
            'supported_difficulties': [diff.value for diff in DocumentDifficulty]
        }