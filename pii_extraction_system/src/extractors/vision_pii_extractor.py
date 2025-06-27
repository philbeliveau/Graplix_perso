"""
Vision-based PII Extractor with Advanced Confidence Scoring and Role-Aware Filtering

This module provides the main Vision-LLM PII extraction capabilities with
confidence scoring, role-aware filtering, and intelligent model routing.
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .base import PIIExtractorBase, PIIExtractionResult, PIIEntity
from .vision_document_classifier import VisionDocumentClassifier, DocumentClassification
from .prompt_router import PromptRouter, RoutingStrategy, RoutingDecision
from ..llm.multimodal_llm_service import llm_service

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """User roles for PII filtering"""
    HR_MANAGER = "hr_manager"
    FINANCE_MANAGER = "finance_manager"
    LEGAL_COUNSEL = "legal_counsel"
    COMPLIANCE_OFFICER = "compliance_officer"
    DATA_ANALYST = "data_analyst"
    GENERAL_USER = "general_user"
    ADMIN = "admin"


class PIICategory(Enum):
    """PII categories for role-based filtering"""
    IDENTITY = "identity"              # Names, IDs, SSN
    CONTACT = "contact"                # Email, phone, address
    FINANCIAL = "financial"           # Bank accounts, credit cards
    MEDICAL = "medical"                # Health information, medical IDs
    EMPLOYMENT = "employment"          # Employee IDs, salary, benefits
    LEGAL = "legal"                    # Legal documents, case numbers
    EDUCATION = "education"            # Student IDs, grades, transcripts
    GOVERNMENT = "government"          # Government IDs, permits, licenses


@dataclass
class PIIFilterRule:
    """Rule for filtering PII based on user role"""
    role: UserRole
    allowed_categories: Set[PIICategory]
    confidence_threshold: float = 0.7
    require_explicit_consent: bool = False
    redaction_level: str = "partial"  # "none", "partial", "full"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ExtractionContext:
    """Context for PII extraction"""
    user_role: UserRole
    purpose: str
    compliance_requirements: List[str] = field(default_factory=list)
    max_processing_time: Optional[float] = None
    max_cost: Optional[float] = None
    min_confidence: float = 0.7
    enable_quality_checks: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisionPIIExtractor(PIIExtractorBase):
    """
    Advanced vision-based PII extractor with confidence scoring,
    role-aware filtering, and intelligent model routing.
    """
    
    def __init__(self,
                 enable_classification: bool = True,
                 enable_routing: bool = True,
                 default_routing_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
                 confidence_threshold: float = 0.7,
                 max_retries: int = 2):
        """
        Initialize the Vision PII Extractor
        
        Args:
            enable_classification: Enable document classification
            enable_routing: Enable intelligent model routing
            default_routing_strategy: Default routing strategy
            confidence_threshold: Minimum confidence threshold
            max_retries: Maximum retry attempts for failed extractions
        """
        super().__init__("vision_pii_extractor")
        
        self.enable_classification = enable_classification
        self.enable_routing = enable_routing
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        
        # Initialize components
        if enable_classification:
            self.classifier = VisionDocumentClassifier()
        else:
            self.classifier = None
            
        if enable_routing:
            self.router = PromptRouter(default_strategy=default_routing_strategy)
        else:
            self.router = None
        
        self.llm_service = llm_service
        
        # Role-based filtering rules
        self._initialize_filtering_rules()
        
        # PII type mappings for standardization
        self.pii_type_mappings = {
            'names': 'PERSON',
            'emails': 'EMAIL',
            'phone_numbers': 'PHONE', 
            'addresses': 'ADDRESS',
            'dates': 'DATE',
            'identification_numbers': 'ID_NUMBER',
            'organizations': 'ORGANIZATION',
            'bank_accounts': 'BANK_ACCOUNT',
            'credit_cards': 'CREDIT_CARD',
            'ssn': 'SSN',
            'driver_license': 'DRIVER_LICENSE',
            'medical_records': 'MEDICAL_RECORD'
        }
        
        # Category mappings
        self.category_mappings = {
            'PERSON': PIICategory.IDENTITY,
            'EMAIL': PIICategory.CONTACT,
            'PHONE': PIICategory.CONTACT,
            'ADDRESS': PIICategory.CONTACT,
            'ID_NUMBER': PIICategory.IDENTITY,
            'SSN': PIICategory.IDENTITY,
            'BANK_ACCOUNT': PIICategory.FINANCIAL,
            'CREDIT_CARD': PIICategory.FINANCIAL,
            'MEDICAL_RECORD': PIICategory.MEDICAL,
            'EMPLOYEE_ID': PIICategory.EMPLOYMENT,
            'ORGANIZATION': PIICategory.IDENTITY
        }
        
        logger.info(f"VisionPIIExtractor initialized - Classification: {enable_classification}, "
                   f"Routing: {enable_routing}, Threshold: {confidence_threshold}")
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """
        Extract PII from document using vision-LLM models
        
        Args:
            document: Processed document containing image data and metadata
            
        Returns:
            PIIExtractionResult with extracted PII entities
        """
        start_time = time.time()
        
        try:
            # Extract context from document
            context = self._extract_context(document)
            
            # Step 1: Document Classification (if enabled)
            classification = None
            if self.enable_classification and self.classifier:
                classification_result = self.classifier.extract_pii(document)
                if classification_result.metadata and 'classification' in classification_result.metadata:
                    classification_data = classification_result.metadata['classification']
                    if classification_data:
                        classification = DocumentClassification(
                            difficulty=classification_data['difficulty'],
                            domain=classification_data['domain'],
                            domain_detail=classification_data['domain_detail'],
                            confidence=classification_data['confidence'],
                            processing_time=classification_data['processing_time'],
                            llm_model_used=classification_data['llm_model_used']
                        )
                        logger.info(f"Document classified as {classification.domain.value} "
                                   f"({classification.difficulty.value}) with confidence {classification.confidence:.3f}")
            
            # Step 2: Model Routing (if enabled)
            routing_decision = None
            selected_model = "gpt-4o-mini"  # Default fallback
            
            if self.enable_routing and self.router:
                routing_decision = self.router.route_request(
                    classification=classification,
                    strategy=self._get_routing_strategy(context, classification),
                    min_confidence=context.min_confidence,
                    max_cost=context.max_cost,
                    max_time=context.max_processing_time
                )
                selected_model = routing_decision.selected_model
                logger.info(f"Routed to model: {selected_model} (reason: {routing_decision.reason})")
            
            # Step 3: PII Extraction
            extraction_result = self._extract_with_model(
                document=document,
                model_key=selected_model,
                classification=classification,
                context=context
            )
            
            # Step 4: Post-processing and filtering
            if extraction_result.pii_entities:
                # Apply role-based filtering
                filtered_entities = self._apply_role_based_filtering(
                    entities=extraction_result.pii_entities,
                    context=context
                )
                
                # Apply confidence filtering
                high_confidence_entities = self._apply_confidence_filtering(
                    entities=filtered_entities,
                    threshold=context.min_confidence
                )
                
                # Quality checks
                if context.enable_quality_checks:
                    validated_entities = self._apply_quality_checks(high_confidence_entities)
                else:
                    validated_entities = high_confidence_entities
                
                extraction_result.pii_entities = validated_entities
            
            # Step 5: Record performance metrics (if routing enabled)
            if self.enable_routing and self.router and routing_decision:
                processing_time = time.time() - start_time
                avg_confidence = sum(e.confidence for e in extraction_result.pii_entities) / len(extraction_result.pii_entities) if extraction_result.pii_entities else 0.0
                estimated_cost = extraction_result.metadata.get('usage', {}).get('estimated_cost', 0.0)
                
                self.router.record_result(
                    model_key=selected_model,
                    processing_time=processing_time,
                    confidence=avg_confidence,
                    cost=estimated_cost,
                    success=extraction_result.error is None,
                    classification=classification
                )
            
            # Update metadata
            extraction_result.metadata.update({
                'classification': classification.to_dict() if classification else None,
                'routing_decision': routing_decision.__dict__ if routing_decision else None,
                'context': {
                    'user_role': context.user_role.value,
                    'purpose': context.purpose,
                    'compliance_requirements': context.compliance_requirements
                },
                'filtering_applied': True,
                'quality_checks_applied': context.enable_quality_checks,
                'total_processing_time': time.time() - start_time
            })
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"Vision PII extraction error: {e}")
            return PIIExtractionResult(
                pii_entities=[],
                confidence_scores=[],
                processing_time=time.time() - start_time,
                error=str(e),
                metadata={'extraction_failed': True}
            )
    
    def _extract_with_model(self,
                          document: Dict[str, Any],
                          model_key: str,
                          classification: Optional[DocumentClassification],
                          context: ExtractionContext) -> PIIExtractionResult:
        """Extract PII using specified model with retries"""
        
        # Get image data
        image_data = self._get_image_data(document)
        if not image_data:
            return PIIExtractionResult(
                pii_entities=[],
                confidence_scores=[],
                processing_time=0.0,
                error="No image data found in document"
            )
        
        # Create domain-specific prompt
        document_type = classification.domain.value if classification else "document"
        
        # Try extraction with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Extraction attempt {attempt + 1} with {model_key}")
                
                # Perform extraction
                result = self.llm_service.extract_pii_from_image(
                    image_data=image_data,
                    model_key=model_key,
                    document_type=document_type,
                    max_tokens=4000,
                    temperature=0.1
                )
                
                if result.get('success'):
                    # Convert to PIIExtractionResult
                    return self._convert_llm_result_to_extraction_result(result, model_key)
                else:
                    last_error = result.get('error', 'Unknown error')
                    logger.warning(f"Extraction attempt {attempt + 1} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.error(f"Extraction attempt {attempt + 1} error: {e}")
        
        # All attempts failed
        return PIIExtractionResult(
            pii_entities=[],
            confidence_scores=[],
            processing_time=0.0,
            error=f"All extraction attempts failed. Last error: {last_error}"
        )
    
    def _convert_llm_result_to_extraction_result(self, 
                                               llm_result: Dict[str, Any],
                                               model_key: str) -> PIIExtractionResult:
        """Convert LLM service result to PIIExtractionResult"""
        
        pii_entities = []
        confidence_scores = []
        
        # Process structured LLM entities
        if 'pii_entities' in llm_result:
            for entity_data in llm_result['pii_entities']:
                entity = PIIEntity(
                    text=entity_data.get('text', ''),
                    pii_type=entity_data.get('type', 'UNKNOWN'),
                    confidence=entity_data.get('confidence', 0.5),
                    start_pos=entity_data.get('start_pos', 0),
                    end_pos=entity_data.get('end_pos', 0),
                    context=entity_data.get('context', ''),
                    extractor=self.name,
                    metadata={
                        'model_used': model_key,
                        'source': entity_data.get('source', 'llm_extraction'),
                        'category': self._get_pii_category(entity_data.get('type', 'UNKNOWN'))
                    }
                )
                pii_entities.append(entity)
                confidence_scores.append(entity.confidence)
        
        # If no structured entities, try to extract from parsed data
        elif 'structured_data' in llm_result and llm_result['structured_data']:
            pii_entities, confidence_scores = self._extract_entities_from_structured_data(
                llm_result['structured_data'], model_key
            )
        
        return PIIExtractionResult(
            pii_entities=pii_entities,
            confidence_scores=confidence_scores,
            processing_time=llm_result.get('processing_time', 0.0),
            metadata={
                'model_used': model_key,
                'extraction_method': llm_result.get('extraction_method', 'vision_llm'),
                'transcribed_text': llm_result.get('transcribed_text', ''),
                'usage': llm_result.get('usage', {}),
                'total_entities_found': len(pii_entities)
            }
        )
    
    def _extract_entities_from_structured_data(self, 
                                             structured_data: Dict[str, Any],
                                             model_key: str) -> Tuple[List[PIIEntity], List[float]]:
        """Extract PIIEntity objects from structured LLM response"""
        
        entities = []
        confidence_scores = []
        
        extracted_info = structured_data.get('extracted_information', {})
        
        # Process each PII type
        for info_type, values in extracted_info.items():
            if not values:
                continue
            
            # Map to standard PII type
            pii_type = self.pii_type_mappings.get(info_type, info_type.upper())
            
            # Handle nested structures (like contact_info)
            if isinstance(values, dict):
                for sub_type, sub_values in values.items():
                    if isinstance(sub_values, list):
                        for value in sub_values:
                            if value and str(value).strip():
                                entity = self._create_pii_entity(
                                    text=str(value).strip(),
                                    pii_type=self.pii_type_mappings.get(sub_type, sub_type.upper()),
                                    model_key=model_key,
                                    info_type=sub_type
                                )
                                entities.append(entity)
                                confidence_scores.append(entity.confidence)
            
            # Handle direct lists
            elif isinstance(values, list):
                for value in values:
                    if value and str(value).strip():
                        entity = self._create_pii_entity(
                            text=str(value).strip(),
                            pii_type=pii_type,
                            model_key=model_key,
                            info_type=info_type
                        )
                        entities.append(entity)
                        confidence_scores.append(entity.confidence)
        
        return entities, confidence_scores
    
    def _create_pii_entity(self, 
                          text: str,
                          pii_type: str,
                          model_key: str,
                          info_type: str) -> PIIEntity:
        """Create a PIIEntity with appropriate confidence and metadata"""
        
        # Base confidence scores by PII type
        base_confidence = {
            'EMAIL': 0.95,
            'PHONE': 0.90,
            'PERSON': 0.85,
            'ADDRESS': 0.80,
            'ID_NUMBER': 0.90,
            'ORGANIZATION': 0.75,
            'DATE': 0.70
        }
        
        confidence = base_confidence.get(pii_type, 0.70)
        
        # Adjust confidence based on text characteristics
        confidence = self._adjust_confidence_by_content(text, pii_type, confidence)
        
        return PIIEntity(
            text=text,
            pii_type=pii_type,
            confidence=confidence,
            extractor=self.name,
            metadata={
                'model_used': model_key,
                'original_type': info_type,
                'category': self._get_pii_category(pii_type),
                'confidence_adjusted': True
            }
        )
    
    def _adjust_confidence_by_content(self, text: str, pii_type: str, base_confidence: float) -> float:
        """Adjust confidence based on content analysis"""
        
        if pii_type == 'EMAIL':
            # Email validation patterns
            if '@' in text and '.' in text:
                return min(0.98, base_confidence + 0.1)
            else:
                return max(0.3, base_confidence - 0.3)
        
        elif pii_type == 'PHONE':
            # Phone number patterns
            digits = ''.join(c for c in text if c.isdigit())
            if len(digits) >= 10:
                return min(0.95, base_confidence + 0.05)
            else:
                return max(0.4, base_confidence - 0.2)
        
        elif pii_type == 'PERSON':
            # Name validation
            if len(text.split()) >= 2 and text.replace(' ', '').isalpha():
                return min(0.90, base_confidence + 0.05)
            else:
                return max(0.5, base_confidence - 0.15)
        
        return base_confidence
    
    def _apply_role_based_filtering(self, 
                                  entities: List[PIIEntity],
                                  context: ExtractionContext) -> List[PIIEntity]:
        """Apply role-based filtering to PII entities"""
        
        if context.user_role not in self.filtering_rules:
            logger.warning(f"No filtering rules for role {context.user_role.value}, allowing all")
            return entities
        
        rule = self.filtering_rules[context.user_role]
        filtered_entities = []
        
        for entity in entities:
            entity_category = self._get_pii_category(entity.pii_type)
            
            if entity_category in rule.allowed_categories:
                # Apply confidence threshold
                if entity.confidence >= rule.confidence_threshold:
                    # Apply redaction if needed
                    if rule.redaction_level != "none":
                        entity = self._apply_redaction(entity, rule.redaction_level)
                    
                    filtered_entities.append(entity)
                else:
                    logger.debug(f"Filtered out {entity.pii_type} due to low confidence: {entity.confidence}")
            else:
                logger.debug(f"Filtered out {entity.pii_type} due to role restrictions")
        
        logger.info(f"Role-based filtering: {len(entities)} -> {len(filtered_entities)} entities")
        return filtered_entities
    
    def _apply_confidence_filtering(self,
                                  entities: List[PIIEntity],
                                  threshold: float) -> List[PIIEntity]:
        """Filter entities by confidence threshold"""
        
        filtered = [e for e in entities if e.confidence >= threshold]
        
        if len(filtered) != len(entities):
            logger.info(f"Confidence filtering: {len(entities)} -> {len(filtered)} entities "
                       f"(threshold: {threshold})")
        
        return filtered
    
    def _apply_quality_checks(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Apply quality checks to filter out low-quality extractions"""
        
        validated_entities = []
        
        for entity in entities:
            if self._validate_entity_quality(entity):
                validated_entities.append(entity)
            else:
                logger.debug(f"Quality check failed for {entity.pii_type}: {entity.text}")
        
        if len(validated_entities) != len(entities):
            logger.info(f"Quality checks: {len(entities)} -> {len(validated_entities)} entities")
        
        return validated_entities
    
    def _validate_entity_quality(self, entity: PIIEntity) -> bool:
        """Validate individual entity quality"""
        
        # Basic text validation
        if not entity.text or len(entity.text.strip()) < 2:
            return False
        
        # Type-specific validation
        if entity.pii_type == 'EMAIL':
            return '@' in entity.text and '.' in entity.text
        
        elif entity.pii_type == 'PHONE':
            digits = ''.join(c for c in entity.text if c.isdigit())
            return len(digits) >= 10
        
        elif entity.pii_type == 'PERSON':
            # Names should have at least 2 parts and be mostly alphabetic
            parts = entity.text.split()
            return len(parts) >= 2 and sum(1 for p in parts if p.isalpha()) >= len(parts) * 0.8
        
        return True
    
    def _apply_redaction(self, entity: PIIEntity, redaction_level: str) -> PIIEntity:
        """Apply redaction to PII entity"""
        
        if redaction_level == "partial":
            if entity.pii_type == 'EMAIL':
                # Redact middle part of email
                parts = entity.text.split('@')
                if len(parts) == 2:
                    name = parts[0]
                    domain = parts[1]
                    if len(name) > 2:
                        redacted_name = name[0] + '*' * (len(name) - 2) + name[-1]
                        entity.text = f"{redacted_name}@{domain}"
            
            elif entity.pii_type == 'PHONE':
                # Redact middle digits
                digits = ''.join(c for c in entity.text if c.isdigit())
                if len(digits) >= 10:
                    entity.text = entity.text.replace(digits[3:7], '****')
            
            elif entity.pii_type == 'PERSON':
                # Redact last name
                parts = entity.text.split()
                if len(parts) >= 2:
                    parts[-1] = parts[-1][0] + '*' * (len(parts[-1]) - 1)
                    entity.text = ' '.join(parts)
        
        elif redaction_level == "full":
            entity.text = f"[REDACTED_{entity.pii_type}]"
        
        # Mark as redacted
        entity.metadata['redacted'] = True
        entity.metadata['redaction_level'] = redaction_level
        
        return entity
    
    def _get_pii_category(self, pii_type: str) -> PIICategory:
        """Get PII category for a given PII type"""
        return self.category_mappings.get(pii_type, PIICategory.IDENTITY)
    
    def _get_image_data(self, document: Dict[str, Any]) -> Optional[str]:
        """Extract image data from document"""
        # Try different possible keys
        for key in ['image_data', 'base64_image', 'image', 'content']:
            if key in document and document[key]:
                return document[key]
        
        # Try to load from file path
        if 'file_path' in document:
            try:
                import base64
                from PIL import Image
                import io
                
                with Image.open(document['file_path']) as img:
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    return base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as e:
                logger.error(f"Failed to load image: {e}")
        
        return None
    
    def _extract_context(self, document: Dict[str, Any]) -> ExtractionContext:
        """Extract extraction context from document metadata"""
        
        metadata = document.get('metadata', {})
        
        # Extract user role
        user_role_str = metadata.get('user_role', 'general_user')
        try:
            user_role = UserRole(user_role_str)
        except ValueError:
            user_role = UserRole.GENERAL_USER
            logger.warning(f"Unknown user role '{user_role_str}', using default")
        
        return ExtractionContext(
            user_role=user_role,
            purpose=metadata.get('purpose', 'pii_extraction'),
            compliance_requirements=metadata.get('compliance_requirements', []),
            max_processing_time=metadata.get('max_processing_time'),
            max_cost=metadata.get('max_cost'),
            min_confidence=metadata.get('min_confidence', self.confidence_threshold),
            enable_quality_checks=metadata.get('enable_quality_checks', True),
            metadata=metadata
        )
    
    def _get_routing_strategy(self,
                            context: ExtractionContext,
                            classification: Optional[DocumentClassification]) -> RoutingStrategy:
        """Get routing strategy based on context and classification"""
        
        # High-stakes domains require accuracy-first routing
        if classification and classification.domain.value in ['Medical', 'Legal', 'Government']:
            return RoutingStrategy.PERFORMANCE_FIRST
        
        # Cost-sensitive contexts
        if context.max_cost and context.max_cost < 0.01:
            return RoutingStrategy.COST_FIRST
        
        # Time-sensitive contexts
        if context.max_processing_time and context.max_processing_time < 10:
            return RoutingStrategy.SPEED_FIRST
        
        # Default to balanced
        return RoutingStrategy.BALANCED
    
    def _initialize_filtering_rules(self):
        """Initialize role-based filtering rules"""
        
        self.filtering_rules = {
            UserRole.HR_MANAGER: PIIFilterRule(
                role=UserRole.HR_MANAGER,
                allowed_categories={PIICategory.IDENTITY, PIICategory.CONTACT, PIICategory.EMPLOYMENT},
                confidence_threshold=0.75,
                redaction_level="partial"
            ),
            
            UserRole.FINANCE_MANAGER: PIIFilterRule(
                role=UserRole.FINANCE_MANAGER,
                allowed_categories={PIICategory.IDENTITY, PIICategory.CONTACT, PIICategory.FINANCIAL, PIICategory.EMPLOYMENT},
                confidence_threshold=0.80,
                redaction_level="partial"
            ),
            
            UserRole.LEGAL_COUNSEL: PIIFilterRule(
                role=UserRole.LEGAL_COUNSEL,
                allowed_categories={PIICategory.IDENTITY, PIICategory.CONTACT, PIICategory.LEGAL, PIICategory.FINANCIAL},
                confidence_threshold=0.85,
                redaction_level="none"
            ),
            
            UserRole.COMPLIANCE_OFFICER: PIIFilterRule(
                role=UserRole.COMPLIANCE_OFFICER,
                allowed_categories=set(PIICategory),  # All categories
                confidence_threshold=0.90,
                redaction_level="none",
                require_explicit_consent=True
            ),
            
            UserRole.DATA_ANALYST: PIIFilterRule(
                role=UserRole.DATA_ANALYST,
                allowed_categories={PIICategory.IDENTITY, PIICategory.CONTACT},
                confidence_threshold=0.70,
                redaction_level="full"
            ),
            
            UserRole.GENERAL_USER: PIIFilterRule(
                role=UserRole.GENERAL_USER,
                allowed_categories={PIICategory.CONTACT},
                confidence_threshold=0.80,
                redaction_level="partial"
            ),
            
            UserRole.ADMIN: PIIFilterRule(
                role=UserRole.ADMIN,
                allowed_categories=set(PIICategory),  # All categories
                confidence_threshold=0.70,
                redaction_level="none"
            )
        }
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics for monitoring"""
        
        stats = {
            'extractor_name': self.name,
            'configuration': {
                'classification_enabled': self.enable_classification,
                'routing_enabled': self.enable_routing,
                'confidence_threshold': self.confidence_threshold,
                'max_retries': self.max_retries
            },
            'supported_roles': [role.value for role in UserRole],
            'supported_categories': [cat.value for cat in PIICategory],
            'pii_type_mappings': self.pii_type_mappings
        }
        
        if self.router:
            stats['routing_statistics'] = self.router.get_routing_statistics()
        
        return stats