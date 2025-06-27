"""
Contextual PII Pipeline - 3-Step Vision-LLM Processing
Solves the core problem: Extract data subject PII while ignoring professional PII

🧩 The Real Problem:
Extract and isolate the PII of the data subject (user whose consent is needed)
while ignoring PII of others (e.g., doctors, lawyers, witnesses)

🎯 Implementation:
Step 1: Document Classification (domain detection)
Step 2: Prompt Routing & Confidence Assessment  
Step 3: PII Extraction + Role Filtering + Flagging
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import base64
from pathlib import Path

# Import existing LLM service
import sys
sys.path.append('../')
from llm.multimodal_llm_service import MultimodalLLMService

logger = logging.getLogger(__name__)


class DocumentDomain(Enum):
    """Document domain classifications"""
    HEALTH = "health"
    HR = "hr" 
    FINANCIAL = "financial"
    LEGAL = "legal"
    DEATH_CERTIFICATE = "death_cert"
    GOVERNMENT = "government"
    EDUCATION = "education"
    UNKNOWN = "unknown"


class ConfidenceLevel(Enum):
    """Confidence levels for classification and extraction"""
    HIGH = "high"      # ≥ 0.85
    MEDIUM = "medium"  # 0.65-0.84
    LOW = "low"        # 0.45-0.64
    VERY_LOW = "very_low"  # < 0.45


@dataclass
class PIIEntity:
    """Enhanced PII entity with role context"""
    type: str  # person_name, phone, email, etc.
    value: str  # actual PII value
    confidence: float  # extraction confidence
    role: str  # "data_subject", "professional", "witness", "unknown"
    context: str  # surrounding text context
    position: Optional[Dict] = None  # document position
    flags: List[str] = None  # any warning flags
    
    def __post_init__(self):
        if self.flags is None:
            self.flags = []


@dataclass
class ClassificationResult:
    """Document classification result"""
    domain: DocumentDomain
    confidence: float
    reasoning: str
    secondary_domains: List[Tuple[DocumentDomain, float]] = None
    
    def __post_init__(self):
        if self.secondary_domains is None:
            self.secondary_domains = []


@dataclass
class ExtractionResult:
    """Complete PII extraction result"""
    document_id: str
    classification: ClassificationResult
    all_entities: List[PIIEntity]  # All found PII
    data_subject_entities: List[PIIEntity]  # Only data subject PII
    professional_entities: List[PIIEntity]  # Only professional PII
    confidence_flags: List[str]
    processing_metadata: Dict[str, Any]


class ContextualPIIPipeline:
    """
    3-Step Vision-LLM Pipeline for Contextual PII Extraction
    """
    
    def __init__(self, test_mode=False):
        """Initialize the pipeline with LLM service"""
        self.test_mode = test_mode
        if not test_mode:
            self.llm_service = MultimodalLLMService()
        else:
            self.llm_service = None
        
        # Confidence thresholds for routing decisions
        self.confidence_thresholds = {
            "high": 0.85,      # Use domain-specific prompt
            "medium": 0.65,    # Use general prompt with hints
            "low": 0.45,       # Use general prompt + flag
            "very_low": 0.45   # Flag for manual review
        }
        
        # Domain-specific role patterns
        self.role_patterns = {
            DocumentDomain.HEALTH: {
                "data_subject_indicators": ["patient", "client", "individual"],
                "professional_indicators": ["doctor", "dr.", "physician", "nurse", "therapist", "psychiatrist", "cardiologist", "md", "rn"],
                "context_keywords": ["medical", "diagnosis", "treatment", "symptoms", "prescription"]
            },
            DocumentDomain.LEGAL: {
                "data_subject_indicators": ["client", "plaintiff", "defendant", "party"],
                "professional_indicators": ["attorney", "lawyer", "counsel", "esq", "judge", "clerk"],
                "context_keywords": ["legal", "court", "case", "lawsuit", "contract", "agreement"]
            },
            DocumentDomain.HR: {
                "data_subject_indicators": ["employee", "candidate", "applicant", "worker"],
                "professional_indicators": ["hr", "manager", "supervisor", "recruiter", "director"],
                "context_keywords": ["employment", "job", "position", "salary", "benefits", "performance"]
            },
            DocumentDomain.FINANCIAL: {
                "data_subject_indicators": ["client", "account holder", "customer", "borrower"],
                "professional_indicators": ["advisor", "broker", "analyst", "accountant", "cpa", "banker"],
                "context_keywords": ["account", "investment", "loan", "credit", "financial", "bank"]
            }
        }
    
    def process_document(self, image_data: bytes, document_id: str = None) -> ExtractionResult:
        """
        Process document through 3-step pipeline
        
        Args:
            image_data: Document image as bytes
            document_id: Optional document identifier
            
        Returns:
            ExtractionResult with contextual PII extraction
        """
        if document_id is None:
            import uuid
            document_id = str(uuid.uuid4())
        
        logger.info(f"Starting contextual PII pipeline for document {document_id}")
        
        # STEP 1: Document Classification
        classification = self._classify_document(image_data)
        logger.info(f"Document classified as {classification.domain.value} with confidence {classification.confidence}")
        
        # STEP 2: Prompt Routing & Confidence Assessment
        prompt, strategy = self._route_prompt(classification)
        logger.info(f"Using {strategy} strategy for extraction")
        
        # STEP 3: PII Extraction + Role Filtering + Flagging
        extraction_result = self._extract_contextual_pii(image_data, prompt, classification, document_id)
        
        logger.info(f"Extraction complete: {len(extraction_result.data_subject_entities)} data subject entities, {len(extraction_result.professional_entities)} professional entities")
        
        return extraction_result
    
    def process_document_test_mode(self, document_id: str = None) -> ExtractionResult:
        """
        Test mode - generates mock results without API calls
        Useful for quota issues or demonstrations
        """
        if document_id is None:
            import uuid
            document_id = str(uuid.uuid4())
        
        logger.info(f"Running TEST MODE for document {document_id}")
        
        # Mock classification
        classification = ClassificationResult(
            domain=DocumentDomain.HEALTH,
            confidence=0.87,
            reasoning="Mock classification for demonstration - appears to be a health document based on medical terminology"
        )
        
        # Mock PII entities
        data_subject_entities = [
            PIIEntity(
                type="person_name",
                value="John Smith",
                confidence=0.95,
                role="data_subject", 
                context="Patient name in header",
                flags=[]
            ),
            PIIEntity(
                type="date_of_birth",
                value="1985-03-15",
                confidence=0.88,
                role="data_subject",
                context="DOB field in patient information",
                flags=[]
            )
        ]
        
        professional_entities = [
            PIIEntity(
                type="person_name",
                value="Dr. Sarah Johnson",
                confidence=0.92,
                role="professional",
                context="Attending physician signature",
                flags=["excluded_professional"]
            )
        ]
        
        all_entities = data_subject_entities + professional_entities
        
        return ExtractionResult(
            document_id=document_id,
            classification=classification,
            all_entities=all_entities,
            data_subject_entities=data_subject_entities,
            professional_entities=professional_entities,
            confidence_flags=[],
            processing_metadata={
                "model_used": "test_mode",
                "processing_time": 1.5,
                "cost": 0.0,
                "test_mode": True
            }
        )
    
    def _classify_document(self, image_data: bytes) -> ClassificationResult:
        """
        STEP 1: Classify document domain with confidence scoring
        """
        # Convert image to base64 for LLM
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        classification_prompt = """
        Analyze this document image and classify its domain. Consider:
        
        DOMAINS:
        - health: Medical records, prescriptions, health forms, patient documents
        - hr: Employment documents, job applications, HR forms, performance reviews
        - financial: Bank statements, investment documents, loan applications, tax forms
        - legal: Contracts, court documents, legal agreements, wills
        - death_cert: Death certificates, funeral documents, estate planning
        - government: Government forms, licenses, permits, official documents
        - education: School records, transcripts, academic documents
        - unknown: Cannot determine or mixed domain
        
        Provide your analysis in this JSON format:
        {
            "domain": "domain_name",
            "confidence": 0.95,
            "reasoning": "Detailed explanation of classification decision",
            "secondary_domains": [["backup_domain", 0.15]]
        }
        
        Be thorough in your reasoning and honest about confidence levels.
        """
        
        try:
            # Try multiple models if quota exceeded
            models_to_try = ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "claude-3.5-haiku"]
            
            for model in models_to_try:
                try:
                    # Get the normalized model key and provider
                    normalized_key = self.llm_service.normalize_model_key(model)
                    if normalized_key not in self.llm_service.providers:
                        continue
                    
                    provider = self.llm_service.providers[normalized_key]
                    response = provider.extract_pii(
                        image_data=image_base64,
                        prompt=classification_prompt
                    )
                    logger.info(f"Successfully used {model} for classification")
                    break
                except Exception as model_error:
                    if "quota" in str(model_error).lower() or "429" in str(model_error):
                        logger.warning(f"{model} quota exceeded, trying next model...")
                        continue
                    else:
                        raise model_error
            else:
                raise Exception("All models failed - quota exceeded or other errors")
            
            # Parse response
            result_data = json.loads(response.get('extracted_text', '{}'))
            
            domain = DocumentDomain(result_data.get('domain', 'unknown'))
            confidence = float(result_data.get('confidence', 0.0))
            reasoning = result_data.get('reasoning', 'No reasoning provided')
            secondary_domains = [(DocumentDomain(d[0]), d[1]) for d in result_data.get('secondary_domains', [])]
            
            return ClassificationResult(
                domain=domain,
                confidence=confidence,
                reasoning=reasoning,
                secondary_domains=secondary_domains
            )
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult(
                domain=DocumentDomain.UNKNOWN,
                confidence=0.0,
                reasoning=f"Classification error: {str(e)}"
            )
    
    def _route_prompt(self, classification: ClassificationResult) -> Tuple[str, str]:
        """
        STEP 2: Route to appropriate prompt based on confidence and domain
        """
        confidence = classification.confidence
        domain = classification.domain
        
        # Determine strategy based on confidence thresholds
        if confidence >= self.confidence_thresholds["high"]:
            strategy = "domain_specific"
            prompt = self._get_domain_specific_prompt(domain)
        elif confidence >= self.confidence_thresholds["medium"]:
            strategy = "general_with_hints"
            prompt = self._get_general_prompt_with_domain_hints(domain)
        elif confidence >= self.confidence_thresholds["low"]:
            strategy = "general_with_flag"
            prompt = self._get_general_prompt()
        else:
            strategy = "manual_review_required"
            prompt = self._get_general_prompt()
        
        return prompt, strategy
    
    def _get_domain_specific_prompt(self, domain: DocumentDomain) -> str:
        """Get domain-specific prompt for high confidence classification"""
        
        role_info = self.role_patterns.get(domain, {})
        data_subject_indicators = role_info.get("data_subject_indicators", ["individual", "person"])
        professional_indicators = role_info.get("professional_indicators", ["professional", "staff"])
        
        base_prompt = f"""
        CONTEXTUAL PII EXTRACTION - {domain.value.upper()} DOCUMENT

        🎯 CRITICAL OBJECTIVE: 
        Extract PII ONLY for the DATA SUBJECT (person whose consent is needed).
        IGNORE PII of professionals, witnesses, or other non-subject individuals.

        📋 DOMAIN CONTEXT: This is a {domain.value} document.
        
        🔍 DATA SUBJECT INDICATORS: {', '.join(data_subject_indicators)}
        ❌ PROFESSIONAL INDICATORS (IGNORE): {', '.join(professional_indicators)}
        
        EXTRACTION RULES:
        1. ONLY extract PII for the main data subject 
        2. IGNORE PII of professionals, doctors, lawyers, staff, witnesses
        3. Pay attention to role context: "Patient: John Doe" vs "Dr. Jane Smith"
        4. Provide confidence scores for each entity
        5. Flag ambiguous cases
        
        Return JSON format:
        {{
            "entities": [
                {{
                    "type": "person_name",
                    "value": "John Doe", 
                    "confidence": 0.95,
                    "role": "data_subject",
                    "context": "Patient name in header",
                    "reasoning": "Clearly identified as patient"
                }}
            ]
        }}
        
        ROLES: "data_subject", "professional", "witness", "unknown"
        TYPES: "person_name", "phone", "email", "address", "ssn", "date_of_birth", "account_number"
        """
        
        return base_prompt
    
    def _get_general_prompt_with_domain_hints(self, domain: DocumentDomain) -> str:
        """Get general prompt with domain hints for medium confidence"""
        return f"""
        CONTEXTUAL PII EXTRACTION - GENERAL WITH {domain.value.upper()} HINTS

        🎯 OBJECTIVE: Extract PII for data subject only, ignore professional PII
        
        This appears to be a {domain.value} document. Use this context to help identify:
        - Who is the main data subject (person needing consent)
        - Who are professionals/staff (ignore their PII)
        
        Apply contextual reasoning to distinguish roles.
        Flag uncertain extractions for review.
        
        Return same JSON format with role classifications.
        """
    
    def _get_general_prompt(self) -> str:
        """Get general prompt for low confidence or unknown domain"""
        return """
        CONTEXTUAL PII EXTRACTION - GENERAL

        🎯 OBJECTIVE: Extract PII for data subject only
        
        Extract PII entities and classify their roles:
        - data_subject: Main individual whose consent is needed
        - professional: Doctors, lawyers, staff (ignore their PII)
        - witness: Witnesses, notaries (ignore their PII)  
        - unknown: Cannot determine role (flag for review)
        
        Use context clues to determine roles. Flag uncertain cases.
        
        Return JSON format with role classifications and confidence scores.
        """
    
    def _extract_contextual_pii(self, image_data: bytes, prompt: str, classification: ClassificationResult, document_id: str) -> ExtractionResult:
        """
        STEP 3: Extract PII with role filtering and flagging
        """
        # Convert image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        try:
            # Try multiple models if quota exceeded
            models_to_try = ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "claude-3.5-haiku"]
            
            for model in models_to_try:
                try:
                    # Get the normalized model key and provider
                    normalized_key = self.llm_service.normalize_model_key(model)
                    if normalized_key not in self.llm_service.providers:
                        continue
                    
                    provider = self.llm_service.providers[normalized_key]
                    response = provider.extract_pii(
                        image_data=image_base64,
                        prompt=prompt
                    )
                    logger.info(f"Successfully used {model} for PII extraction")
                    break
                except Exception as model_error:
                    if "quota" in str(model_error).lower() or "429" in str(model_error):
                        logger.warning(f"{model} quota exceeded, trying next model...")
                        continue
                    else:
                        raise model_error
            else:
                # If all models fail, create a minimal result
                logger.error("All models failed - creating fallback result")
                return ExtractionResult(
                    document_id=document_id,
                    classification=classification,
                    all_entities=[],
                    data_subject_entities=[],
                    professional_entities=[],
                    confidence_flags=["all_models_quota_exceeded"],
                    processing_metadata={"error": "All API quotas exceeded"}
                )
            
            # Parse entities
            entities_data = json.loads(response.get('extracted_text', '{}'))
            all_entities = []
            
            for entity_data in entities_data.get('entities', []):
                entity = PIIEntity(
                    type=entity_data.get('type', 'unknown'),
                    value=entity_data.get('value', ''),
                    confidence=float(entity_data.get('confidence', 0.0)),
                    role=entity_data.get('role', 'unknown'),
                    context=entity_data.get('context', ''),
                    flags=[]
                )
                
                # Add confidence flags
                if entity.confidence < 0.7:
                    entity.flags.append("low_confidence")
                if entity.role == "unknown":
                    entity.flags.append("role_ambiguity")
                
                all_entities.append(entity)
            
            # Separate by role
            data_subject_entities = [e for e in all_entities if e.role == "data_subject"]
            professional_entities = [e for e in all_entities if e.role == "professional"]
            
            # Generate confidence flags
            confidence_flags = []
            if classification.confidence < 0.7:
                confidence_flags.append("low_classification_confidence")
            if len(data_subject_entities) == 0:
                confidence_flags.append("no_data_subject_identified")
            if any(e.role == "unknown" for e in all_entities):
                confidence_flags.append("role_ambiguity_detected")
            
            return ExtractionResult(
                document_id=document_id,
                classification=classification,
                all_entities=all_entities,
                data_subject_entities=data_subject_entities,
                professional_entities=professional_entities,
                confidence_flags=confidence_flags,
                processing_metadata={
                    "model_used": "gpt-4o",
                    "processing_time": response.get('processing_time', 0),
                    "cost": response.get('cost', 0.0),
                    "total_entities": len(all_entities),
                    "data_subject_entities_count": len(data_subject_entities),
                    "professional_entities_count": len(professional_entities)
                }
            )
            
        except Exception as e:
            logger.error(f"PII extraction failed: {e}")
            return ExtractionResult(
                document_id=document_id,
                classification=classification,
                all_entities=[],
                data_subject_entities=[],
                professional_entities=[],
                confidence_flags=["extraction_failed"],
                processing_metadata={"error": str(e)}
            )
    
    def validate_extraction_quality(self, result: ExtractionResult) -> Dict[str, Any]:
        """
        Validate extraction quality and provide recommendations
        """
        quality_score = 1.0
        recommendations = []
        
        # Check classification confidence
        if result.classification.confidence < 0.8:
            quality_score -= 0.2
            recommendations.append("Low classification confidence - consider manual review")
        
        # Check for data subject identification
        if len(result.data_subject_entities) == 0:
            quality_score -= 0.3
            recommendations.append("No data subject identified - document may need manual processing")
        
        # Check for role ambiguity
        unknown_roles = len([e for e in result.all_entities if e.role == "unknown"])
        if unknown_roles > 0:
            quality_score -= 0.1 * unknown_roles
            recommendations.append(f"{unknown_roles} entities with unknown roles - review needed")
        
        # Check confidence flags
        critical_flags = ["extraction_failed", "no_data_subject_identified"]
        if any(flag in result.confidence_flags for flag in critical_flags):
            quality_score -= 0.4
            recommendations.append("Critical issues detected - manual intervention required")
        
        return {
            "quality_score": max(0.0, quality_score),
            "grade": "A" if quality_score >= 0.9 else "B" if quality_score >= 0.7 else "C" if quality_score >= 0.5 else "D",
            "recommendations": recommendations,
            "requires_review": quality_score < 0.7 or len(recommendations) > 0
        }