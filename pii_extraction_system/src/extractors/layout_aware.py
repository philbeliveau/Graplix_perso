"""Layout-aware PII extraction using LayoutLM and Donut models."""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import io
import base64
from transformers import (
    LayoutLMTokenizer, LayoutLMForTokenClassification,
    AutoProcessor, VisionEncoderDecoderModel
)
from .base import PIIExtractorBase, PIIEntity, PIIExtractionResult
import logging

logger = logging.getLogger(__name__)


class LayoutLMExtractor(PIIExtractorBase):
    """LayoutLM-based PII extractor for document layout understanding."""
    
    def __init__(self, 
                 model_name: str = "microsoft/layoutlm-base-uncased",
                 confidence_threshold: float = 0.85,
                 device: Optional[str] = None):
        """Initialize LayoutLM extractor."""
        super().__init__("layoutlm")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        try:
            self.tokenizer = LayoutLMTokenizer.from_pretrained(model_name)
            self.model = LayoutLMForTokenClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"LayoutLM model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load LayoutLM model: {e}")
            raise
        
        # PII type mapping for NER labels
        self._label_to_pii_type = {
            'B-PER': 'person_name',
            'I-PER': 'person_name', 
            'B-ORG': 'organization',
            'I-ORG': 'organization',
            'B-LOC': 'location',
            'I-LOC': 'location',
            'B-MISC': 'misc_pii',
            'I-MISC': 'misc_pii'
        }
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """Extract PII using LayoutLM with layout information."""
        try:
            start_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            end_time = torch.cuda.Event(enable_timing=True) if self.device == "cuda" else None
            
            if start_time:
                start_time.record()
            
            # Extract text and layout information
            text = document.get('text', '')
            layout_info = document.get('layout', {})
            
            if not text:
                return PIIExtractionResult(error="No text content found")
            
            # Prepare inputs with layout information
            encoding = self._prepare_layout_input(text, layout_info)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**encoding)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_token_class_ids = predictions.argmax(dim=-1)
            
            # Convert predictions to PII entities
            entities = self._convert_predictions_to_entities(
                text, encoding, predicted_token_class_ids, predictions
            )
            
            # Calculate processing time
            processing_time = 0.0
            if start_time:
                end_time.record()
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time) / 1000.0
            
            return PIIExtractionResult(
                pii_entities=entities,
                processing_time=processing_time,
                metadata={
                    'model_name': self.model_name,
                    'device': self.device,
                    'num_tokens': len(encoding['input_ids'][0])
                }
            )
            
        except Exception as e:
            logger.error(f"LayoutLM extraction failed: {e}")
            return PIIExtractionResult(error=str(e))
    
    def _prepare_layout_input(self, text: str, layout_info: Dict) -> Dict[str, torch.Tensor]:
        """Prepare input with layout information for LayoutLM."""
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Create bounding boxes (normalized coordinates 0-1000)
        words = text.split()
        bbox = []
        
        if 'bounding_boxes' in layout_info:
            # Use provided bounding boxes
            boxes = layout_info['bounding_boxes'][:len(words)]
            for box in boxes:
                # Normalize coordinates to 0-1000 range
                bbox.append([
                    int(box.get('x1', 0) * 1000),
                    int(box.get('y1', 0) * 1000), 
                    int(box.get('x2', 100) * 1000),
                    int(box.get('y2', 100) * 1000)
                ])
        else:
            # Create dummy bounding boxes
            for i, word in enumerate(words):
                bbox.append([0, i * 20, len(word) * 10, (i + 1) * 20])
        
        # Pad or truncate bounding boxes to match token length
        token_count = len(encoding['input_ids'][0])
        while len(bbox) < token_count:
            bbox.append([0, 0, 0, 0])
        bbox = bbox[:token_count]
        
        encoding['bbox'] = torch.tensor([bbox], dtype=torch.long)
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        return encoding
    
    def _convert_predictions_to_entities(self, 
                                       text: str,
                                       encoding: Dict[str, torch.Tensor],
                                       predictions: torch.Tensor,
                                       probabilities: torch.Tensor) -> List[PIIEntity]:
        """Convert model predictions to PII entities."""
        entities = []
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        current_entity = None
        current_tokens = []
        
        for i, (token, pred_id, probs) in enumerate(zip(tokens, predictions[0], probabilities[0])):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            label = self.model.config.id2label[pred_id.item()]
            confidence = probs[pred_id].item()
            
            if confidence < self.confidence_threshold:
                continue
            
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, current_tokens, text))
                
                pii_type = self._label_to_pii_type.get(label, 'unknown')
                current_entity = {
                    'type': pii_type,
                    'confidence': confidence,
                    'start_token': i
                }
                current_tokens = [token]
                
            elif label.startswith('I-') and current_entity:
                # Continuation of entity
                current_tokens.append(token)
                # Update confidence (average)
                current_entity['confidence'] = (current_entity['confidence'] + confidence) / 2
        
        # Finalize last entity
        if current_entity:
            entities.append(self._finalize_entity(current_entity, current_tokens, text))
        
        return [e for e in entities if self.validate_entity(e)]
    
    def _finalize_entity(self, entity_info: Dict, tokens: List[str], text: str) -> PIIEntity:
        """Convert token-level entity to final PII entity."""
        # Reconstruct text from tokens
        entity_text = self.tokenizer.convert_tokens_to_string(tokens)
        
        # Find position in original text (approximate)
        start_pos = text.find(entity_text)
        end_pos = start_pos + len(entity_text) if start_pos >= 0 else 0
        
        # Extract context
        context = self._extract_context(text, start_pos, end_pos) if start_pos >= 0 else ""
        
        return self._create_entity(
            text=entity_text,
            pii_type=entity_info['type'],
            confidence=entity_info['confidence'],
            start_pos=start_pos,
            end_pos=end_pos,
            context=context,
            token_count=len(tokens)
        )


class DonutExtractor(PIIExtractorBase):
    """Donut model for OCR-free document understanding and PII extraction."""
    
    def __init__(self,
                 model_name: str = "naver-clova-ix/donut-base-finetuned-docvqa",
                 confidence_threshold: float = 0.8,
                 device: Optional[str] = None):
        """Initialize Donut extractor."""
        super().__init__("donut")
        
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Donut model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Donut model: {e}")
            raise
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """Extract PII using Donut vision-language model."""
        try:
            # Get image data
            image_data = document.get('image_data')
            if not image_data:
                return PIIExtractionResult(error="No image data found for Donut processing")
            
            # Load image
            image = self._load_image(image_data)
            if image is None:
                return PIIExtractionResult(error="Failed to load image")
            
            # Process with Donut
            entities = self._process_with_donut(image)
            
            return PIIExtractionResult(
                pii_entities=entities,
                metadata={
                    'model_name': self.model_name,
                    'device': self.device,
                    'image_size': image.size
                }
            )
            
        except Exception as e:
            logger.error(f"Donut extraction failed: {e}")
            return PIIExtractionResult(error=str(e))
    
    def _load_image(self, image_data: Any) -> Optional[Image.Image]:
        """Load image from various input formats."""
        try:
            if isinstance(image_data, str):
                # Base64 encoded image
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                return Image.open(io.BytesIO(image_bytes)).convert('RGB')
            elif isinstance(image_data, bytes):
                return Image.open(io.BytesIO(image_data)).convert('RGB')
            elif isinstance(image_data, Image.Image):
                return image_data.convert('RGB')
            else:
                logger.error(f"Unsupported image data type: {type(image_data)}")
                return None
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def _process_with_donut(self, image: Image.Image) -> List[PIIEntity]:
        """Process image with Donut model to extract PII."""
        entities = []
        
        # Define PII extraction prompts
        pii_prompts = [
            "<s_docvqa><s_question>What are the names mentioned in this document?</s_question><s_answer>",
            "<s_docvqa><s_question>What are the email addresses in this document?</s_question><s_answer>",
            "<s_docvqa><s_question>What are the phone numbers in this document?</s_question><s_answer>",
            "<s_docvqa><s_question>What are the addresses in this document?</s_question><s_answer>",
        ]
        
        pii_types = ['person_name', 'email_address', 'phone_number', 'address']
        
        for prompt, pii_type in zip(pii_prompts, pii_types):
            try:
                # Prepare inputs
                pixel_values = self.processor(image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                # Generate with prompt
                decoder_input_ids = self.processor.tokenizer(
                    prompt, 
                    add_special_tokens=False,
                    return_tensors="pt"
                ).input_ids.to(self.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        pixel_values,
                        decoder_input_ids=decoder_input_ids,
                        max_length=self.model.decoder.config.max_position_embeddings,
                        early_stopping=True,
                        pad_token_id=self.processor.tokenizer.pad_token_id,
                        eos_token_id=self.processor.tokenizer.eos_token_id,
                        use_cache=True,
                        num_beams=1,
                        bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # Decode response
                sequence = outputs.sequences[0]
                response = self.processor.batch_decode([sequence], skip_special_tokens=True)[0]
                
                # Extract PII from response
                pii_entities = self._extract_pii_from_response(response, pii_type)
                entities.extend(pii_entities)
                
            except Exception as e:
                logger.warning(f"Failed to process prompt for {pii_type}: {e}")
                continue
        
        return entities
    
    def _extract_pii_from_response(self, response: str, pii_type: str) -> List[PIIEntity]:
        """Extract PII entities from Donut model response."""
        entities = []
        
        # Clean response
        response = response.strip()
        if not response or response.lower() in ['none', 'no', 'not found', 'n/a']:
            return entities
        
        # Split response into potential entities
        potential_entities = [item.strip() for item in response.split(',')]
        
        for entity_text in potential_entities:
            if entity_text and len(entity_text) > 1:
                # Basic validation based on PII type
                if self._validate_pii_text(entity_text, pii_type):
                    entity = self._create_entity(
                        text=entity_text,
                        pii_type=pii_type,
                        confidence=self.confidence_threshold,  # Default confidence
                        model_source="donut_vision"
                    )
                    entities.append(entity)
        
        return entities
    
    def _validate_pii_text(self, text: str, pii_type: str) -> bool:
        """Basic validation of extracted PII text."""
        text = text.strip()
        
        if pii_type == 'email_address':
            return '@' in text and '.' in text
        elif pii_type == 'phone_number':
            return any(char.isdigit() for char in text) and len(text) >= 7
        elif pii_type == 'person_name':
            return len(text) >= 2 and not text.isdigit()
        elif pii_type == 'address':
            return len(text) >= 5
        
        return len(text) >= 2


class LayoutAwareEnsemble(PIIExtractorBase):
    """Ensemble of layout-aware extractors for improved accuracy."""
    
    def __init__(self, 
                 use_layoutlm: bool = True,
                 use_donut: bool = True,
                 ensemble_method: str = "weighted_vote",
                 confidence_threshold: float = 0.7):
        """Initialize ensemble extractor."""
        super().__init__("layout_aware_ensemble")
        
        self.extractors = []
        self.ensemble_method = ensemble_method
        self.confidence_threshold = confidence_threshold
        
        # Initialize extractors
        if use_layoutlm:
            try:
                self.extractors.append(LayoutLMExtractor())
                logger.info("LayoutLM extractor added to ensemble")
            except Exception as e:
                logger.warning(f"Failed to initialize LayoutLM: {e}")
        
        if use_donut:
            try:
                self.extractors.append(DonutExtractor())
                logger.info("Donut extractor added to ensemble")
            except Exception as e:
                logger.warning(f"Failed to initialize Donut: {e}")
        
        if not self.extractors:
            raise ValueError("No extractors successfully initialized")
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """Extract PII using ensemble of layout-aware models."""
        all_results = []
        processing_times = []
        
        # Run all extractors
        for extractor in self.extractors:
            try:
                result = extractor.extract_pii(document)
                all_results.append(result)
                processing_times.append(result.processing_time)
            except Exception as e:
                logger.warning(f"Extractor {extractor.name} failed: {e}")
                continue
        
        if not all_results:
            return PIIExtractionResult(error="All extractors failed")
        
        # Combine results using ensemble method
        if self.ensemble_method == "weighted_vote":
            combined_entities = self._weighted_vote_ensemble(all_results)
        elif self.ensemble_method == "union":
            combined_entities = self._union_ensemble(all_results)
        else:
            combined_entities = self._majority_vote_ensemble(all_results)
        
        return PIIExtractionResult(
            pii_entities=combined_entities,
            processing_time=sum(processing_times),
            metadata={
                'ensemble_method': self.ensemble_method,
                'num_extractors': len(self.extractors),
                'extractor_names': [e.name for e in self.extractors]
            }
        )
    
    def _weighted_vote_ensemble(self, results: List[PIIExtractionResult]) -> List[PIIEntity]:
        """Combine results using weighted voting based on confidence."""
        entity_groups = {}
        
        # Group similar entities
        for result in results:
            for entity in result.pii_entities:
                key = (entity.text.lower(), entity.pii_type)
                if key not in entity_groups:
                    entity_groups[key] = []
                entity_groups[key].append(entity)
        
        # Select best entity from each group
        final_entities = []
        for entities in entity_groups.values():
            if len(entities) == 1:
                final_entities.append(entities[0])
            else:
                # Weight by confidence and number of extractors
                weighted_confidence = sum(e.confidence for e in entities) / len(entities)
                if weighted_confidence >= self.confidence_threshold:
                    best_entity = max(entities, key=lambda x: x.confidence)
                    # Update confidence with weighted average
                    best_entity.confidence = weighted_confidence
                    best_entity.metadata['ensemble_votes'] = len(entities)
                    final_entities.append(best_entity)
        
        return final_entities
    
    def _union_ensemble(self, results: List[PIIExtractionResult]) -> List[PIIEntity]:
        """Combine results using union (all unique entities)."""
        all_entities = []
        seen_entities = set()
        
        for result in results:
            for entity in result.pii_entities:
                key = (entity.text.lower(), entity.pii_type)
                if key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(entity)
        
        return all_entities
    
    def _majority_vote_ensemble(self, results: List[PIIExtractionResult]) -> List[PIIEntity]:
        """Combine results using majority voting."""
        entity_votes = {}
        
        # Count votes for each entity
        for result in results:
            for entity in result.pii_entities:
                key = (entity.text.lower(), entity.pii_type)
                if key not in entity_votes:
                    entity_votes[key] = []
                entity_votes[key].append(entity)
        
        # Select entities with majority votes
        final_entities = []
        majority_threshold = len(self.extractors) / 2
        
        for entities in entity_votes.values():
            if len(entities) > majority_threshold:
                best_entity = max(entities, key=lambda x: x.confidence)
                best_entity.metadata['vote_count'] = len(entities)
                final_entities.append(best_entity)
        
        return final_entities


# Alias for backward compatibility and pipeline integration
LayoutAwareExtractor = LayoutAwareEnsemble