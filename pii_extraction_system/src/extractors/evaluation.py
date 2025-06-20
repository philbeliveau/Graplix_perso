"""Evaluation framework for PII extraction performance metrics."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

from extractors.base import PIIEntity, PIIExtractionResult
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GroundTruthEntity:
    """Ground truth PII entity for evaluation."""
    text: str
    pii_type: str
    start_pos: int
    end_pos: int
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'type': self.pii_type,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos,
            'context': self.context,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroundTruthEntity':
        """Create from dictionary."""
        return cls(
            text=data['text'],
            pii_type=data.get('type', data.get('pii_type')),
            start_pos=data['start_pos'],
            end_pos=data['end_pos'],
            context=data.get('context', ''),
            metadata=data.get('metadata', {})
        )


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for PII extraction."""
    
    # Per-type metrics
    precision_per_type: Dict[str, float] = field(default_factory=dict)
    recall_per_type: Dict[str, float] = field(default_factory=dict)
    f1_score_per_type: Dict[str, float] = field(default_factory=dict)
    
    # Overall metrics
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1_score: float = 0.0
    
    # Detailed counts
    true_positives: Dict[str, int] = field(default_factory=dict)
    false_positives: Dict[str, int] = field(default_factory=dict)
    false_negatives: Dict[str, int] = field(default_factory=dict)
    
    # Additional metrics
    total_entities_predicted: int = 0
    total_entities_ground_truth: int = 0
    accuracy: float = 0.0
    
    # Performance metrics
    processing_time: float = 0.0
    entities_per_second: float = 0.0
    
    # Error analysis
    common_errors: List[Dict[str, Any]] = field(default_factory=list)
    confidence_distribution: Dict[str, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'precision_per_type': self.precision_per_type,
            'recall_per_type': self.recall_per_type,
            'f1_score_per_type': self.f1_score_per_type,
            'overall_precision': self.overall_precision,
            'overall_recall': self.overall_recall,
            'overall_f1_score': self.overall_f1_score,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'total_entities_predicted': self.total_entities_predicted,
            'total_entities_ground_truth': self.total_entities_ground_truth,
            'accuracy': self.accuracy,
            'processing_time': self.processing_time,
            'entities_per_second': self.entities_per_second,
            'common_errors': self.common_errors,
            'confidence_distribution': self.confidence_distribution
        }


class EvaluationResultWrapper:
    """
    Wrapper class to provide backward compatibility for evaluation results.
    
    This class wraps EvaluationMetrics to provide the expected interface
    with precision, recall, and f1_score attributes.
    """
    
    def __init__(self, metrics: EvaluationMetrics):
        """
        Initialize wrapper with evaluation metrics.
        
        Args:
            metrics: EvaluationMetrics object from evaluate_extraction_result
        """
        self._metrics = metrics
    
    @property
    def precision(self) -> float:
        """Overall precision score."""
        return self._metrics.overall_precision
    
    @property
    def recall(self) -> float:
        """Overall recall score."""
        return self._metrics.overall_recall
    
    @property
    def f1_score(self) -> float:
        """Overall F1 score."""
        return self._metrics.overall_f1_score
    
    @property
    def accuracy(self) -> float:
        """Overall accuracy score."""
        return self._metrics.accuracy
    
    @property
    def metrics(self) -> EvaluationMetrics:
        """Access to underlying metrics object."""
        return self._metrics
    
    def __str__(self) -> str:
        """String representation."""
        return f"EvaluationResult(precision={self.precision:.3f}, recall={self.recall:.3f}, f1_score={self.f1_score:.3f})"
    
    def __repr__(self) -> str:
        """String representation."""
        return self.__str__()


class PIIEvaluator:
    """Evaluator for comparing PII extraction results against ground truth."""
    
    def __init__(self, 
                 position_tolerance: int = 5,
                 text_similarity_threshold: float = 0.8):
        """
        Initialize the evaluator.
        
        Args:
            position_tolerance: Character position tolerance for matching entities
            text_similarity_threshold: Threshold for text similarity matching
        """
        self.position_tolerance = position_tolerance
        self.text_similarity_threshold = text_similarity_threshold
        
        # Mapping for similar PII types
        self.type_aliases = {
            'person_name': ['name', 'person', 'individual'],
            'email_address': ['email', 'mail', 'e-mail'],
            'phone_number': ['phone', 'telephone', 'tel'],
            'organization': ['org', 'company', 'institution'],
            'location': ['place', 'address', 'geo'],
            'date_of_birth': ['dob', 'birth_date', 'birthdate']
        }
        
        logger.info("PII Evaluator initialized")
    
    def evaluate(self, 
                predicted_entities: List[Any],
                ground_truth_entities: List[Any], 
                threshold: Optional[float] = None) -> 'EvaluationResultWrapper':
        """
        Compatibility wrapper for the evaluate method.
        
        This method provides backward compatibility by converting the input parameters
        to the format expected by evaluate_extraction_result() and wrapping the result.
        
        Args:
            predicted_entities: List of predicted entities (from LLM extraction)
            ground_truth_entities: List of ground truth entities  
            threshold: Optional threshold for matching (updates text_similarity_threshold)
            
        Returns:
            EvaluationResultWrapper with precision, recall, f1_score attributes
        """
        try:
            logger.info(f"evaluate() called with {len(predicted_entities)} predicted and {len(ground_truth_entities)} ground truth entities")
            
            # Update threshold if provided
            if threshold is not None:
                original_threshold = self.text_similarity_threshold
                self.text_similarity_threshold = threshold
            else:
                original_threshold = None
            
            # Convert predicted entities to PIIEntity objects
            pii_entities = self._convert_to_pii_entities(predicted_entities)
            
            # Convert ground truth entities to GroundTruthEntity objects  
            gt_entities = self._convert_to_ground_truth_entities(ground_truth_entities)
            
            # Create a PIIExtractionResult wrapper
            predicted_result = PIIExtractionResult(
                pii_entities=pii_entities,
                confidence_scores={},
                processing_time=0.0,
                metadata={}
            )
            
            # Call the existing evaluation method
            metrics = self.evaluate_extraction_result(predicted_result, gt_entities)
            
            # Restore original threshold if it was changed
            if original_threshold is not None:
                self.text_similarity_threshold = original_threshold
            
            # Return wrapped result for compatibility
            return EvaluationResultWrapper(metrics)
            
        except Exception as e:
            logger.error(f"Error in evaluate() method: {e}")
            # Return default metrics on error
            default_metrics = EvaluationMetrics()
            return EvaluationResultWrapper(default_metrics)
    
    def _convert_to_pii_entities(self, entities: List[Any]) -> List[PIIEntity]:
        """
        Convert various entity formats to PIIEntity objects.
        
        Args:
            entities: List of entities in various formats (dict, PIIEntity, etc.)
            
        Returns:
            List of PIIEntity objects
        """
        pii_entities = []
        
        for entity in entities:
            try:
                if isinstance(entity, PIIEntity):
                    # Already a PIIEntity
                    pii_entities.append(entity)
                elif isinstance(entity, dict):
                    # Convert from dictionary format
                    pii_entity = PIIEntity(
                        text=str(entity.get('text', entity.get('value', ''))),
                        pii_type=str(entity.get('type', entity.get('pii_type', entity.get('label', 'unknown')))),
                        start_pos=int(entity.get('start_pos', entity.get('start', 0))),
                        end_pos=int(entity.get('end_pos', entity.get('end', 0))),
                        confidence=float(entity.get('confidence', entity.get('score', 1.0))),
                        context=str(entity.get('context', '')),
                        metadata=entity.get('metadata', {})
                    )
                    pii_entities.append(pii_entity)
                else:
                    # Try to extract from object attributes
                    text = getattr(entity, 'text', getattr(entity, 'value', ''))
                    pii_type = getattr(entity, 'type', getattr(entity, 'pii_type', getattr(entity, 'label', 'unknown')))
                    start_pos = getattr(entity, 'start_pos', getattr(entity, 'start', 0))
                    end_pos = getattr(entity, 'end_pos', getattr(entity, 'end', 0))
                    confidence = getattr(entity, 'confidence', getattr(entity, 'score', 1.0))
                    context = getattr(entity, 'context', '')
                    metadata = getattr(entity, 'metadata', {})
                    
                    pii_entity = PIIEntity(
                        text=str(text),
                        pii_type=str(pii_type),
                        start_pos=int(start_pos),
                        end_pos=int(end_pos),
                        confidence=float(confidence),
                        context=str(context),
                        metadata=metadata
                    )
                    pii_entities.append(pii_entity)
                    
            except Exception as e:
                logger.warning(f"Failed to convert entity to PIIEntity: {entity}, error: {e}")
                # Create a default entity
                pii_entity = PIIEntity(
                    text=str(entity) if not isinstance(entity, dict) else str(entity.get('text', 'unknown')),
                    pii_type='unknown',
                    start_pos=0,
                    end_pos=0,
                    confidence=0.0,
                    context='',
                    metadata={}
                )
                pii_entities.append(pii_entity)
        
        logger.info(f"Converted {len(entities)} entities to {len(pii_entities)} PIIEntity objects")
        return pii_entities
    
    def _convert_to_ground_truth_entities(self, entities: List[Any]) -> List[GroundTruthEntity]:
        """
        Convert various entity formats to GroundTruthEntity objects.
        
        Args:
            entities: List of entities in various formats (dict, GroundTruthEntity, etc.)
            
        Returns:
            List of GroundTruthEntity objects
        """
        gt_entities = []
        
        for entity in entities:
            try:
                if isinstance(entity, GroundTruthEntity):
                    # Already a GroundTruthEntity
                    gt_entities.append(entity)
                elif isinstance(entity, dict):
                    # Convert from dictionary format
                    gt_entity = GroundTruthEntity(
                        text=str(entity.get('text', entity.get('value', ''))),
                        pii_type=str(entity.get('type', entity.get('pii_type', entity.get('label', 'unknown')))),
                        start_pos=int(entity.get('start_pos', entity.get('start', 0))),
                        end_pos=int(entity.get('end_pos', entity.get('end', 0))),
                        context=str(entity.get('context', '')),
                        metadata=entity.get('metadata', {})
                    )
                    gt_entities.append(gt_entity)
                else:
                    # Try to extract from object attributes
                    text = getattr(entity, 'text', getattr(entity, 'value', ''))
                    pii_type = getattr(entity, 'type', getattr(entity, 'pii_type', getattr(entity, 'label', 'unknown')))
                    start_pos = getattr(entity, 'start_pos', getattr(entity, 'start', 0))
                    end_pos = getattr(entity, 'end_pos', getattr(entity, 'end', 0))
                    context = getattr(entity, 'context', '')
                    metadata = getattr(entity, 'metadata', {})
                    
                    gt_entity = GroundTruthEntity(
                        text=str(text),
                        pii_type=str(pii_type),
                        start_pos=int(start_pos),
                        end_pos=int(end_pos),
                        context=str(context),
                        metadata=metadata
                    )
                    gt_entities.append(gt_entity)
                    
            except Exception as e:
                logger.warning(f"Failed to convert entity to GroundTruthEntity: {entity}, error: {e}")
                # Create a default entity
                gt_entity = GroundTruthEntity(
                    text=str(entity) if not isinstance(entity, dict) else str(entity.get('text', 'unknown')),
                    pii_type='unknown',
                    start_pos=0,
                    end_pos=0,
                    context='',
                    metadata={}
                )
                gt_entities.append(gt_entity)
        
        logger.info(f"Converted {len(entities)} entities to {len(gt_entities)} GroundTruthEntity objects")
        return gt_entities
    
    def evaluate_extraction_result(self,
                                 predicted_result: PIIExtractionResult,
                                 ground_truth: List[GroundTruthEntity],
                                 document_text: Optional[str] = None) -> EvaluationMetrics:
        """
        Evaluate a single extraction result against ground truth.
        
        Args:
            predicted_result: PII extraction result from extractor
            ground_truth: List of ground truth entities
            document_text: Original document text (optional, for better error analysis)
            
        Returns:
            EvaluationMetrics with comprehensive evaluation results
        """
        logger.info(f"Evaluating extraction result: {len(predicted_result.pii_entities)} predicted vs {len(ground_truth)} ground truth")
        
        # Initialize metrics
        metrics = EvaluationMetrics()
        metrics.total_entities_predicted = len(predicted_result.pii_entities)
        metrics.total_entities_ground_truth = len(ground_truth)
        metrics.processing_time = predicted_result.processing_time
        
        # Get all unique PII types
        all_types = self._get_all_pii_types(predicted_result.pii_entities, ground_truth)
        
        # Match predicted entities with ground truth
        matches = self._match_entities(predicted_result.pii_entities, ground_truth)
        
        # Calculate metrics for each type
        for pii_type in all_types:
            tp, fp, fn = self._calculate_counts_for_type(
                pii_type, matches, predicted_result.pii_entities, ground_truth
            )
            
            metrics.true_positives[pii_type] = tp
            metrics.false_positives[pii_type] = fp
            metrics.false_negatives[pii_type] = fn
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics.precision_per_type[pii_type] = precision
            metrics.recall_per_type[pii_type] = recall
            metrics.f1_score_per_type[pii_type] = f1_score
        
        # Calculate overall metrics
        metrics.overall_precision = self._calculate_overall_precision(metrics)
        metrics.overall_recall = self._calculate_overall_recall(metrics)
        metrics.overall_f1_score = self._calculate_overall_f1_score(metrics)
        
        # Calculate additional metrics
        metrics.accuracy = self._calculate_accuracy(metrics)
        metrics.entities_per_second = len(predicted_result.pii_entities) / max(metrics.processing_time, 0.001)
        
        # Error analysis
        metrics.common_errors = self._analyze_errors(matches, predicted_result.pii_entities, ground_truth)
        metrics.confidence_distribution = self._analyze_confidence_distribution(predicted_result.pii_entities)
        
        logger.info(f"Evaluation complete: P={metrics.overall_precision:.3f}, R={metrics.overall_recall:.3f}, F1={metrics.overall_f1_score:.3f}")
        
        return metrics
    
    def evaluate_multiple_results(self,
                                results: List[Tuple[PIIExtractionResult, List[GroundTruthEntity]]],
                                aggregate: bool = True) -> List[EvaluationMetrics]:
        """
        Evaluate multiple extraction results.
        
        Args:
            results: List of (predicted_result, ground_truth) tuples
            aggregate: Whether to return aggregated metrics
            
        Returns:
            List of evaluation metrics (or single aggregated metrics if aggregate=True)
        """
        logger.info(f"Evaluating {len(results)} extraction results")
        
        individual_metrics = []
        
        for i, (predicted_result, ground_truth) in enumerate(results):
            try:
                metrics = self.evaluate_extraction_result(predicted_result, ground_truth)
                individual_metrics.append(metrics)
                logger.info(f"Evaluated result {i+1}/{len(results)}")
            except Exception as e:
                logger.error(f"Error evaluating result {i+1}: {e}")
        
        if aggregate and individual_metrics:
            aggregated = self._aggregate_metrics(individual_metrics)
            return [aggregated]
        
        return individual_metrics
    
    def _get_all_pii_types(self, predicted: List[PIIEntity], ground_truth: List[GroundTruthEntity]) -> Set[str]:
        """Get all unique PII types from both predicted and ground truth."""
        types = set()
        
        for entity in predicted:
            types.add(entity.pii_type)
        
        for entity in ground_truth:
            types.add(entity.pii_type)
        
        return types
    
    def _match_entities(self, 
                       predicted: List[PIIEntity], 
                       ground_truth: List[GroundTruthEntity]) -> Dict[str, Any]:
        """Match predicted entities with ground truth entities."""
        matches = {
            'matched_pairs': [],
            'unmatched_predicted': list(predicted),
            'unmatched_ground_truth': list(ground_truth)
        }
        
        # Sort by position for efficient matching
        predicted_sorted = sorted(predicted, key=lambda x: x.start_pos)
        ground_truth_sorted = sorted(ground_truth, key=lambda x: x.start_pos)
        
        for pred_idx, pred_entity in enumerate(predicted_sorted):
            best_match = None
            best_score = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_entity in enumerate(ground_truth_sorted):
                if gt_entity in matches['unmatched_ground_truth']:
                    match_score = self._calculate_match_score(pred_entity, gt_entity)
                    
                    if match_score > best_score and match_score > 0.5:
                        best_match = gt_entity
                        best_score = match_score
                        best_gt_idx = gt_idx
            
            if best_match:
                matches['matched_pairs'].append((pred_entity, best_match, best_score))
                matches['unmatched_predicted'].remove(pred_entity)
                matches['unmatched_ground_truth'].remove(best_match)
        
        return matches
    
    def _calculate_match_score(self, predicted: PIIEntity, ground_truth: GroundTruthEntity) -> float:
        """Calculate match score between predicted and ground truth entities."""
        score = 0.0
        
        # Type matching (required)
        if not self._types_match(predicted.pii_type, ground_truth.pii_type):
            return 0.0
        
        score += 0.3  # Base score for type match
        
        # Position overlap
        pred_range = range(predicted.start_pos, predicted.end_pos)
        gt_range = range(ground_truth.start_pos, ground_truth.end_pos)
        
        overlap = len(set(pred_range) & set(gt_range))
        union = len(set(pred_range) | set(gt_range))
        
        if union > 0:
            position_score = overlap / union
            score += 0.4 * position_score
        
        # Text similarity
        text_similarity = self._calculate_text_similarity(predicted.text, ground_truth.text)
        score += 0.3 * text_similarity
        
        return score
    
    def _types_match(self, type1: str, type2: str) -> bool:
        """Check if two PII types are considered equivalent."""
        if type1 == type2:
            return True
        
        # Check aliases
        for canonical_type, aliases in self.type_aliases.items():
            if (type1 == canonical_type and type2 in aliases) or \
               (type2 == canonical_type and type1 in aliases) or \
               (type1 in aliases and type2 in aliases):
                return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two strings."""
        # Simple character-based similarity
        text1_clean = text1.lower().strip()
        text2_clean = text2.lower().strip()
        
        if text1_clean == text2_clean:
            return 1.0
        
        # Jaccard similarity
        set1 = set(text1_clean.split())
        set2 = set(text2_clean.split())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_counts_for_type(self,
                                 pii_type: str,
                                 matches: Dict[str, Any],
                                 predicted: List[PIIEntity],
                                 ground_truth: List[GroundTruthEntity]) -> Tuple[int, int, int]:
        """Calculate TP, FP, FN counts for a specific PII type."""
        # True Positives: matched entities of this type
        tp = sum(1 for pred, gt, score in matches['matched_pairs'] 
                if self._types_match(pred.pii_type, pii_type))
        
        # False Positives: unmatched predicted entities of this type
        fp = sum(1 for entity in matches['unmatched_predicted'] 
                if self._types_match(entity.pii_type, pii_type))
        
        # False Negatives: unmatched ground truth entities of this type
        fn = sum(1 for entity in matches['unmatched_ground_truth'] 
                if self._types_match(entity.pii_type, pii_type))
        
        return tp, fp, fn
    
    def _calculate_overall_precision(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall precision (micro-average)."""
        total_tp = sum(metrics.true_positives.values())
        total_fp = sum(metrics.false_positives.values())
        
        return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    
    def _calculate_overall_recall(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall recall (micro-average)."""
        total_tp = sum(metrics.true_positives.values())
        total_fn = sum(metrics.false_negatives.values())
        
        return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    def _calculate_overall_f1_score(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall F1 score."""
        precision = metrics.overall_precision
        recall = metrics.overall_recall
        
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _calculate_accuracy(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall accuracy."""
        total_tp = sum(metrics.true_positives.values())
        total_fp = sum(metrics.false_positives.values())
        total_fn = sum(metrics.false_negatives.values())
        
        total = total_tp + total_fp + total_fn
        return total_tp / total if total > 0 else 0.0
    
    def _analyze_errors(self,
                       matches: Dict[str, Any],
                       predicted: List[PIIEntity],
                       ground_truth: List[GroundTruthEntity]) -> List[Dict[str, Any]]:
        """Analyze common errors for improvement insights."""
        errors = []
        
        # False Positives
        for entity in matches['unmatched_predicted'][:10]:  # Top 10 errors
            errors.append({
                'type': 'false_positive',
                'entity_type': entity.pii_type,
                'text': entity.text,
                'confidence': entity.confidence,
                'context': entity.context[:100] if entity.context else ""
            })
        
        # False Negatives
        for entity in matches['unmatched_ground_truth'][:10]:  # Top 10 errors
            errors.append({
                'type': 'false_negative',
                'entity_type': entity.pii_type,
                'text': entity.text,
                'context': entity.context[:100] if entity.context else ""
            })
        
        return errors
    
    def _analyze_confidence_distribution(self, predicted: List[PIIEntity]) -> Dict[str, List[float]]:
        """Analyze confidence score distribution by PII type."""
        distribution = defaultdict(list)
        
        for entity in predicted:
            distribution[entity.pii_type].append(entity.confidence)
        
        return dict(distribution)
    
    def _aggregate_metrics(self, metrics_list: List[EvaluationMetrics]) -> EvaluationMetrics:
        """Aggregate multiple evaluation metrics."""
        if not metrics_list:
            return EvaluationMetrics()
        
        # Aggregate counts
        aggregated = EvaluationMetrics()
        all_types = set()
        
        for metrics in metrics_list:
            all_types.update(metrics.true_positives.keys())
        
        for pii_type in all_types:
            aggregated.true_positives[pii_type] = sum(
                m.true_positives.get(pii_type, 0) for m in metrics_list
            )
            aggregated.false_positives[pii_type] = sum(
                m.false_positives.get(pii_type, 0) for m in metrics_list
            )
            aggregated.false_negatives[pii_type] = sum(
                m.false_negatives.get(pii_type, 0) for m in metrics_list
            )
        
        # Recalculate metrics
        for pii_type in all_types:
            tp = aggregated.true_positives[pii_type]
            fp = aggregated.false_positives[pii_type]
            fn = aggregated.false_negatives[pii_type]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            aggregated.precision_per_type[pii_type] = precision
            aggregated.recall_per_type[pii_type] = recall
            aggregated.f1_score_per_type[pii_type] = f1_score
        
        # Overall metrics
        aggregated.overall_precision = self._calculate_overall_precision(aggregated)
        aggregated.overall_recall = self._calculate_overall_recall(aggregated)
        aggregated.overall_f1_score = self._calculate_overall_f1_score(aggregated)
        
        # Other aggregated values
        aggregated.total_entities_predicted = sum(m.total_entities_predicted for m in metrics_list)
        aggregated.total_entities_ground_truth = sum(m.total_entities_ground_truth for m in metrics_list)
        aggregated.processing_time = sum(m.processing_time for m in metrics_list)
        aggregated.accuracy = self._calculate_accuracy(aggregated)
        aggregated.entities_per_second = aggregated.total_entities_predicted / max(aggregated.processing_time, 0.001)
        
        return aggregated
    
    def save_evaluation_report(self, 
                             metrics: EvaluationMetrics,
                             output_path: Path,
                             extractor_name: str = "unknown") -> bool:
        """Save evaluation report to file."""
        try:
            report = {
                'extractor_name': extractor_name,
                'evaluation_timestamp': __import__('datetime').datetime.now().isoformat(),
                'metrics': metrics.to_dict()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation report saved to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
            return False