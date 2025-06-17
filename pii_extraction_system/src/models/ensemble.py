"""Ensemble methods for PII extraction."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
from sklearn.metrics import f1_score, precision_score, recall_score
from extractors.base import PIIExtractorBase, PIIEntity, PIIExtractionResult
from extractors.rule_based import RuleBasedExtractor
from extractors.layout_aware import LayoutAwareEnsemble

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods."""
    method: str = "weighted_vote"  # weighted_vote, majority_vote, stacking, meta_learning
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.8
    weight_by_performance: bool = True
    use_confidence_scores: bool = True
    normalize_weights: bool = True
    
    # Stacking specific parameters
    meta_model_type: str = "logistic_regression"
    meta_model_features: List[str] = field(default_factory=lambda: [
        "confidence", "extractor_type", "entity_length", "context_length"
    ])
    
    # Performance weights (if known)
    extractor_weights: Dict[str, float] = field(default_factory=dict)


class EntitySimilarityMatcher:
    """Matches similar entities across different extractors."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """Initialize similarity matcher."""
        self.similarity_threshold = similarity_threshold
    
    def group_similar_entities(self, entities: List[PIIEntity]) -> List[List[PIIEntity]]:
        """Group similar entities together."""
        groups = []
        ungrouped = entities.copy()
        
        while ungrouped:
            current_entity = ungrouped.pop(0)
            current_group = [current_entity]
            
            # Find similar entities
            similar_indices = []
            for i, entity in enumerate(ungrouped):
                if self._are_entities_similar(current_entity, entity):
                    current_group.append(entity)
                    similar_indices.append(i)
            
            # Remove similar entities from ungrouped
            for i in reversed(similar_indices):
                ungrouped.pop(i)
            
            groups.append(current_group)
        
        return groups
    
    def _are_entities_similar(self, entity1: PIIEntity, entity2: PIIEntity) -> bool:
        """Check if two entities are similar enough to be considered the same."""
        # Must be same type
        if entity1.pii_type != entity2.pii_type:
            return False
        
        # Check text similarity
        text_similarity = self._calculate_text_similarity(entity1.text, entity2.text)
        if text_similarity < self.similarity_threshold:
            return False
        
        # Check position overlap if available
        if (entity1.start_pos > 0 and entity1.end_pos > 0 and 
            entity2.start_pos > 0 and entity2.end_pos > 0):
            overlap = self._calculate_position_overlap(entity1, entity2)
            return overlap > 0.5  # At least 50% overlap
        
        return True
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        if text1 == text2:
            return 1.0
        
        # Jaccard similarity for words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_position_overlap(self, entity1: PIIEntity, entity2: PIIEntity) -> float:
        """Calculate position overlap between two entities."""
        start1, end1 = entity1.start_pos, entity1.end_pos
        start2, end2 = entity2.start_pos, entity2.end_pos
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start >= overlap_end:
            return 0.0  # No overlap
        
        overlap_length = overlap_end - overlap_start
        entity1_length = end1 - start1
        entity2_length = end2 - start2
        
        # Return overlap as percentage of average entity length
        avg_length = (entity1_length + entity2_length) / 2
        return overlap_length / avg_length if avg_length > 0 else 0.0


class EnsembleVotingSystem:
    """Implements various voting strategies for ensemble methods."""
    
    def __init__(self, config: EnsembleConfig):
        """Initialize voting system."""
        self.config = config
        self.similarity_matcher = EntitySimilarityMatcher(config.similarity_threshold)
    
    def majority_vote(self, entity_groups: List[List[PIIEntity]]) -> List[PIIEntity]:
        """Select entities using majority voting."""
        final_entities = []
        
        for group in entity_groups:
            if len(group) >= 2:  # Majority requires at least 2 votes
                # Select entity with highest confidence from the group
                best_entity = max(group, key=lambda x: x.confidence)
                best_entity.metadata['vote_count'] = len(group)
                best_entity.metadata['voting_method'] = 'majority'
                final_entities.append(best_entity)
        
        return final_entities
    
    def weighted_vote(self, entity_groups: List[List[PIIEntity]]) -> List[PIIEntity]:
        """Select entities using weighted voting."""
        final_entities = []
        
        for group in entity_groups:
            if not group:
                continue
            
            # Calculate weighted confidence
            total_weight = 0
            weighted_confidence = 0
            
            for entity in group:
                weight = self._get_extractor_weight(entity.extractor)
                if self.config.use_confidence_scores:
                    weight *= entity.confidence
                
                weighted_confidence += entity.confidence * weight
                total_weight += weight
            
            if total_weight > 0:
                avg_confidence = weighted_confidence / total_weight
                
                if avg_confidence >= self.config.confidence_threshold:
                    # Select best entity and update confidence
                    best_entity = max(group, key=lambda x: x.confidence)
                    best_entity.confidence = avg_confidence
                    best_entity.metadata['weighted_confidence'] = avg_confidence
                    best_entity.metadata['vote_count'] = len(group)
                    best_entity.metadata['voting_method'] = 'weighted'
                    final_entities.append(best_entity)
        
        return final_entities
    
    def union_vote(self, entity_groups: List[List[PIIEntity]]) -> List[PIIEntity]:
        """Include all unique entities (union approach)."""
        final_entities = []
        
        for group in entity_groups:
            if group:
                # Select entity with highest confidence
                best_entity = max(group, key=lambda x: x.confidence)
                best_entity.metadata['vote_count'] = len(group)
                best_entity.metadata['voting_method'] = 'union'
                final_entities.append(best_entity)
        
        return final_entities
    
    def consensus_vote(self, entity_groups: List[List[PIIEntity]], 
                      consensus_threshold: float = 0.8) -> List[PIIEntity]:
        """Select entities that reach consensus threshold."""
        final_entities = []
        total_extractors = self._estimate_total_extractors(entity_groups)
        
        for group in entity_groups:
            if not group:
                continue
            
            consensus_ratio = len(group) / total_extractors
            if consensus_ratio >= consensus_threshold:
                best_entity = max(group, key=lambda x: x.confidence)
                best_entity.metadata['consensus_ratio'] = consensus_ratio
                best_entity.metadata['vote_count'] = len(group)
                best_entity.metadata['voting_method'] = 'consensus'
                final_entities.append(best_entity)
        
        return final_entities
    
    def _get_extractor_weight(self, extractor_name: str) -> float:
        """Get weight for an extractor."""
        if extractor_name in self.config.extractor_weights:
            return self.config.extractor_weights[extractor_name]
        return 1.0  # Default weight
    
    def _estimate_total_extractors(self, entity_groups: List[List[PIIEntity]]) -> int:
        """Estimate total number of extractors from entity groups."""
        all_extractors = set()
        for group in entity_groups:
            for entity in group:
                all_extractors.add(entity.extractor)
        return len(all_extractors)


class MetaLearningEnsemble:
    """Advanced ensemble using meta-learning approach."""
    
    def __init__(self, config: EnsembleConfig):
        """Initialize meta-learning ensemble."""
        self.config = config
        self.meta_model = None
        self.feature_extractor = EntityFeatureExtractor()
        self.is_trained = False
    
    def train_meta_model(self, training_results: List[Tuple[List[PIIExtractionResult], List[PIIEntity]]]):
        """Train meta-model on extractor results and ground truth."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        
        # Extract features and labels
        X, y = self._prepare_meta_training_data(training_results)
        
        if len(X) == 0:
            logger.warning("No training data available for meta-model")
            return
        
        # Initialize meta-model
        if self.config.meta_model_type == "logistic_regression":
            self.meta_model = LogisticRegression(random_state=42)
        elif self.config.meta_model_type == "random_forest":
            self.meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.config.meta_model_type == "svm":
            self.meta_model = SVC(probability=True, random_state=42)
        else:
            self.meta_model = LogisticRegression(random_state=42)
        
        # Train meta-model
        self.meta_model.fit(X, y)
        self.is_trained = True
        
        logger.info(f"Meta-model trained with {len(X)} examples")
    
    def predict_with_meta_model(self, entity_groups: List[List[PIIEntity]]) -> List[PIIEntity]:
        """Use meta-model to select best entities."""
        if not self.is_trained:
            logger.warning("Meta-model not trained, falling back to weighted voting")
            voting_system = EnsembleVotingSystem(self.config)
            return voting_system.weighted_vote(entity_groups)
        
        final_entities = []
        
        for group in entity_groups:
            if not group:
                continue
            
            # Extract features for each entity in group
            features = []
            for entity in group:
                entity_features = self.feature_extractor.extract_features(entity)
                features.append(entity_features)
            
            if features:
                # Predict probabilities
                X = np.array(features)
                probabilities = self.meta_model.predict_proba(X)
                
                # Select entity with highest probability of being correct
                best_idx = np.argmax(probabilities[:, 1])  # Assuming class 1 is "correct"
                best_entity = group[best_idx]
                best_entity.confidence = probabilities[best_idx, 1]
                best_entity.metadata['meta_prediction'] = True
                best_entity.metadata['vote_count'] = len(group)
                
                final_entities.append(best_entity)
        
        return final_entities
    
    def _prepare_meta_training_data(self, 
                                  training_results: List[Tuple[List[PIIExtractionResult], List[PIIEntity]]]) -> Tuple[List, List]:
        """Prepare training data for meta-model."""
        X, y = [], []
        
        for extractor_results, ground_truth in training_results:
            # Combine all extracted entities
            all_entities = []
            for result in extractor_results:
                all_entities.extend(result.pii_entities)
            
            # Group similar entities
            similarity_matcher = EntitySimilarityMatcher(self.config.similarity_threshold)
            entity_groups = similarity_matcher.group_similar_entities(all_entities)
            
            # Create training examples
            for group in entity_groups:
                for entity in group:
                    features = self.feature_extractor.extract_features(entity)
                    
                    # Determine if entity is correct (matches ground truth)
                    is_correct = self._is_entity_correct(entity, ground_truth)
                    
                    X.append(features)
                    y.append(1 if is_correct else 0)
        
        return X, y
    
    def _is_entity_correct(self, entity: PIIEntity, ground_truth: List[PIIEntity]) -> bool:
        """Check if entity matches ground truth."""
        for gt_entity in ground_truth:
            if (entity.pii_type == gt_entity.pii_type and 
                entity.text.lower().strip() == gt_entity.text.lower().strip()):
                return True
        return False


class EntityFeatureExtractor:
    """Extract features from PII entities for meta-learning."""
    
    def extract_features(self, entity: PIIEntity) -> List[float]:
        """Extract numerical features from entity."""
        features = []
        
        # Basic features
        features.append(entity.confidence)  # Confidence score
        features.append(len(entity.text))   # Entity length
        features.append(len(entity.context)) # Context length
        
        # Entity type features (one-hot encoding)
        entity_types = [
            'person_name', 'email_address', 'phone_number', 'address',
            'date_of_birth', 'social_security_number', 'credit_card_number',
            'organization', 'location', 'misc_pii'
        ]
        
        for pii_type in entity_types:
            features.append(1.0 if entity.pii_type == pii_type else 0.0)
        
        # Extractor type features
        extractor_types = ['rule_based', 'layoutlm', 'donut', 'ner_model']
        for extractor_type in extractor_types:
            features.append(1.0 if extractor_type in entity.extractor.lower() else 0.0)
        
        # Position features (if available)
        if entity.start_pos > 0 and entity.end_pos > 0:
            features.append(entity.start_pos)  # Start position
            features.append(entity.end_pos - entity.start_pos)  # Length
        else:
            features.extend([0.0, 0.0])
        
        # Text characteristics
        features.append(float(entity.text.isupper()))   # All uppercase
        features.append(float(entity.text.islower()))   # All lowercase
        features.append(float(entity.text.isdigit()))   # All digits
        features.append(float('@' in entity.text))      # Contains @
        features.append(float('.' in entity.text))      # Contains dot
        features.append(float('-' in entity.text))      # Contains dash
        
        return features


class AdvancedEnsemble(PIIExtractorBase):
    """Advanced ensemble combining multiple extraction strategies."""
    
    def __init__(self, 
                 config: EnsembleConfig,
                 extractors: Optional[List[PIIExtractorBase]] = None):
        """Initialize advanced ensemble."""
        super().__init__("advanced_ensemble")
        
        self.config = config
        self.extractors = extractors or self._create_default_extractors()
        self.voting_system = EnsembleVotingSystem(config)
        self.meta_learner = MetaLearningEnsemble(config) if config.method == "meta_learning" else None
        
        logger.info(f"Advanced ensemble initialized with {len(self.extractors)} extractors")
    
    def _create_default_extractors(self) -> List[PIIExtractorBase]:
        """Create default set of extractors."""
        extractors = []
        
        try:
            # Rule-based extractor
            extractors.append(RuleBasedExtractor())
            
            # Layout-aware ensemble
            extractors.append(LayoutAwareEnsemble())
            
        except Exception as e:
            logger.warning(f"Failed to initialize some extractors: {e}")
        
        return extractors
    
    def extract_pii(self, document: Dict[str, Any]) -> PIIExtractionResult:
        """Extract PII using advanced ensemble methods."""
        try:
            # Run all extractors
            all_results = []
            processing_times = []
            
            for extractor in self.extractors:
                try:
                    result = extractor.extract_pii(document)
                    if result.error is None:
                        all_results.append(result)
                        processing_times.append(result.processing_time)
                except Exception as e:
                    logger.warning(f"Extractor {extractor.name} failed: {e}")
                    continue
            
            if not all_results:
                return PIIExtractionResult(error="All extractors failed")
            
            # Combine all entities
            all_entities = []
            for result in all_results:
                all_entities.extend(result.pii_entities)
            
            # Group similar entities
            entity_groups = self.voting_system.similarity_matcher.group_similar_entities(all_entities)
            
            # Apply ensemble method
            if self.config.method == "majority_vote":
                final_entities = self.voting_system.majority_vote(entity_groups)
            elif self.config.method == "weighted_vote":
                final_entities = self.voting_system.weighted_vote(entity_groups)
            elif self.config.method == "union":
                final_entities = self.voting_system.union_vote(entity_groups)
            elif self.config.method == "consensus":
                final_entities = self.voting_system.consensus_vote(entity_groups)
            elif self.config.method == "meta_learning" and self.meta_learner:
                final_entities = self.meta_learner.predict_with_meta_model(entity_groups)
            else:
                final_entities = self.voting_system.weighted_vote(entity_groups)
            
            return PIIExtractionResult(
                pii_entities=final_entities,
                processing_time=sum(processing_times),
                metadata={
                    'ensemble_method': self.config.method,
                    'num_extractors': len(self.extractors),
                    'num_groups': len(entity_groups),
                    'total_entities_found': len(all_entities),
                    'final_entities_selected': len(final_entities)
                }
            )
            
        except Exception as e:
            logger.error(f"Advanced ensemble extraction failed: {e}")
            return PIIExtractionResult(error=str(e))
    
    def train_meta_model(self, training_data_path: str):
        """Train meta-model if using meta-learning approach."""
        if self.meta_learner:
            # This would require ground truth data
            # For now, just log that training is needed
            logger.info("Meta-model training requires ground truth data")
            # self.meta_learner.train_meta_model(training_results)
    
    def evaluate_ensemble(self, test_documents: List[Dict[str, Any]], 
                         ground_truth: List[List[PIIEntity]]) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        all_predictions = []
        all_ground_truth = []
        
        for doc, gt in zip(test_documents, ground_truth):
            result = self.extract_pii(doc)
            
            # Convert to label format for evaluation
            pred_labels = [entity.pii_type for entity in result.pii_entities]
            gt_labels = [entity.pii_type for entity in gt]
            
            all_predictions.extend(pred_labels)
            all_ground_truth.extend(gt_labels)
        
        # Calculate metrics
        if all_predictions and all_ground_truth:
            f1 = f1_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
            precision = precision_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
            
            return {
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
        
        return {'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}