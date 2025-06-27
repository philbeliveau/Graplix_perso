"""
Confidence-based Prompt Router for Vision-LLM PII Extraction

This module provides intelligent routing of extraction requests to optimal LLM models
based on document characteristics, confidence requirements, and performance metrics.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import defaultdict, deque

from .vision_document_classifier import DocumentClassification, DocumentDifficulty, DocumentDomain
from ..llm.multimodal_llm_service import llm_service

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model performance tiers"""
    FAST = "fast"           # Fast, cost-effective models
    BALANCED = "balanced"   # Good balance of speed, cost, and accuracy
    ACCURACY = "accuracy"   # High-accuracy models, slower and more expensive
    PREMIUM = "premium"     # Best available models for critical tasks


class RoutingStrategy(Enum):
    """Routing strategies for model selection"""
    PERFORMANCE_FIRST = "performance_first"    # Prioritize accuracy
    COST_FIRST = "cost_first"                 # Prioritize cost efficiency
    SPEED_FIRST = "speed_first"               # Prioritize processing speed
    BALANCED = "balanced"                     # Balance all factors
    ADAPTIVE = "adaptive"                     # Learn from past performance


@dataclass
class ModelPerformanceMetrics:
    """Track model performance metrics"""
    model_key: str
    total_requests: int = 0
    successful_requests: int = 0
    average_processing_time: float = 0.0
    average_confidence: float = 0.0
    average_cost: float = 0.0
    error_rate: float = 0.0
    domain_performance: Dict[str, float] = field(default_factory=dict)
    difficulty_performance: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    
    def update_metrics(self, 
                      processing_time: float,
                      confidence: float,
                      cost: float,
                      success: bool,
                      domain: str = None,
                      difficulty: str = None):
        """Update performance metrics with new data point"""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            
            # Update running averages
            alpha = 0.1  # Learning rate for exponential moving average
            self.average_processing_time = (1 - alpha) * self.average_processing_time + alpha * processing_time
            self.average_confidence = (1 - alpha) * self.average_confidence + alpha * confidence
            self.average_cost = (1 - alpha) * self.average_cost + alpha * cost
            
            # Update domain performance
            if domain:
                if domain not in self.domain_performance:
                    self.domain_performance[domain] = confidence
                else:
                    self.domain_performance[domain] = (1 - alpha) * self.domain_performance[domain] + alpha * confidence
            
            # Update difficulty performance
            if difficulty:
                if difficulty not in self.difficulty_performance:
                    self.difficulty_performance[difficulty] = confidence
                else:
                    self.difficulty_performance[difficulty] = (1 - alpha) * self.difficulty_performance[difficulty] + alpha * confidence
        
        self.error_rate = 1.0 - (self.successful_requests / self.total_requests)
        self.last_updated = time.time()
    
    def get_score(self, 
                 strategy: RoutingStrategy,
                 domain: str = None,
                 difficulty: str = None) -> float:
        """Calculate routing score based on strategy and context"""
        if self.total_requests == 0:
            return 0.5  # Neutral score for untested models
        
        # Base scores (0-1)
        accuracy_score = self.average_confidence
        speed_score = max(0, 1 - min(self.average_processing_time / 30.0, 1))  # Normalize to 30s max
        cost_score = max(0, 1 - min(self.average_cost / 0.01, 1))  # Normalize to $0.01 max
        reliability_score = 1 - self.error_rate
        
        # Context-specific adjustments
        context_bonus = 0.0
        if domain and domain in self.domain_performance:
            context_bonus += (self.domain_performance[domain] - 0.5) * 0.1
        if difficulty and difficulty in self.difficulty_performance:
            context_bonus += (self.difficulty_performance[difficulty] - 0.5) * 0.1
        
        # Strategy-based weighting
        if strategy == RoutingStrategy.PERFORMANCE_FIRST:
            score = 0.5 * accuracy_score + 0.3 * reliability_score + 0.15 * speed_score + 0.05 * cost_score
        elif strategy == RoutingStrategy.COST_FIRST:
            score = 0.5 * cost_score + 0.25 * reliability_score + 0.15 * accuracy_score + 0.1 * speed_score
        elif strategy == RoutingStrategy.SPEED_FIRST:
            score = 0.5 * speed_score + 0.25 * reliability_score + 0.15 * accuracy_score + 0.1 * cost_score
        elif strategy == RoutingStrategy.BALANCED:
            score = 0.25 * accuracy_score + 0.25 * speed_score + 0.25 * cost_score + 0.25 * reliability_score
        else:  # ADAPTIVE
            # Adaptive strategy learns optimal weights
            score = 0.3 * accuracy_score + 0.25 * reliability_score + 0.25 * speed_score + 0.2 * cost_score
        
        return min(1.0, max(0.0, score + context_bonus))


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    selected_model: str
    confidence: float
    reason: str
    alternatives: List[str]
    estimated_cost: float
    estimated_time: float
    strategy_used: RoutingStrategy
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptRouter:
    """
    Intelligent router for selecting optimal LLM models based on document
    characteristics, performance history, and routing strategies.
    """
    
    def __init__(self,
                 default_strategy: RoutingStrategy = RoutingStrategy.BALANCED,
                 enable_learning: bool = True,
                 max_history: int = 1000):
        """
        Initialize the Prompt Router
        
        Args:
            default_strategy: Default routing strategy
            enable_learning: Whether to enable adaptive learning
            max_history: Maximum number of performance records to keep
        """
        self.default_strategy = default_strategy
        self.enable_learning = enable_learning
        self.max_history = max_history
        
        # Performance tracking
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        self.routing_history = deque(maxlen=max_history)
        
        # Model tiers configuration
        self.model_tiers = {
            ModelTier.FAST: [
                "gpt-4o-mini",
                "claude-3-5-haiku-20241022",
                "gemini-1.5-flash"
            ],
            ModelTier.BALANCED: [
                "gpt-4o",
                "claude-3-5-sonnet-20241022", 
                "gemini-1.5-pro"
            ],
            ModelTier.ACCURACY: [
                "gpt-4-turbo",
                "claude-3-opus-20240229"
            ],
            ModelTier.PREMIUM: [
                "gpt-4",
                "claude-3-opus-20240229"
            ]
        }
        
        # Domain-specific model preferences
        self.domain_preferences = {
            DocumentDomain.MEDICAL: {
                'preferred_tier': ModelTier.ACCURACY,
                'min_confidence': 0.85,
                'fallback_strategy': RoutingStrategy.PERFORMANCE_FIRST
            },
            DocumentDomain.LEGAL: {
                'preferred_tier': ModelTier.ACCURACY,
                'min_confidence': 0.8,
                'fallback_strategy': RoutingStrategy.PERFORMANCE_FIRST
            },
            DocumentDomain.FINANCE: {
                'preferred_tier': ModelTier.BALANCED,
                'min_confidence': 0.8,
                'fallback_strategy': RoutingStrategy.BALANCED
            },
            DocumentDomain.GOVERNMENT: {
                'preferred_tier': ModelTier.ACCURACY,
                'min_confidence': 0.85,
                'fallback_strategy': RoutingStrategy.PERFORMANCE_FIRST
            },
            DocumentDomain.HR: {
                'preferred_tier': ModelTier.BALANCED,
                'min_confidence': 0.75,
                'fallback_strategy': RoutingStrategy.BALANCED
            }
        }
        
        # Initialize model metrics for available models
        self._initialize_model_metrics()
        
        logger.info(f"PromptRouter initialized with strategy: {default_strategy.value}")
    
    def route_request(self,
                     classification: Optional[DocumentClassification] = None,
                     strategy: Optional[RoutingStrategy] = None,
                     min_confidence: Optional[float] = None,
                     max_cost: Optional[float] = None,
                     max_time: Optional[float] = None,
                     exclude_models: Optional[List[str]] = None) -> RoutingDecision:
        """
        Route a PII extraction request to the optimal model
        
        Args:
            classification: Document classification result
            strategy: Routing strategy override
            min_confidence: Minimum required confidence
            max_cost: Maximum acceptable cost
            max_time: Maximum acceptable processing time
            exclude_models: Models to exclude from selection
            
        Returns:
            RoutingDecision with selected model and reasoning
        """
        start_time = time.time()
        
        # Use provided strategy or default
        routing_strategy = strategy or self.default_strategy
        
        # Get domain-specific preferences
        domain_prefs = {}
        if classification and classification.domain in self.domain_preferences:
            domain_prefs = self.domain_preferences[classification.domain]
            
            # Override strategy if domain has preference
            if 'fallback_strategy' in domain_prefs and not strategy:
                routing_strategy = domain_prefs['fallback_strategy']
        
        # Determine confidence requirement
        required_confidence = min_confidence or domain_prefs.get('min_confidence', 0.7)
        
        # Get candidate models
        candidates = self._get_candidate_models(
            classification=classification,
            strategy=routing_strategy,
            domain_prefs=domain_prefs,
            exclude_models=exclude_models or []
        )
        
        if not candidates:
            logger.warning("No candidate models available for routing")
            return RoutingDecision(
                selected_model="gpt-4o-mini",  # Fallback
                confidence=0.5,
                reason="No suitable models available - using fallback",
                alternatives=[],
                estimated_cost=0.001,
                estimated_time=5.0,
                strategy_used=routing_strategy,
                metadata={'fallback_used': True}
            )
        
        # Score and rank candidates
        scored_candidates = []
        for model_key in candidates:
            if model_key not in self.model_metrics:
                self._initialize_model_metrics_for(model_key)
            
            metrics = self.model_metrics[model_key]
            domain_str = classification.domain.value if classification else None
            difficulty_str = classification.difficulty.value if classification else None
            
            score = metrics.get_score(
                strategy=routing_strategy,
                domain=domain_str,
                difficulty=difficulty_str
            )
            
            # Apply constraints
            if max_cost and metrics.average_cost > max_cost:
                score *= 0.5  # Penalize expensive models
            
            if max_time and metrics.average_processing_time > max_time:
                score *= 0.5  # Penalize slow models
            
            scored_candidates.append({
                'model': model_key,
                'score': score,
                'metrics': metrics
            })
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Select best model
        best_candidate = scored_candidates[0]
        selected_model = best_candidate['model']
        selected_metrics = best_candidate['metrics']
        
        # Create routing decision
        decision = RoutingDecision(
            selected_model=selected_model,
            confidence=best_candidate['score'],
            reason=self._create_selection_reason(best_candidate, routing_strategy, classification),
            alternatives=[c['model'] for c in scored_candidates[1:6]],  # Top 5 alternatives
            estimated_cost=selected_metrics.average_cost,
            estimated_time=selected_metrics.average_processing_time,
            strategy_used=routing_strategy,
            metadata={
                'total_candidates': len(candidates),
                'selection_time': time.time() - start_time,
                'domain_preferences_applied': bool(domain_prefs),
                'required_confidence': required_confidence
            }
        )
        
        # Record routing decision
        self.routing_history.append({
            'timestamp': time.time(),
            'decision': decision,
            'classification': classification,
            'constraints': {
                'min_confidence': min_confidence,
                'max_cost': max_cost,
                'max_time': max_time
            }
        })
        
        logger.info(f"Routed to {selected_model} (score: {best_candidate['score']:.3f}) "
                   f"using {routing_strategy.value} strategy")
        
        return decision
    
    def record_result(self,
                     model_key: str,
                     processing_time: float,
                     confidence: float,
                     cost: float,
                     success: bool,
                     classification: Optional[DocumentClassification] = None):
        """
        Record the result of a model execution for learning
        
        Args:
            model_key: Model that was used
            processing_time: Time taken for processing
            confidence: Confidence of the result
            cost: Cost of the operation
            success: Whether the extraction was successful
            classification: Document classification if available
        """
        if not self.enable_learning:
            return
        
        if model_key not in self.model_metrics:
            self._initialize_model_metrics_for(model_key)
        
        domain_str = classification.domain.value if classification else None
        difficulty_str = classification.difficulty.value if classification else None
        
        self.model_metrics[model_key].update_metrics(
            processing_time=processing_time,
            confidence=confidence,
            cost=cost,
            success=success,
            domain=domain_str,
            difficulty=difficulty_str
        )
        
        logger.debug(f"Updated metrics for {model_key}: "
                    f"success={success}, confidence={confidence:.3f}, "
                    f"time={processing_time:.2f}s, cost=${cost:.4f}")
    
    def get_model_recommendations(self,
                                classification: Optional[DocumentClassification] = None,
                                strategy: Optional[RoutingStrategy] = None) -> Dict[str, Any]:
        """
        Get model recommendations for a given scenario
        
        Args:
            classification: Document classification
            strategy: Routing strategy
            
        Returns:
            Dictionary with model recommendations and analysis
        """
        routing_strategy = strategy or self.default_strategy
        
        # Get all available models with scores
        all_models = list(self.model_metrics.keys())
        recommendations = []
        
        for model_key in all_models:
            metrics = self.model_metrics[model_key]
            domain_str = classification.domain.value if classification else None
            difficulty_str = classification.difficulty.value if classification else None
            
            score = metrics.get_score(
                strategy=routing_strategy,
                domain=domain_str,
                difficulty=difficulty_str
            )
            
            recommendations.append({
                'model': model_key,
                'score': score,
                'tier': self._get_model_tier(model_key),
                'performance_summary': {
                    'success_rate': 1 - metrics.error_rate,
                    'avg_confidence': metrics.average_confidence,
                    'avg_processing_time': metrics.average_processing_time,
                    'avg_cost': metrics.average_cost,
                    'total_requests': metrics.total_requests
                }
            })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'strategy': routing_strategy.value,
            'classification': classification.to_dict() if classification else None,
            'recommendations': recommendations,
            'top_3_models': [r['model'] for r in recommendations[:3]],
            'analysis': {
                'best_for_accuracy': max(recommendations, key=lambda x: x['performance_summary']['avg_confidence'])['model'],
                'best_for_speed': min(recommendations, key=lambda x: x['performance_summary']['avg_processing_time'])['model'],
                'best_for_cost': min(recommendations, key=lambda x: x['performance_summary']['avg_cost'])['model']
            }
        }
    
    def _initialize_model_metrics(self):
        """Initialize performance metrics for all available models"""
        available_models = llm_service.get_available_models()
        
        for model_key in available_models:
            self._initialize_model_metrics_for(model_key)
        
        logger.info(f"Initialized metrics for {len(available_models)} models")
    
    def _initialize_model_metrics_for(self, model_key: str):
        """Initialize metrics for a specific model"""
        if model_key not in self.model_metrics:
            self.model_metrics[model_key] = ModelPerformanceMetrics(model_key=model_key)
    
    def _get_candidate_models(self,
                            classification: Optional[DocumentClassification],
                            strategy: RoutingStrategy,
                            domain_prefs: Dict[str, Any],
                            exclude_models: List[str]) -> List[str]:
        """Get candidate models based on criteria"""
        candidates = []
        
        # Start with preferred tier if specified
        if 'preferred_tier' in domain_prefs:
            preferred_tier = domain_prefs['preferred_tier']
            candidates.extend(self.model_tiers.get(preferred_tier, []))
        
        # Add models based on strategy
        if strategy == RoutingStrategy.SPEED_FIRST:
            candidates.extend(self.model_tiers[ModelTier.FAST])
            candidates.extend(self.model_tiers[ModelTier.BALANCED])
        elif strategy == RoutingStrategy.COST_FIRST:
            candidates.extend(self.model_tiers[ModelTier.FAST])
            candidates.extend(self.model_tiers[ModelTier.BALANCED])
        elif strategy == RoutingStrategy.PERFORMANCE_FIRST:
            candidates.extend(self.model_tiers[ModelTier.ACCURACY])
            candidates.extend(self.model_tiers[ModelTier.PREMIUM])
            candidates.extend(self.model_tiers[ModelTier.BALANCED])
        else:  # BALANCED or ADAPTIVE
            for tier in [ModelTier.BALANCED, ModelTier.FAST, ModelTier.ACCURACY]:
                candidates.extend(self.model_tiers[tier])
        
        # Filter out excluded models and unavailable models
        available_models = set(llm_service.get_available_models())
        candidates = [m for m in candidates if m not in exclude_models and m in available_models]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for model in candidates:
            if model not in seen:
                seen.add(model)
                unique_candidates.append(model)
        
        return unique_candidates
    
    def _get_model_tier(self, model_key: str) -> str:
        """Get the tier of a model"""
        for tier, models in self.model_tiers.items():
            if model_key in models:
                return tier.value
        return "unknown"
    
    def _create_selection_reason(self,
                               candidate: Dict[str, Any],
                               strategy: RoutingStrategy,
                               classification: Optional[DocumentClassification]) -> str:
        """Create human-readable reason for model selection"""
        model = candidate['model']
        score = candidate['score']
        metrics = candidate['metrics']
        
        reason_parts = [f"Selected {model} (score: {score:.3f})"]
        
        if strategy == RoutingStrategy.PERFORMANCE_FIRST:
            reason_parts.append(f"prioritizing accuracy (avg confidence: {metrics.average_confidence:.3f})")
        elif strategy == RoutingStrategy.SPEED_FIRST:
            reason_parts.append(f"prioritizing speed (avg time: {metrics.average_processing_time:.2f}s)")
        elif strategy == RoutingStrategy.COST_FIRST:
            reason_parts.append(f"prioritizing cost efficiency (avg cost: ${metrics.average_cost:.4f})")
        else:
            reason_parts.append("using balanced criteria")
        
        if classification:
            if classification.domain.value in metrics.domain_performance:
                domain_perf = metrics.domain_performance[classification.domain.value]
                reason_parts.append(f"good {classification.domain.value} performance ({domain_perf:.3f})")
        
        if metrics.total_requests > 0:
            success_rate = 1 - metrics.error_rate
            reason_parts.append(f"{success_rate:.1%} success rate")
        
        return "; ".join(reason_parts)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics for monitoring and analysis"""
        total_routes = len(self.routing_history)
        
        if total_routes == 0:
            return {
                'total_routing_decisions': 0,
                'model_usage': {},
                'strategy_usage': {},
                'performance_summary': {}
            }
        
        # Analyze routing history
        model_usage = defaultdict(int)
        strategy_usage = defaultdict(int)
        
        for record in self.routing_history:
            decision = record['decision']
            model_usage[decision.selected_model] += 1
            strategy_usage[decision.strategy_used.value] += 1
        
        # Calculate model performance summary
        performance_summary = {}
        for model_key, metrics in self.model_metrics.items():
            if metrics.total_requests > 0:
                performance_summary[model_key] = {
                    'total_requests': metrics.total_requests,
                    'success_rate': 1 - metrics.error_rate,
                    'avg_confidence': metrics.average_confidence,
                    'avg_processing_time': metrics.average_processing_time,
                    'avg_cost': metrics.average_cost,
                    'last_used': metrics.last_updated
                }
        
        return {
            'total_routing_decisions': total_routes,
            'model_usage': dict(model_usage),
            'strategy_usage': dict(strategy_usage),
            'performance_summary': performance_summary,
            'learning_enabled': self.enable_learning,
            'default_strategy': self.default_strategy.value
        }