"""Model versioning and A/B testing framework."""

import json
import time
import hashlib
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a model version with metadata."""
    version_id: str
    model_name: str
    framework: str  # 'pytorch', 'sklearn', 'custom'
    created_at: datetime
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_path: str = ""
    mlflow_run_id: str = ""
    status: str = "active"  # active, deprecated, archived
    performance_benchmark: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelVersion':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class ExperimentConfig:
    """Configuration for A/B testing experiments."""
    experiment_name: str
    description: str
    start_date: datetime
    end_date: Optional[datetime] = None
    traffic_split: Dict[str, float] = field(default_factory=lambda: {"A": 0.5, "B": 0.5})
    success_metrics: List[str] = field(default_factory=lambda: ["f1_score", "precision", "recall"])
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    status: str = "planned"  # planned, running, completed, stopped
    
    def validate(self) -> bool:
        """Validate experiment configuration."""
        # Check traffic split sums to 1.0
        total_traffic = sum(self.traffic_split.values())
        if abs(total_traffic - 1.0) > 0.01:
            return False
        
        # Check dates
        if self.end_date and self.end_date <= self.start_date:
            return False
        
        return True


class ModelRegistry:
    """Registry for managing model versions and metadata."""
    
    def __init__(self, registry_path: str = "models/registry"):
        """Initialize model registry."""
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.registry_path / "registry.json"
        self.models = self._load_registry()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(str(self.registry_path / "mlruns"))
        
    def _load_registry(self) -> Dict[str, List[ModelVersion]]:
        """Load registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                
            registry = {}
            for model_name, versions_data in data.items():
                registry[model_name] = [
                    ModelVersion.from_dict(version_data) 
                    for version_data in versions_data
                ]
            return registry
        return {}
    
    def _save_registry(self):
        """Save registry to file."""
        data = {}
        for model_name, versions in self.models.items():
            data[model_name] = [version.to_dict() for version in versions]
        
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def register_model(self, 
                      model_name: str,
                      model_artifact: Any,
                      framework: str,
                      description: str = "",
                      tags: Dict[str, str] = None,
                      metrics: Dict[str, float] = None,
                      hyperparameters: Dict[str, Any] = None) -> ModelVersion:
        """Register a new model version."""
        
        tags = tags or {}
        metrics = metrics or {}
        hyperparameters = hyperparameters or {}
        
        # Generate version ID
        timestamp = str(int(time.time()))
        model_hash = hashlib.md5(str(hyperparameters).encode()).hexdigest()[:8]
        version_id = f"{model_name}_v{timestamp}_{model_hash}"
        
        # Start MLflow run
        with mlflow.start_run(run_name=version_id) as run:
            # Log parameters
            for key, value in hyperparameters.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log tags
            for key, value in tags.items():
                mlflow.set_tag(key, value)
            
            # Save model
            model_path = str(self.registry_path / "models" / version_id)
            Path(model_path).mkdir(parents=True, exist_ok=True)
            
            if framework == "pytorch":
                mlflow.pytorch.log_model(model_artifact, "model")
            elif framework == "sklearn":
                mlflow.sklearn.log_model(model_artifact, "model")
            else:
                # Custom model - save with joblib
                import joblib
                model_file = Path(model_path) / "model.pkl"
                joblib.dump(model_artifact, model_file)
                mlflow.log_artifact(str(model_file))
            
            mlflow_run_id = run.info.run_id
        
        # Create model version
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            framework=framework,
            created_at=datetime.now(),
            description=description,
            tags=tags,
            metrics=metrics,
            hyperparameters=hyperparameters,
            model_path=model_path,
            mlflow_run_id=mlflow_run_id
        )
        
        # Add to registry
        if model_name not in self.models:
            self.models[model_name] = []
        self.models[model_name].append(version)
        
        self._save_registry()
        
        logger.info(f"Registered model version: {version_id}")
        return version
    
    def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """Get all versions of a model."""
        return self.models.get(model_name, [])
    
    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get latest version of a model."""
        versions = self.get_model_versions(model_name)
        if not versions:
            return None
        
        return max(versions, key=lambda v: v.created_at)
    
    def get_best_version(self, model_name: str, metric: str = "f1_score") -> Optional[ModelVersion]:
        """Get best performing version of a model."""
        versions = self.get_model_versions(model_name)
        if not versions:
            return None
        
        valid_versions = [v for v in versions if metric in v.metrics]
        if not valid_versions:
            return None
        
        return max(valid_versions, key=lambda v: v.metrics[metric])
    
    def load_model(self, version_id: str) -> Any:
        """Load a specific model version."""
        for versions in self.models.values():
            for version in versions:
                if version.version_id == version_id:
                    if version.mlflow_run_id:
                        # Load from MLflow
                        model_uri = f"runs:/{version.mlflow_run_id}/model"
                        if version.framework == "pytorch":
                            return mlflow.pytorch.load_model(model_uri)
                        elif version.framework == "sklearn":
                            return mlflow.sklearn.load_model(model_uri)
                    else:
                        # Load from file
                        import joblib
                        model_file = Path(version.model_path) / "model.pkl"
                        return joblib.load(model_file)
        
        raise ValueError(f"Model version {version_id} not found")
    
    def deprecate_version(self, version_id: str):
        """Mark a model version as deprecated."""
        for versions in self.models.values():
            for version in versions:
                if version.version_id == version_id:
                    version.status = "deprecated"
                    self._save_registry()
                    logger.info(f"Deprecated model version: {version_id}")
                    return
        
        raise ValueError(f"Model version {version_id} not found")
    
    def compare_versions(self, version_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple model versions."""
        versions = []
        for version_id in version_ids:
            for model_versions in self.models.values():
                for version in model_versions:
                    if version.version_id == version_id:
                        versions.append(version)
                        break
        
        if len(versions) != len(version_ids):
            raise ValueError("Some version IDs not found")
        
        comparison = {
            "versions": [v.version_id for v in versions],
            "metrics_comparison": {},
            "hyperparameters_comparison": {},
            "performance_ranking": {}
        }
        
        # Compare metrics
        all_metrics = set()
        for version in versions:
            all_metrics.update(version.metrics.keys())
        
        for metric in all_metrics:
            comparison["metrics_comparison"][metric] = {
                v.version_id: v.metrics.get(metric, None) for v in versions
            }
        
        # Compare hyperparameters
        all_params = set()
        for version in versions:
            all_params.update(version.hyperparameters.keys())
        
        for param in all_params:
            comparison["hyperparameters_comparison"][param] = {
                v.version_id: v.hyperparameters.get(param, None) for v in versions
            }
        
        # Performance ranking
        for metric in all_metrics:
            metric_values = [(v.version_id, v.metrics.get(metric, 0)) for v in versions]
            metric_values.sort(key=lambda x: x[1], reverse=True)
            comparison["performance_ranking"][metric] = [v[0] for v in metric_values]
        
        return comparison


class ABTestingFramework:
    """Framework for A/B testing different model versions."""
    
    def __init__(self, registry: ModelRegistry):
        """Initialize A/B testing framework."""
        self.registry = registry
        self.experiments = {}
        self.experiment_results = {}
        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B testing experiment."""
        if not config.validate():
            raise ValueError("Invalid experiment configuration")
        
        experiment_id = f"exp_{config.experiment_name}_{int(time.time())}"
        self.experiments[experiment_id] = config
        
        logger.info(f"Created experiment: {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str, model_versions: Dict[str, str]):
        """Start an A/B testing experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        
        # Validate model versions exist
        for variant, version_id in model_versions.items():
            if variant not in config.traffic_split:
                raise ValueError(f"Variant {variant} not in traffic split configuration")
            
            # Check if version exists
            found = False
            for versions in self.registry.models.values():
                if any(v.version_id == version_id for v in versions):
                    found = True
                    break
            if not found:
                raise ValueError(f"Model version {version_id} not found")
        
        # Initialize experiment results
        self.experiment_results[experiment_id] = {
            "model_versions": model_versions,
            "start_time": datetime.now(),
            "samples": {variant: [] for variant in config.traffic_split.keys()},
            "metrics": {variant: {} for variant in config.traffic_split.keys()},
            "status": "running"
        }
        
        config.status = "running"
        logger.info(f"Started experiment: {experiment_id}")
    
    def route_request(self, experiment_id: str, request_id: str = None) -> str:
        """Route request to appropriate model variant."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        if config.status != "running":
            raise ValueError(f"Experiment {experiment_id} is not running")
        
        # Determine variant based on traffic split
        if request_id:
            # Deterministic routing based on request ID hash
            hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
            random_value = (hash_value % 10000) / 10000.0
        else:
            # Random routing
            random_value = np.random.random()
        
        cumulative_prob = 0
        for variant, probability in config.traffic_split.items():
            cumulative_prob += probability
            if random_value <= cumulative_prob:
                return variant
        
        # Fallback to last variant
        return list(config.traffic_split.keys())[-1]
    
    def record_result(self, 
                     experiment_id: str, 
                     variant: str, 
                     ground_truth: List[Any], 
                     predictions: List[Any],
                     additional_metrics: Dict[str, float] = None):
        """Record experiment result for a variant."""
        if experiment_id not in self.experiment_results:
            raise ValueError(f"Experiment {experiment_id} not found or not started")
        
        results = self.experiment_results[experiment_id]
        additional_metrics = additional_metrics or {}
        
        # Calculate standard metrics
        metrics = {}
        try:
            metrics["accuracy"] = accuracy_score(ground_truth, predictions)
            metrics["precision"] = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
            metrics["recall"] = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
            metrics["f1_score"] = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
        except Exception as e:
            logger.warning(f"Could not calculate standard metrics: {e}")
        
        # Add additional metrics
        metrics.update(additional_metrics)
        
        # Store sample
        sample = {
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "sample_size": len(ground_truth)
        }
        
        results["samples"][variant].append(sample)
        
        # Update aggregate metrics
        if variant not in results["metrics"]:
            results["metrics"][variant] = {}
        
        # Calculate running averages
        samples = results["samples"][variant]
        total_samples = sum(s["sample_size"] for s in samples)
        
        for metric_name in metrics.keys():
            weighted_sum = sum(s["metrics"].get(metric_name, 0) * s["sample_size"] for s in samples)
            results["metrics"][variant][metric_name] = weighted_sum / total_samples if total_samples > 0 else 0
        
        results["metrics"][variant]["total_samples"] = total_samples
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results."""
        if experiment_id not in self.experiment_results:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        results = self.experiment_results[experiment_id]
        
        analysis = {
            "experiment_id": experiment_id,
            "status": config.status,
            "duration": None,
            "variants": list(config.traffic_split.keys()),
            "metrics_comparison": {},
            "statistical_significance": {},
            "recommendations": []
        }
        
        # Calculate duration
        if results.get("start_time"):
            duration = datetime.now() - results["start_time"]
            analysis["duration"] = str(duration)
        
        # Compare metrics across variants
        all_metrics = set()
        for variant_metrics in results["metrics"].values():
            all_metrics.update(variant_metrics.keys())
        
        for metric in all_metrics:
            analysis["metrics_comparison"][metric] = {}
            for variant in config.traffic_split.keys():
                value = results["metrics"].get(variant, {}).get(metric, 0)
                analysis["metrics_comparison"][metric][variant] = value
        
        # Statistical significance testing (simplified)
        for metric in config.success_metrics:
            if metric in analysis["metrics_comparison"]:
                variant_values = analysis["metrics_comparison"][metric]
                if len(variant_values) >= 2:
                    variants = list(variant_values.keys())
                    variant_a, variant_b = variants[0], variants[1]
                    
                    value_a = variant_values[variant_a]
                    value_b = variant_values[variant_b]
                    
                    # Simple difference test (could be improved with proper statistical tests)
                    relative_difference = abs(value_a - value_b) / max(value_a, value_b, 0.001)
                    
                    analysis["statistical_significance"][metric] = {
                        "variants_compared": [variant_a, variant_b],
                        "values": [value_a, value_b],
                        "relative_difference": relative_difference,
                        "significant": relative_difference > 0.05  # 5% threshold
                    }
        
        # Generate recommendations
        for metric in config.success_metrics:
            if metric in analysis["metrics_comparison"]:
                variant_values = analysis["metrics_comparison"][metric]
                best_variant = max(variant_values.keys(), key=lambda v: variant_values[v])
                analysis["recommendations"].append(
                    f"For {metric}, variant {best_variant} performs best with {variant_values[best_variant]:.4f}"
                )
        
        return analysis
    
    def stop_experiment(self, experiment_id: str):
        """Stop a running experiment."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        config = self.experiments[experiment_id]
        config.status = "completed"
        
        if experiment_id in self.experiment_results:
            self.experiment_results[experiment_id]["status"] = "completed"
            self.experiment_results[experiment_id]["end_time"] = datetime.now()
        
        logger.info(f"Stopped experiment: {experiment_id}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        summary = {
            "total_experiments": len(self.experiments),
            "running_experiments": 0,
            "completed_experiments": 0,
            "experiments": []
        }
        
        for exp_id, config in self.experiments.items():
            exp_summary = {
                "experiment_id": exp_id,
                "name": config.experiment_name,
                "status": config.status,
                "start_date": config.start_date.isoformat(),
                "variants": list(config.traffic_split.keys())
            }
            
            if config.status == "running":
                summary["running_experiments"] += 1
            elif config.status == "completed":
                summary["completed_experiments"] += 1
            
            if exp_id in self.experiment_results:
                results = self.experiment_results[exp_id]
                total_samples = sum(
                    results["metrics"].get(v, {}).get("total_samples", 0)
                    for v in config.traffic_split.keys()
                )
                exp_summary["total_samples"] = total_samples
            
            summary["experiments"].append(exp_summary)
        
        return summary


def create_example_experiment():
    """Create an example A/B testing experiment."""
    # Initialize registry and framework
    registry = ModelRegistry("models/test_registry")
    ab_framework = ABTestingFramework(registry)
    
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="layout_vs_rule_based",
        description="Compare LayoutLM-based extractor vs rule-based extractor",
        start_date=datetime.now(),
        traffic_split={"rule_based": 0.5, "layout_aware": 0.5},
        success_metrics=["f1_score", "precision", "recall"],
        minimum_sample_size=500
    )
    
    # Create experiment
    experiment_id = ab_framework.create_experiment(config)
    
    logger.info(f"Example experiment created: {experiment_id}")
    return experiment_id, ab_framework