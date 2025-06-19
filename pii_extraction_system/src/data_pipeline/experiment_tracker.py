"""
Experiment Tracker Module

Manages experiment results, versioning, and performance tracking for ML experiments.
Provides comprehensive logging and comparison capabilities.
"""

import json
import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import existing components
import sys
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.logging_config import get_logger
from .batch_processor import BatchResult

logger = get_logger(__name__)

class ExperimentStatus(Enum):
    """Experiment execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ExperimentType(Enum):
    """Types of experiments."""
    BASELINE = "baseline"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_TUNING = "model_tuning"
    ABLATION_STUDY = "ablation_study"
    COMPARISON = "comparison"
    VALIDATION = "validation"

@dataclass
class ExperimentConfiguration:
    """Experiment configuration and parameters."""
    
    # Basic configuration
    experiment_name: str
    experiment_type: ExperimentType
    description: str = ""
    
    # Model and pipeline configuration
    model_config: Dict = field(default_factory=dict)
    pipeline_config: Dict = field(default_factory=dict)
    preprocessing_config: Dict = field(default_factory=dict)
    
    # Data configuration
    dataset_version: str = ""
    data_splits: Dict[str, List[str]] = field(default_factory=dict)
    
    # Hyperparameters
    hyperparameters: Dict = field(default_factory=dict)
    
    # Environment info
    environment_info: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'experiment_name': self.experiment_name,
            'experiment_type': self.experiment_type.value,
            'description': self.description,
            'model_config': self.model_config,
            'pipeline_config': self.pipeline_config,
            'preprocessing_config': self.preprocessing_config,
            'dataset_version': self.dataset_version,
            'data_splits': self.data_splits,
            'hyperparameters': self.hyperparameters,
            'environment_info': self.environment_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentConfiguration':
        """Create instance from dictionary."""
        return cls(
            experiment_name=data['experiment_name'],
            experiment_type=ExperimentType(data['experiment_type']),
            description=data.get('description', ''),
            model_config=data.get('model_config', {}),
            pipeline_config=data.get('pipeline_config', {}),
            preprocessing_config=data.get('preprocessing_config', {}),
            dataset_version=data.get('dataset_version', ''),
            data_splits=data.get('data_splits', {}),
            hyperparameters=data.get('hyperparameters', {}),
            environment_info=data.get('environment_info', {})
        )

@dataclass
class ExperimentMetrics:
    """Experiment performance metrics."""
    
    # Primary metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Document processing metrics
    processing_time: Optional[float] = None
    throughput: Optional[float] = None
    success_rate: Optional[float] = None
    
    # PII-specific metrics
    pii_detection_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    entity_level_accuracy: Optional[float] = None
    
    # Quality metrics
    confidence_score: Optional[float] = None
    consistency_score: Optional[float] = None
    
    # Custom metrics
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'processing_time': self.processing_time,
            'throughput': self.throughput,
            'success_rate': self.success_rate,
            'pii_detection_rate': self.pii_detection_rate,
            'false_positive_rate': self.false_positive_rate,
            'entity_level_accuracy': self.entity_level_accuracy,
            'confidence_score': self.confidence_score,
            'consistency_score': self.consistency_score,
            'custom_metrics': self.custom_metrics
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentMetrics':
        """Create instance from dictionary."""
        return cls(
            accuracy=data.get('accuracy'),
            precision=data.get('precision'),
            recall=data.get('recall'),
            f1_score=data.get('f1_score'),
            processing_time=data.get('processing_time'),
            throughput=data.get('throughput'),
            success_rate=data.get('success_rate'),
            pii_detection_rate=data.get('pii_detection_rate'),
            false_positive_rate=data.get('false_positive_rate'),
            entity_level_accuracy=data.get('entity_level_accuracy'),
            confidence_score=data.get('confidence_score'),
            consistency_score=data.get('consistency_score'),
            custom_metrics=data.get('custom_metrics', {})
        )

@dataclass
class ExperimentResult:
    """Complete experiment result with all metadata."""
    
    # Experiment identification
    experiment_id: str
    experiment_name: str
    configuration: ExperimentConfiguration
    
    # Execution metadata
    status: ExperimentStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # Results
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)
    batch_results: List[BatchResult] = field(default_factory=list)
    
    # Outputs and artifacts
    model_artifacts: Dict[str, str] = field(default_factory=dict)  # artifact_name -> file_path
    result_files: Dict[str, str] = field(default_factory=dict)  # result_type -> file_path
    logs: List[str] = field(default_factory=list)
    
    # Quality and validation
    validation_results: Dict = field(default_factory=dict)
    error_analysis: Dict = field(default_factory=dict)
    
    # Versioning and reproducibility
    code_version: str = ""
    data_version: str = ""
    config_hash: str = ""
    
    # Notes and metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate configuration hash for reproducibility
        if self.configuration:
            config_str = json.dumps(self.configuration.to_dict(), sort_keys=True)
            self.config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def mark_completed(self):
        """Mark experiment as completed."""
        self.status = ExperimentStatus.COMPLETED
        self.end_time = datetime.now()
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
    
    def mark_failed(self, error_message: str = ""):
        """Mark experiment as failed."""
        self.status = ExperimentStatus.FAILED
        self.end_time = datetime.now()
        if error_message:
            self.logs.append(f"ERROR: {error_message}")
    
    def add_batch_result(self, batch_result: BatchResult):
        """Add a batch processing result."""
        self.batch_results.append(batch_result)
        self._update_metrics_from_batch()
    
    def _update_metrics_from_batch(self):
        """Update experiment metrics based on batch results."""
        if not self.batch_results:
            return
        
        # Aggregate processing metrics
        total_time = sum(br.total_processing_time for br in self.batch_results)
        total_docs = sum(len(br.results) for br in self.batch_results)
        successful_docs = sum(br.status.successful_documents for br in self.batch_results)
        
        self.metrics.processing_time = total_time
        self.metrics.throughput = total_docs / total_time if total_time > 0 else 0
        self.metrics.success_rate = successful_docs / total_docs if total_docs > 0 else 0
        
        # Aggregate confidence scores
        all_confidences = []
        for br in self.batch_results:
            for result in br.results:
                if result.success and result.confidence_score > 0:
                    all_confidences.append(result.confidence_score)
        
        if all_confidences:
            self.metrics.confidence_score = sum(all_confidences) / len(all_confidences)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'configuration': self.configuration.to_dict(),
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration,
            'metrics': self.metrics.to_dict(),
            'batch_results': [br.to_dict() for br in self.batch_results],
            'model_artifacts': self.model_artifacts,
            'result_files': self.result_files,
            'logs': self.logs,
            'validation_results': self.validation_results,
            'error_analysis': self.error_analysis,
            'code_version': self.code_version,
            'data_version': self.data_version,
            'config_hash': self.config_hash,
            'notes': self.notes,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentResult':
        """Create instance from dictionary."""
        result = cls(
            experiment_id=data['experiment_id'],
            experiment_name=data['experiment_name'],
            configuration=ExperimentConfiguration.from_dict(data['configuration']),
            status=ExperimentStatus(data['status']),
            start_time=datetime.fromisoformat(data['start_time'].replace('Z', '+00:00')),
            end_time=datetime.fromisoformat(data['end_time'].replace('Z', '+00:00')) if data.get('end_time') else None,
            duration=data.get('duration'),
            metrics=ExperimentMetrics.from_dict(data.get('metrics', {})),
            model_artifacts=data.get('model_artifacts', {}),
            result_files=data.get('result_files', {}),
            logs=data.get('logs', []),
            validation_results=data.get('validation_results', {}),
            error_analysis=data.get('error_analysis', {}),
            code_version=data.get('code_version', ''),
            data_version=data.get('data_version', ''),
            config_hash=data.get('config_hash', ''),
            notes=data.get('notes', ''),
            tags=data.get('tags', [])
        )
        
        # Note: batch_results would need custom deserialization
        # For now, we'll skip them in deserialization
        
        return result


class ExperimentTracker:
    """
    Comprehensive experiment tracking and management system.
    """
    
    def __init__(self, storage_path: Union[str, Path] = None):
        """
        Initialize experiment tracker.
        
        Args:
            storage_path: Path to store experiment data
        """
        if storage_path is None:
            self.storage_path = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/memory/experiments")
        else:
            self.storage_path = Path(storage_path)
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.experiments_cache: Dict[str, ExperimentResult] = {}
        
        # Load existing experiments
        self._load_existing_experiments()
        
        logger.info(f"ExperimentTracker initialized with storage at: {self.storage_path}")
    
    def create_experiment(self, 
                         configuration: ExperimentConfiguration,
                         experiment_id: str = None) -> ExperimentResult:
        """Create a new experiment."""
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())
        
        experiment = ExperimentResult(
            experiment_id=experiment_id,
            experiment_name=configuration.experiment_name,
            configuration=configuration,
            status=ExperimentStatus.PENDING,
            start_time=datetime.now()
        )
        
        # Cache and save
        self.experiments_cache[experiment_id] = experiment
        self._save_experiment(experiment)
        
        logger.info(f"Created experiment {experiment_id}: {configuration.experiment_name}")
        return experiment
    
    def start_experiment(self, experiment_id: str):
        """Mark experiment as started."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now()
        
        self._save_experiment(experiment)
        logger.info(f"Started experiment {experiment_id}")
    
    def complete_experiment(self, experiment_id: str, final_metrics: ExperimentMetrics = None):
        """Mark experiment as completed with optional final metrics."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.mark_completed()
        
        if final_metrics:
            experiment.metrics = final_metrics
        
        self._save_experiment(experiment)
        logger.info(f"Completed experiment {experiment_id} in {experiment.duration:.2f} seconds")
    
    def fail_experiment(self, experiment_id: str, error_message: str = ""):
        """Mark experiment as failed."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.mark_failed(error_message)
        
        self._save_experiment(experiment)
        logger.error(f"Failed experiment {experiment_id}: {error_message}")
    
    def log_batch_result(self, experiment_id: str, batch_result: BatchResult):
        """Log a batch processing result to an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.add_batch_result(batch_result)
        self._save_experiment(experiment)
        
        logger.debug(f"Added batch result to experiment {experiment_id}")
    
    def update_experiment_metrics(self, experiment_id: str, metrics: ExperimentMetrics):
        """Update experiment metrics."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.metrics = metrics
        self._save_experiment(experiment)
        
        logger.info(f"Updated metrics for experiment {experiment_id}")
    
    def add_experiment_artifact(self, experiment_id: str, artifact_name: str, file_path: str):
        """Add an artifact (model, results, etc.) to an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        experiment.model_artifacts[artifact_name] = file_path
        self._save_experiment(experiment)
        
        logger.info(f"Added artifact '{artifact_name}' to experiment {experiment_id}")
    
    def add_experiment_tag(self, experiment_id: str, tag: str):
        """Add a tag to an experiment."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        if tag not in experiment.tags:
            experiment.tags.append(tag)
            self._save_experiment(experiment)
            
        logger.info(f"Added tag '{tag}' to experiment {experiment_id}")
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment by ID."""
        return self.experiments_cache.get(experiment_id)
    
    def get_experiments_by_name(self, experiment_name: str) -> List[ExperimentResult]:
        """Get all experiments with a specific name."""
        return [exp for exp in self.experiments_cache.values() 
                if exp.experiment_name == experiment_name]
    
    def get_experiments_by_type(self, experiment_type: ExperimentType) -> List[ExperimentResult]:
        """Get all experiments of a specific type."""
        return [exp for exp in self.experiments_cache.values() 
                if exp.configuration.experiment_type == experiment_type]
    
    def get_experiments_by_status(self, status: ExperimentStatus) -> List[ExperimentResult]:
        """Get all experiments with a specific status."""
        return [exp for exp in self.experiments_cache.values() if exp.status == status]
    
    def get_experiments_by_tag(self, tag: str) -> List[ExperimentResult]:
        """Get all experiments with a specific tag."""
        return [exp for exp in self.experiments_cache.values() if tag in exp.tags]
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict:
        """Compare multiple experiments."""
        experiments = [self.get_experiment(exp_id) for exp_id in experiment_ids]
        experiments = [exp for exp in experiments if exp is not None]
        
        if not experiments:
            return {}
        
        comparison = {
            'experiment_count': len(experiments),
            'experiments': {},
            'metric_comparison': {},
            'configuration_differences': {}
        }
        
        # Individual experiment summaries
        for exp in experiments:
            comparison['experiments'][exp.experiment_id] = {
                'name': exp.experiment_name,
                'type': exp.configuration.experiment_type.value,
                'status': exp.status.value,
                'duration': exp.duration,
                'metrics': exp.metrics.to_dict()
            }
        
        # Metric comparison
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'processing_time', 
                       'throughput', 'success_rate', 'confidence_score']
        
        for metric_name in metric_names:
            values = []
            for exp in experiments:
                value = getattr(exp.metrics, metric_name, None)
                if value is not None:
                    values.append((exp.experiment_id, value))
            
            if values:
                comparison['metric_comparison'][metric_name] = {
                    'values': dict(values),
                    'best': max(values, key=lambda x: x[1]) if values else None,
                    'worst': min(values, key=lambda x: x[1]) if values else None,
                    'average': sum(v[1] for v in values) / len(values)
                }
        
        return comparison
    
    def get_experiment_statistics(self) -> Dict:
        """Get overall experiment statistics."""
        experiments = list(self.experiments_cache.values())
        
        stats = {
            'total_experiments': len(experiments),
            'by_status': {},
            'by_type': {},
            'success_metrics': {},
            'duration_stats': {}
        }
        
        # Status breakdown
        for status in ExperimentStatus:
            count = sum(1 for exp in experiments if exp.status == status)
            stats['by_status'][status.value] = count
        
        # Type breakdown
        for exp_type in ExperimentType:
            count = sum(1 for exp in experiments if exp.configuration.experiment_type == exp_type)
            stats['by_type'][exp_type.value] = count
        
        # Success metrics
        completed_experiments = [exp for exp in experiments if exp.status == ExperimentStatus.COMPLETED]
        if completed_experiments:
            success_rates = [exp.metrics.success_rate for exp in completed_experiments 
                           if exp.metrics.success_rate is not None]
            if success_rates:
                stats['success_metrics'] = {
                    'average_success_rate': sum(success_rates) / len(success_rates),
                    'best_success_rate': max(success_rates),
                    'worst_success_rate': min(success_rates)
                }
        
        # Duration statistics
        durations = [exp.duration for exp in completed_experiments if exp.duration is not None]
        if durations:
            stats['duration_stats'] = {
                'average_duration': sum(durations) / len(durations),
                'shortest_duration': min(durations),
                'longest_duration': max(durations)
            }
        
        return stats
    
    def export_experiment(self, experiment_id: str, output_path: Union[str, Path]) -> bool:
        """Export experiment data to file."""
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False
        
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Exported experiment {experiment_id} to {output_path}")
        return True
    
    def _load_existing_experiments(self):
        """Load existing experiments from storage."""
        try:
            experiments_dir = self.storage_path / "experiments"
            if experiments_dir.exists():
                for exp_file in experiments_dir.glob("*.json"):
                    with open(exp_file, 'r') as f:
                        data = json.load(f)
                        experiment = ExperimentResult.from_dict(data)
                        self.experiments_cache[experiment.experiment_id] = experiment
            
            logger.info(f"Loaded {len(self.experiments_cache)} experiments")
            
        except Exception as e:
            logger.warning(f"Failed to load existing experiments: {e}")
    
    def _save_experiment(self, experiment: ExperimentResult):
        """Save experiment to storage."""
        try:
            experiments_dir = self.storage_path / "experiments"
            experiments_dir.mkdir(exist_ok=True)
            
            exp_file = experiments_dir / f"{experiment.experiment_id}.json"
            with open(exp_file, 'w') as f:
                json.dump(experiment.to_dict(), f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save experiment {experiment.experiment_id}: {e}")