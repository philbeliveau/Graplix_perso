"""Advanced PII Extraction System - Agent 3 Integration."""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import mlflow
from .extractors.layout_aware import LayoutLMExtractor, DonutExtractor, LayoutAwareEnsemble
from .models.training_pipeline import PIIModelTrainer, TrainingConfig, ModelFineTuner
from .models.ensemble import AdvancedEnsemble, EnsembleConfig
from .privacy.redaction import ComprehensivePrivacyProcessor, RedactionConfig, TokenizationConfig
# Optional SageMaker import
try:
    from .deployment.sagemaker_config import SageMakerDeploymentManager, SageMakerConfig
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SageMakerDeploymentManager = None
    SageMakerConfig = None
    SAGEMAKER_AVAILABLE = False
from .models.version_control import ModelRegistry, ABTestingFramework, ExperimentConfig
from .extractors.base import PIIExtractorBase, PIIExtractionResult
from .core import get_pipeline

logger = logging.getLogger(__name__)


class AdvancedPIISystem:
    """
    Advanced PII Extraction System integrating all Agent 3 deliverables:
    - Layout-aware models (LayoutLM, Donut)
    - Custom model fine-tuning capabilities
    - Ensemble method integration
    - Privacy-preserving techniques and redaction
    - SageMaker deployment configuration
    - Model versioning and A/B testing framework
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 enable_mlflow: bool = True,
                 enable_sagemaker: bool = False):
        """Initialize the advanced PII system."""
        
        self.config_path = config_path
        self.enable_mlflow = enable_mlflow
        self.enable_sagemaker = enable_sagemaker
        
        # Initialize core components
        self.model_registry = ModelRegistry("models/registry")
        self.ab_testing = ABTestingFramework(self.model_registry)
        
        # Initialize extractors
        self.extractors = self._initialize_extractors()
        
        # Initialize ensemble
        ensemble_config = EnsembleConfig(
            method="weighted_vote",
            confidence_threshold=0.7,
            similarity_threshold=0.8
        )
        self.ensemble = AdvancedEnsemble(ensemble_config, self.extractors)
        
        # Initialize privacy processor
        self.privacy_processor = ComprehensivePrivacyProcessor(
            redaction_config=RedactionConfig(),
            tokenization_config=TokenizationConfig(),
            enable_differential_privacy=True
        )
        
        # Initialize SageMaker manager if enabled
        self.sagemaker_manager = None
        if enable_sagemaker and SAGEMAKER_AVAILABLE:
            try:
                sagemaker_config = SageMakerConfig()
                self.sagemaker_manager = SageMakerDeploymentManager(sagemaker_config)
                logger.info("SageMaker deployment manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SageMaker manager: {e}")
        elif enable_sagemaker and not SAGEMAKER_AVAILABLE:
            logger.warning("SageMaker requested but not available (missing dependencies)")
        
        # Initialize MLflow if enabled
        if enable_mlflow:
            try:
                mlflow.set_tracking_uri("models/mlruns")
                mlflow.set_experiment("advanced_pii_system")
                logger.info("MLflow tracking initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")
        
        logger.info("Advanced PII System initialized successfully")
    
    def _initialize_extractors(self) -> List[PIIExtractorBase]:
        """Initialize all available extractors."""
        extractors = []
        
        try:
            # Layout-aware extractors
            extractors.append(LayoutLMExtractor())
            logger.info("LayoutLM extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize LayoutLM extractor: {e}")
        
        try:
            extractors.append(DonutExtractor())
            logger.info("Donut extractor initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Donut extractor: {e}")
        
        try:
            # Layout-aware ensemble
            extractors.append(LayoutAwareEnsemble())
            logger.info("Layout-aware ensemble initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize layout-aware ensemble: {e}")
        
        # Import rule-based and NER extractors from Agent 2
        try:
            from .extractors.rule_based import RuleBasedExtractor
            extractors.append(RuleBasedExtractor())
            logger.info("Rule-based extractor added")
        except Exception as e:
            logger.warning(f"Failed to import rule-based extractor: {e}")
        
        try:
            from .extractors.ner_extractor import NERExtractor
            extractors.append(NERExtractor())
            logger.info("NER extractor added")
        except Exception as e:
            logger.warning(f"Failed to import NER extractor: {e}")
        
        return extractors
    
    def extract_pii(self, 
                   document: Dict[str, Any],
                   method: str = "ensemble",
                   privacy_mode: str = "none",
                   experiment_id: str = None) -> Dict[str, Any]:
        """
        Extract PII from document using specified method.
        
        Args:
            document: Document to process
            method: Extraction method ('ensemble', 'layoutlm', 'donut', 'layout_ensemble')
            privacy_mode: Privacy processing ('none', 'redaction', 'tokenization', 'mixed')
            experiment_id: A/B testing experiment ID
            
        Returns:
            Results with extracted PII and processing metadata
        """
        
        # Route request for A/B testing if experiment specified
        if experiment_id and experiment_id in self.ab_testing.experiments:
            variant = self.ab_testing.route_request(experiment_id)
            method = variant  # Use variant as method
        
        # Select extractor based on method
        if method == "ensemble":
            extractor = self.ensemble
        elif method == "layoutlm":
            extractor = next((e for e in self.extractors if isinstance(e, LayoutLMExtractor)), None)
        elif method == "donut":
            extractor = next((e for e in self.extractors if isinstance(e, DonutExtractor)), None)
        elif method == "layout_ensemble":
            extractor = next((e for e in self.extractors if isinstance(e, LayoutAwareEnsemble)), None)
        else:
            # Default to ensemble
            extractor = self.ensemble
        
        if not extractor:
            raise ValueError(f"Extractor for method '{method}' not available")
        
        # Extract PII
        with mlflow.start_run(nested=True) if self.enable_mlflow else nullcontext():
            extraction_result = extractor.extract_pii(document)
            
            if self.enable_mlflow:
                # Log metrics to MLflow
                mlflow.log_metric("num_entities", len(extraction_result.pii_entities))
                mlflow.log_metric("processing_time", extraction_result.processing_time)
                if extraction_result.error:
                    mlflow.log_metric("has_error", 1)
                else:
                    mlflow.log_metric("has_error", 0)
        
        # Apply privacy processing if requested
        processed_document = document
        if privacy_mode != "none" and extraction_result.pii_entities:
            processed_document = self.privacy_processor.process_document(
                document, extraction_result, privacy_mode
            )
        
        # Prepare results
        results = {
            "extraction_result": extraction_result.to_dict(),
            "processed_document": processed_document,
            "method_used": method,
            "privacy_mode": privacy_mode,
            "metadata": {
                "extractor_name": extractor.name,
                "num_entities": len(extraction_result.pii_entities),
                "processing_time": extraction_result.processing_time,
                "entity_types": list(set(e.pii_type for e in extraction_result.pii_entities))
            }
        }
        
        return results
    
    def train_custom_model(self, 
                          training_data_path: str,
                          model_name: str = "bert-base-multilingual-cased",
                          training_config: Optional[TrainingConfig] = None) -> str:
        """Train a custom PII extraction model."""
        
        if not training_config:
            training_config = TrainingConfig(
                model_name=model_name,
                num_train_epochs=3,
                per_device_train_batch_size=16,
                learning_rate=2e-5
            )
        
        # Use model fine-tuner
        fine_tuner = ModelFineTuner()
        
        # Start MLflow run for training
        with mlflow.start_run(run_name=f"custom_training_{model_name}") if self.enable_mlflow else nullcontext():
            results = fine_tuner.fine_tune_for_pii(
                model_name=model_name,
                training_data_path=training_data_path,
                output_dir=f"models/custom/{model_name}",
                config_overrides=training_config.__dict__
            )
            
            # Register model in registry
            model_version = self.model_registry.register_model(
                model_name=f"custom_{model_name}",
                model_artifact=results["model_path"],  # Path to model
                framework="pytorch",
                description=f"Fine-tuned {model_name} for PII extraction",
                metrics=results["eval_metrics"],
                hyperparameters=training_config.__dict__
            )
            
            logger.info(f"Custom model trained and registered: {model_version.version_id}")
            return model_version.version_id
    
    def deploy_to_sagemaker(self, 
                           model_version_id: str,
                           endpoint_name: str = None) -> str:
        """Deploy model to SageMaker endpoint."""
        
        if not self.sagemaker_manager:
            raise ValueError("SageMaker manager not initialized")
        
        # Get model version
        model_version = None
        for versions in self.model_registry.models.values():
            for version in versions:
                if version.version_id == model_version_id:
                    model_version = version
                    break
        
        if not model_version:
            raise ValueError(f"Model version {model_version_id} not found")
        
        # Deploy to SageMaker
        endpoint_name = self.sagemaker_manager.deploy_model(
            model_data_s3=model_version.model_path,
            entry_point="inference.py",
            source_dir="scripts",
            endpoint_name=endpoint_name
        )
        
        logger.info(f"Model deployed to SageMaker endpoint: {endpoint_name}")
        return endpoint_name
    
    def create_ab_experiment(self, 
                           experiment_name: str,
                           model_versions: Dict[str, str],
                           traffic_split: Dict[str, float] = None) -> str:
        """Create A/B testing experiment."""
        
        traffic_split = traffic_split or {"A": 0.5, "B": 0.5}
        
        # Create experiment configuration
        config = ExperimentConfig(
            experiment_name=experiment_name,
            description=f"A/B test comparing {', '.join(model_versions.keys())}",
            start_date=datetime.now(),
            traffic_split=traffic_split,
            success_metrics=["f1_score", "precision", "recall"]
        )
        
        # Create and start experiment
        experiment_id = self.ab_testing.create_experiment(config)
        self.ab_testing.start_experiment(experiment_id, model_versions)
        
        logger.info(f"A/B experiment created and started: {experiment_id}")
        return experiment_id
    
    def evaluate_system(self, 
                       test_documents: List[Dict[str, Any]],
                       ground_truth: List[List[Any]],
                       methods: List[str] = None) -> Dict[str, Any]:
        """Evaluate system performance on test data."""
        
        methods = methods or ["ensemble", "layoutlm", "donut"]
        evaluation_results = {}
        
        for method in methods:
            method_results = {
                "predictions": [],
                "processing_times": [],
                "errors": 0
            }
            
            for doc in test_documents:
                try:
                    result = self.extract_pii(doc, method=method)
                    method_results["predictions"].append(result["extraction_result"]["pii_entities"])
                    method_results["processing_times"].append(result["metadata"]["processing_time"])
                except Exception as e:
                    logger.error(f"Error processing document with {method}: {e}")
                    method_results["errors"] += 1
                    method_results["predictions"].append([])
                    method_results["processing_times"].append(0)
            
            # Calculate metrics if we have ground truth
            if ground_truth:
                try:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    
                    # Convert to label format for evaluation
                    pred_labels = []
                    true_labels = []
                    
                    for preds, truth in zip(method_results["predictions"], ground_truth):
                        pred_types = [entity["type"] for entity in preds]
                        true_types = [entity.pii_type if hasattr(entity, 'pii_type') else entity["type"] for entity in truth]
                        
                        pred_labels.extend(pred_types)
                        true_labels.extend(true_types)
                    
                    if pred_labels and true_labels:
                        method_results["precision"] = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
                        method_results["recall"] = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
                        method_results["f1_score"] = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
                
                except Exception as e:
                    logger.warning(f"Could not calculate metrics for {method}: {e}")
            
            # Calculate average processing time
            if method_results["processing_times"]:
                method_results["avg_processing_time"] = sum(method_results["processing_times"]) / len(method_results["processing_times"])
            
            evaluation_results[method] = method_results
        
        return evaluation_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        status = {
            "extractors": {
                "available": [e.name for e in self.extractors],
                "count": len(self.extractors)
            },
            "model_registry": {
                "total_models": len(self.model_registry.models),
                "total_versions": sum(len(versions) for versions in self.model_registry.models.values())
            },
            "ab_testing": self.ab_testing.get_experiment_summary(),
            "privacy_processor": {
                "redaction_enabled": True,
                "tokenization_enabled": True,
                "differential_privacy_enabled": True
            },
            "sagemaker": {
                "enabled": self.sagemaker_manager is not None,
                "manager_available": self.sagemaker_manager is not None
            },
            "mlflow": {
                "enabled": self.enable_mlflow
            }
        }
        
        return status
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize system performance based on historical data."""
        
        optimization_results = {
            "recommendations": [],
            "performance_analysis": {}
        }
        
        # Analyze extractor performance
        if len(self.extractors) > 1:
            # This would require historical performance data
            # For now, provide general recommendations
            optimization_results["recommendations"].append(
                "Use ensemble method for best accuracy"
            )
            optimization_results["recommendations"].append(
                "Use LayoutLM for document-heavy workloads"
            )
            optimization_results["recommendations"].append(
                "Use rule-based extractor for high-speed processing"
            )
        
        # Memory optimization recommendations
        optimization_results["recommendations"].append(
            "Consider model quantization for deployment"
        )
        optimization_results["recommendations"].append(
            "Use batch processing for large document sets"
        )
        
        return optimization_results


# Context manager for MLflow (Python 3.7+ compatibility)
class nullcontext:
    """Null context manager for Python < 3.7 compatibility."""
    def __enter__(self):
        return None
    def __exit__(self, *excinfo):
        return False


def create_demo_system() -> AdvancedPIISystem:
    """Create a demo instance of the advanced PII system."""
    
    system = AdvancedPIISystem(
        enable_mlflow=True,
        enable_sagemaker=False  # Disabled for demo
    )
    
    logger.info("Demo Advanced PII System created")
    return system


def main():
    """Main function for testing the advanced PII system."""
    
    # Create demo system
    system = create_demo_system()
    
    # Test document
    test_document = {
        "text": "John Doe works at ACME Corp. His email is john.doe@acme.com and phone is 555-123-4567.",
        "metadata": {"source": "demo", "language": "en"}
    }
    
    # Test PII extraction
    print("Testing PII extraction...")
    results = system.extract_pii(test_document, method="ensemble")
    
    print(f"Found {results['metadata']['num_entities']} PII entities:")
    for entity in results['extraction_result']['pii_entities']:
        print(f"  - {entity['text']} ({entity['type']}) - Confidence: {entity['confidence']:.3f}")
    
    # Test privacy processing
    print("\nTesting privacy processing...")
    privacy_results = system.extract_pii(test_document, privacy_mode="redaction")
    print(f"Redacted text: {privacy_results['processed_document']['text']}")
    
    # Get system status
    print("\nSystem Status:")
    status = system.get_system_status()
    print(f"Available extractors: {status['extractors']['available']}")
    print(f"Model registry: {status['model_registry']['total_models']} models, {status['model_registry']['total_versions']} versions")
    
    print("\nAdvanced PII System demo completed successfully!")


if __name__ == "__main__":
    main()