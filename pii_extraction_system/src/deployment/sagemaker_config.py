"""SageMaker deployment configuration and scripts."""

import os
import json
import boto3
import joblib
import torch
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.sklearn import SKLearn, SKLearnModel
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat
import logging

logger = logging.getLogger(__name__)


@dataclass
class SageMakerConfig:
    """Configuration for SageMaker deployment."""
    # AWS Configuration
    region: str = "us-east-1"
    role: Optional[str] = None
    bucket: Optional[str] = None
    
    # Training Configuration
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    volume_size: int = 30
    max_run: int = 86400  # 24 hours
    
    # Inference Configuration
    inference_instance_type: str = "ml.m5.large"
    initial_instance_count: int = 1
    max_concurrent_transforms: int = 1
    
    # Model Configuration
    framework_version: str = "1.12"
    py_version: str = "py38"
    
    # Pipeline Configuration
    pipeline_name: str = "pii-extraction-pipeline"
    model_package_group_name: str = "pii-extraction-models"


class SageMakerDeploymentManager:
    """Manages SageMaker deployment operations."""
    
    def __init__(self, config: SageMakerConfig):
        """Initialize deployment manager."""
        self.config = config
        self.session = sagemaker.Session()
        
        # Set up AWS resources
        if not self.config.role:
            self.config.role = sagemaker.get_execution_role()
        
        if not self.config.bucket:
            self.config.bucket = self.session.default_bucket()
        
        self.s3_client = boto3.client('s3', region_name=self.config.region)
        
        logger.info(f"SageMaker deployment manager initialized")
        logger.info(f"Role: {self.config.role}")
        logger.info(f"Bucket: {self.config.bucket}")
    
    def create_training_job(self, 
                          source_dir: str,
                          entry_point: str,
                          training_data_s3: str,
                          hyperparameters: Dict[str, Any] = None) -> str:
        """Create SageMaker training job."""
        
        hyperparameters = hyperparameters or {
            'epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'model_name': 'bert-base-multilingual-cased'
        }
        
        # Create PyTorch estimator
        estimator = PyTorch(
            entry_point=entry_point,
            source_dir=source_dir,
            role=self.config.role,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            hyperparameters=hyperparameters,
            volume_size=self.config.volume_size,
            max_run=self.config.max_run,
            base_job_name='pii-extraction-training'
        )
        
        # Start training
        estimator.fit({'training': training_data_s3})
        
        logger.info(f"Training job started: {estimator.latest_training_job.name}")
        return estimator.latest_training_job.name
    
    def deploy_model(self, 
                    model_data_s3: str,
                    entry_point: str,
                    source_dir: str,
                    endpoint_name: str = None) -> str:
        """Deploy model to SageMaker endpoint."""
        
        endpoint_name = endpoint_name or "pii-extraction-endpoint"
        
        # Create PyTorch model
        model = PyTorchModel(
            model_data=model_data_s3,
            role=self.config.role,
            entry_point=entry_point,
            source_dir=source_dir,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version
        )
        
        # Deploy to endpoint
        predictor = model.deploy(
            initial_instance_count=self.config.initial_instance_count,
            instance_type=self.config.inference_instance_type,
            endpoint_name=endpoint_name
        )
        
        logger.info(f"Model deployed to endpoint: {endpoint_name}")
        return endpoint_name
    
    def create_batch_transform_job(self,
                                 model_name: str,
                                 input_data_s3: str,
                                 output_data_s3: str,
                                 job_name: str = None) -> str:
        """Create batch transform job for bulk processing."""
        
        job_name = job_name or f"pii-extraction-batch-{int(time.time())}"
        
        transformer = sagemaker.transformer.Transformer(
            model_name=model_name,
            instance_count=self.config.initial_instance_count,
            instance_type=self.config.inference_instance_type,
            output_path=output_data_s3,
            max_concurrent_transforms=self.config.max_concurrent_transforms
        )
        
        transformer.transform(
            data=input_data_s3,
            job_name=job_name,
            content_type='application/json',
            split_type='Line'
        )
        
        logger.info(f"Batch transform job started: {job_name}")
        return job_name
    
    def create_pipeline(self) -> Pipeline:
        """Create SageMaker ML pipeline for end-to-end workflow."""
        
        # Pipeline parameters
        input_data = ParameterString(name="InputData", default_value=f"s3://{self.config.bucket}/data/raw")
        model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
        
        # Processing step for data preparation
        processing_step = self._create_processing_step(input_data)
        
        # Training step
        training_step = self._create_training_step(processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri)
        
        # Model evaluation step
        evaluation_step = self._create_evaluation_step(training_step, processing_step)
        
        # Model registration step
        registration_step = self._create_registration_step(training_step, evaluation_step, model_approval_status)
        
        # Create pipeline
        pipeline = Pipeline(
            name=self.config.pipeline_name,
            parameters=[input_data, model_approval_status],
            steps=[processing_step, training_step, evaluation_step, registration_step]
        )
        
        return pipeline
    
    def _create_processing_step(self, input_data: ParameterString) -> ProcessingStep:
        """Create data processing step."""
        from sagemaker.processing import ScriptProcessor
        
        processor = ScriptProcessor(
            command=["python3"],
            image_uri=f"763104351884.dkr.ecr.{self.config.region}.amazonaws.com/pytorch-training:1.12.0-cpu-py38",
            role=self.config.role,
            instance_count=1,
            instance_type="ml.m5.large"
        )
        
        step = ProcessingStep(
            name="DataProcessing",
            processor=processor,
            inputs=[
                ProcessingInput(
                    source=input_data,
                    destination="/opt/ml/processing/input"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train",
                    source="/opt/ml/processing/train"
                ),
                ProcessingOutput(
                    output_name="validation",
                    source="/opt/ml/processing/validation"
                ),
                ProcessingOutput(
                    output_name="test",
                    source="/opt/ml/processing/test"
                )
            ],
            code="scripts/preprocessing.py"
        )
        
        return step
    
    def _create_training_step(self, training_data_uri: str) -> TrainingStep:
        """Create model training step."""
        estimator = PyTorch(
            entry_point="train.py",
            source_dir="scripts",
            role=self.config.role,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            hyperparameters={
                'epochs': 3,
                'batch_size': 16,
                'learning_rate': 2e-5
            }
        )
        
        step = TrainingStep(
            name="ModelTraining",
            estimator=estimator,
            inputs={
                'training': sagemaker.inputs.TrainingInput(
                    s3_data=training_data_uri,
                    content_type="application/json"
                )
            }
        )
        
        return step
    
    def _create_evaluation_step(self, training_step: TrainingStep, processing_step: ProcessingStep) -> ProcessingStep:
        """Create model evaluation step."""
        from sagemaker.processing import ScriptProcessor
        
        processor = ScriptProcessor(
            command=["python3"],
            image_uri=f"763104351884.dkr.ecr.{self.config.region}.amazonaws.com/pytorch-inference:1.12.0-cpu-py38",
            role=self.config.role,
            instance_count=1,
            instance_type="ml.m5.large"
        )
        
        step = ProcessingStep(
            name="ModelEvaluation",
            processor=processor,
            inputs=[
                ProcessingInput(
                    source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation"
                )
            ],
            code="scripts/evaluation.py"
        )
        
        return step
    
    def _create_registration_step(self, 
                                training_step: TrainingStep,
                                evaluation_step: ProcessingStep,
                                model_approval_status: ParameterString):
        """Create model registration step."""
        from sagemaker.workflow.steps import RegisterModel
        
        model = PyTorchModel(
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            role=self.config.role,
            entry_point="inference.py",
            source_dir="scripts",
            framework_version=self.config.framework_version,
            py_version=self.config.py_version
        )
        
        step = RegisterModel(
            name="RegisterModel",
            estimator=training_step.estimator,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large", "ml.m5.xlarge"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=self.config.model_package_group_name,
            approval_status=model_approval_status
        )
        
        return step


def create_training_script() -> str:
    """Create SageMaker training script."""
    script_content = '''
import argparse
import json
import logging
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import sagemaker

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """Load model for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    return {"model": model, "tokenizer": tokenizer}


def input_fn(request_body, content_type):
    """Parse input for inference."""
    if content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_dict):
    """Make prediction."""
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    text = input_data.get("text", "")
    
    # Tokenize input
    encoding = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**encoding)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_token_class_ids = predictions.argmax(dim=-1)
    
    # Convert predictions to labels
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    predicted_labels = [model.config.id2label[pred_id.item()] for pred_id in predicted_token_class_ids[0]]
    
    # Extract entities
    entities = []
    current_entity = None
    
    for token, label in zip(tokens, predicted_labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        if label.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "text": token,
                "label": label[2:],
                "confidence": predictions[0][len(entities)].max().item()
            }
        elif label.startswith("I-") and current_entity:
            current_entity["text"] += " " + token
    
    if current_entity:
        entities.append(current_entity)
    
    return {"entities": entities}


def output_fn(prediction, accept):
    """Format output."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


def train():
    """Training function."""
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ.get("SM_HOSTS")))
    parser.add_argument("--current-host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    
    # Model specific arguments
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
    
    args = parser.parse_args()
    
    # Load training data
    train_data_path = os.path.join(args.train, "train.json")
    with open(train_data_path, "r") as f:
        train_data = json.load(f)
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=17  # Adjust based on your label set
    )
    
    # Training code would go here...
    # For brevity, we'll just save the pre-trained model
    
    # Save model
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    
    logger.info("Training completed and model saved")


if __name__ == "__main__":
    train()
'''
    
    return script_content


def create_inference_script() -> str:
    """Create SageMaker inference script."""
    script_content = '''
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """Load model for inference."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForTokenClassification.from_pretrained(model_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def input_fn(request_body, content_type):
    """Parse input for inference."""
    if content_type == "application/json":
        try:
            input_data = json.loads(request_body)
            return input_data
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model_dict):
    """Make prediction."""
    try:
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        device = model_dict["device"]
        
        text = input_data.get("text", "")
        if not text:
            return {"entities": [], "error": "No text provided"}
        
        # Tokenize input
        encoding = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        encoding = {k: v.to(device) for k, v in encoding.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_token_class_ids = predictions.argmax(dim=-1)
        
        # Extract entities
        entities = extract_entities(
            text, encoding, predicted_token_class_ids, predictions, tokenizer, model
        )
        
        return {
            "entities": entities,
            "num_entities": len(entities),
            "processing_successful": True
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {
            "entities": [],
            "error": str(e),
            "processing_successful": False
        }


def extract_entities(text, encoding, predictions, probabilities, tokenizer, model):
    """Extract entities from model predictions."""
    entities = []
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    
    current_entity = None
    current_tokens = []
    
    for i, (token, pred_id, probs) in enumerate(zip(tokens, predictions[0], probabilities[0])):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
        
        label = model.config.id2label[pred_id.item()]
        confidence = probs[pred_id].item()
        
        if confidence < 0.5:  # Confidence threshold
            continue
        
        if label.startswith("B-"):
            # Start of new entity
            if current_entity:
                entities.append(finalize_entity(current_entity, current_tokens, text, tokenizer))
            
            current_entity = {
                "type": label[2:],
                "confidence": confidence,
                "start_token": i
            }
            current_tokens = [token]
            
        elif label.startswith("I-") and current_entity:
            # Continuation of entity
            current_tokens.append(token)
            current_entity["confidence"] = (current_entity["confidence"] + confidence) / 2
    
    # Finalize last entity
    if current_entity:
        entities.append(finalize_entity(current_entity, current_tokens, text, tokenizer))
    
    return entities


def finalize_entity(entity_info, tokens, text, tokenizer):
    """Convert token-level entity to final entity."""
    entity_text = tokenizer.convert_tokens_to_string(tokens)
    
    # Find position in original text
    start_pos = text.find(entity_text)
    end_pos = start_pos + len(entity_text) if start_pos >= 0 else 0
    
    return {
        "text": entity_text,
        "type": entity_info["type"],
        "confidence": round(entity_info["confidence"], 3),
        "start_pos": start_pos,
        "end_pos": end_pos
    }


def output_fn(prediction, accept):
    """Format output."""
    if accept == "application/json":
        return json.dumps(prediction), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''
    
    return script_content


def setup_sagemaker_scripts(output_dir: str):
    """Create all necessary SageMaker scripts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create training script
    train_script = create_training_script()
    with open(output_path / "train.py", "w") as f:
        f.write(train_script)
    
    # Create inference script
    inference_script = create_inference_script()
    with open(output_path / "inference.py", "w") as f:
        f.write(inference_script)
    
    # Create requirements file
    requirements = """
torch>=1.12.0
transformers>=4.30.0
sagemaker>=2.100.0
numpy>=1.21.0
scikit-learn>=1.0.0
"""
    
    with open(output_path / "requirements.txt", "w") as f:
        f.write(requirements)
    
    logger.info(f"SageMaker scripts created in {output_path}")
    
    return output_path