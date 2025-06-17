"""Custom model training pipeline for PII extraction."""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, f1_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str = "bert-base-multilingual-cased"
    output_dir: str = "models/custom_pii"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    warmup_steps: int = 500
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    greater_is_better: bool = True
    patience: int = 3
    fp16: bool = True
    dataloader_num_workers: int = 4
    seed: int = 42
    
    def to_training_args(self) -> TrainingArguments:
        """Convert to HuggingFace TrainingArguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            logging_steps=self.logging_steps,
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            fp16=self.fp16,
            dataloader_num_workers=self.dataloader_num_workers,
            seed=self.seed,
            report_to=["mlflow"],
            run_name=f"pii_training_{self.model_name.split('/')[-1]}"
        )


class PIIDatasetProcessor:
    """Process datasets for PII extraction training."""
    
    def __init__(self, tokenizer_name: str = "bert-base-multilingual-cased"):
        """Initialize dataset processor."""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Standard PII label mapping
        self.pii_labels = [
            'O',  # Outside
            'B-PERSON', 'I-PERSON',
            'B-EMAIL', 'I-EMAIL', 
            'B-PHONE', 'I-PHONE',
            'B-ADDRESS', 'I-ADDRESS',
            'B-DATE', 'I-DATE',
            'B-SSN', 'I-SSN',
            'B-CREDIT_CARD', 'I-CREDIT_CARD',
            'B-ORG', 'I-ORG',
            'B-LOCATION', 'I-LOCATION',
            'B-MISC_PII', 'I-MISC_PII'
        ]
        
        self.label2id = {label: i for i, label in enumerate(self.pii_labels)}
        self.id2label = {i: label for i, label in enumerate(self.pii_labels)}
    
    def prepare_dataset(self, data_path: str, test_size: float = 0.2) -> DatasetDict:
        """Prepare dataset from annotations."""
        # Load annotated data
        training_data = self._load_training_data(data_path)
        
        # Split into train/validation
        train_data, eval_data = self._train_test_split(training_data, test_size)
        
        # Tokenize and align labels
        train_dataset = self._tokenize_and_align(train_data)
        eval_dataset = self._tokenize_and_align(eval_data)
        
        return DatasetDict({
            'train': Dataset.from_dict(train_dataset),
            'validation': Dataset.from_dict(eval_dataset)
        })
    
    def _load_training_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load training data from various formats."""
        data_path = Path(data_path)
        training_data = []
        
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                training_data.extend(data)
        elif data_path.suffix == '.jsonl':
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    training_data.append(json.loads(line))
        elif data_path.is_dir():
            # Load all JSON files in directory
            for json_file in data_path.glob('*.json'):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        training_data.extend(data)
                    else:
                        training_data.append(data)
        
        logger.info(f"Loaded {len(training_data)} training examples")
        return training_data
    
    def _train_test_split(self, data: List[Dict], test_size: float) -> Tuple[List[Dict], List[Dict]]:
        """Split data into train and test sets."""
        np.random.seed(42)
        indices = np.random.permutation(len(data))
        split_idx = int(len(data) * (1 - test_size))
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_data = [data[i] for i in train_indices]
        test_data = [data[i] for i in test_indices]
        
        return train_data, test_data
    
    def _tokenize_and_align(self, data: List[Dict]) -> Dict[str, List]:
        """Tokenize text and align labels."""
        tokenized_inputs = []
        labels = []
        
        for example in data:
            text = example['text']
            entities = example.get('entities', [])
            
            # Tokenize
            tokenized = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_offsets_mapping=True,
                return_tensors=None
            )
            
            # Align labels
            aligned_labels = self._align_labels_with_tokens(
                tokenized, entities, text
            )
            
            tokenized_inputs.append({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': aligned_labels
            })
        
        # Convert to format expected by Dataset
        result = {
            'input_ids': [],
            'attention_mask': [],
            'labels': []
        }
        
        for item in tokenized_inputs:
            result['input_ids'].append(item['input_ids'])
            result['attention_mask'].append(item['attention_mask'])
            result['labels'].append(item['labels'])
        
        return result
    
    def _align_labels_with_tokens(self, tokenized, entities: List[Dict], text: str) -> List[int]:
        """Align entity labels with tokens."""
        labels = [self.label2id['O']] * len(tokenized['input_ids'])
        offset_mapping = tokenized['offset_mapping']
        
        for entity in entities:
            start_pos = entity['start']
            end_pos = entity['end']
            pii_type = entity['type'].upper()
            
            # Find tokens that overlap with entity
            token_start_idx = None
            token_end_idx = None
            
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start_idx is None and token_start <= start_pos < token_end:
                    token_start_idx = i
                if token_start < end_pos <= token_end:
                    token_end_idx = i
                    break
            
            if token_start_idx is not None:
                if token_end_idx is None:
                    token_end_idx = token_start_idx
                
                # Assign B- and I- labels
                b_label = f'B-{pii_type}'
                i_label = f'I-{pii_type}'
                
                if b_label in self.label2id:
                    labels[token_start_idx] = self.label2id[b_label]
                    
                    if i_label in self.label2id:
                        for i in range(token_start_idx + 1, token_end_idx + 1):
                            if i < len(labels):
                                labels[i] = self.label2id[i_label]
        
        return labels


class PIIModelTrainer:
    """Custom trainer for PII extraction models."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer."""
        self.config = config
        self.tokenizer = None
        self.model = None
        self.dataset_processor = None
        self.trainer = None
    
    def setup_model(self, num_labels: int, label2id: Dict[str, int], id2label: Dict[int, str]):
        """Setup model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label
        )
        
        logger.info(f"Model setup complete: {self.config.model_name}")
    
    def train(self, dataset: DatasetDict, experiment_name: str = "pii_extraction") -> Dict[str, Any]:
        """Train the model with MLflow tracking."""
        
        # Start MLflow experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log configuration
            mlflow.log_params(asdict(self.config))
            
            # Setup data collator
            data_collator = DataCollatorForTokenClassification(
                tokenizer=self.tokenizer,
                padding=True
            )
            
            # Setup training arguments
            training_args = self.config.to_training_args()
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self._compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.patience)]
            )
            
            # Train model
            train_result = self.trainer.train()
            
            # Evaluate model
            eval_result = self.trainer.evaluate()
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "eval_loss": eval_result["eval_loss"],
                "eval_f1": eval_result["eval_f1"],
                "eval_precision": eval_result["eval_precision"],
                "eval_recall": eval_result["eval_recall"]
            })
            
            # Save model
            model_path = Path(self.config.output_dir)
            model_path.mkdir(parents=True, exist_ok=True)
            
            self.trainer.save_model(str(model_path))
            self.tokenizer.save_pretrained(str(model_path))
            
            # Log model to MLflow
            mlflow.pytorch.log_model(
                self.model,
                "model",
                registered_model_name=f"pii_extractor_{self.config.model_name.split('/')[-1]}"
            )
            
            # Save training configuration
            with open(model_path / "training_config.json", 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            logger.info(f"Training complete. Model saved to {model_path}")
            
            return {
                "train_loss": train_result.training_loss,
                "eval_metrics": eval_result,
                "model_path": str(model_path)
            }
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [self.dataset_processor.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.dataset_processor.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Flatten for sklearn metrics
        flat_true_labels = [label for sublist in true_labels for label in sublist]
        flat_predictions = [pred for sublist in true_predictions for pred in sublist]
        
        # Calculate metrics
        f1 = f1_score(flat_true_labels, flat_predictions, average='weighted')
        
        # Detailed classification report
        report = classification_report(
            flat_true_labels, 
            flat_predictions, 
            output_dict=True,
            zero_division=0
        )
        
        return {
            "f1": f1,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall']
        }


class ModelFineTuner:
    """Fine-tuning utilities for pre-trained models."""
    
    def __init__(self):
        """Initialize fine-tuner."""
        self.supported_models = [
            "bert-base-multilingual-cased",
            "xlm-roberta-base", 
            "distilbert-base-multilingual-cased",
            "microsoft/layoutlm-base-uncased"
        ]
    
    def fine_tune_for_pii(self, 
                         model_name: str,
                         training_data_path: str,
                         output_dir: str,
                         config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Fine-tune a model for PII extraction."""
        
        if model_name not in self.supported_models:
            logger.warning(f"Model {model_name} not in tested models: {self.supported_models}")
        
        # Setup configuration
        config = TrainingConfig(
            model_name=model_name,
            output_dir=output_dir
        )
        
        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Setup dataset processor
        dataset_processor = PIIDatasetProcessor(model_name)
        
        # Prepare dataset
        dataset = dataset_processor.prepare_dataset(training_data_path)
        
        # Setup trainer
        trainer = PIIModelTrainer(config)
        trainer.dataset_processor = dataset_processor
        trainer.setup_model(
            num_labels=len(dataset_processor.pii_labels),
            label2id=dataset_processor.label2id,
            id2label=dataset_processor.id2label
        )
        
        # Train model
        results = trainer.train(dataset, f"pii_finetune_{model_name.split('/')[-1]}")
        
        return results
    
    def evaluate_model(self, model_path: str, test_data_path: str) -> Dict[str, Any]:
        """Evaluate a fine-tuned model."""
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        # Load test data
        dataset_processor = PIIDatasetProcessor(model_path)
        test_dataset = dataset_processor.prepare_dataset(test_data_path, test_size=1.0)
        
        # Setup trainer for evaluation
        training_args = TrainingArguments(
            output_dir="./tmp",
            per_device_eval_batch_size=16
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            compute_metrics=lambda eval_pred: dataset_processor._compute_metrics(eval_pred)
        )
        
        # Evaluate
        eval_results = trainer.evaluate(test_dataset['validation'])
        
        return eval_results


def create_training_data_from_documents(documents_dir: str, output_path: str):
    """Helper function to create training data from annotated documents."""
    # This would typically involve manual annotation or semi-automated labeling
    # For now, create a template structure
    
    training_examples = []
    
    # Example structure for training data
    example = {
        "text": "John Doe works at ACME Corp. His email is john.doe@acme.com and phone is 555-123-4567.",
        "entities": [
            {"start": 0, "end": 8, "type": "PERSON", "text": "John Doe"},
            {"start": 18, "end": 27, "type": "ORG", "text": "ACME Corp"},
            {"start": 42, "end": 60, "type": "EMAIL", "text": "john.doe@acme.com"},
            {"start": 74, "end": 86, "type": "PHONE", "text": "555-123-4567"}
        ]
    }
    
    training_examples.append(example)
    
    # Save training data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Training data template saved to {output_path}")
    return output_path