"""
Ground Truth Management System

Manages training and validation datasets, ground truth annotations,
and dataset versioning for machine learning experiments.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Set, Tuple
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
from .data_loader import DataLoader, DocumentMetadata

logger = get_logger(__name__)

class DatasetType(Enum):
    """Types of datasets."""
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"
    UNLABELED = "unlabeled"

class AnnotationStatus(Enum):
    """Status of annotations."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"

@dataclass
class GroundTruthAnnotation:
    """Single ground truth annotation."""
    
    annotation_id: str
    document_id: str
    annotator_id: str
    
    # PII annotations
    pii_entities: List[Dict] = field(default_factory=list)
    
    # Document-level labels
    document_labels: List[str] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    
    # Annotation metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: AnnotationStatus = AnnotationStatus.PENDING
    
    # Quality control
    review_notes: str = ""
    quality_score: Optional[float] = None
    is_consensus: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'annotation_id': self.annotation_id,
            'document_id': self.document_id,
            'annotator_id': self.annotator_id,
            'pii_entities': self.pii_entities,
            'document_labels': self.document_labels,
            'confidence_scores': self.confidence_scores,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.value,
            'review_notes': self.review_notes,
            'quality_score': self.quality_score,
            'is_consensus': self.is_consensus
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GroundTruthAnnotation':
        """Create instance from dictionary."""
        return cls(
            annotation_id=data['annotation_id'],
            document_id=data['document_id'],
            annotator_id=data['annotator_id'],
            pii_entities=data.get('pii_entities', []),
            document_labels=data.get('document_labels', []),
            confidence_scores=data.get('confidence_scores', []),
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00')),
            status=AnnotationStatus(data.get('status', 'pending')),
            review_notes=data.get('review_notes', ''),
            quality_score=data.get('quality_score'),
            is_consensus=data.get('is_consensus', False)
        )

@dataclass
class GroundTruthEntry:
    """Complete ground truth entry for a document."""
    
    entry_id: str
    document_metadata: DocumentMetadata
    dataset_type: DatasetType
    
    # Annotations from multiple annotators
    annotations: List[GroundTruthAnnotation] = field(default_factory=list)
    
    # Consensus/final annotation
    consensus_annotation: Optional[GroundTruthAnnotation] = None
    
    # Dataset management
    dataset_version: str = "1.0"
    split_assignment: str = ""  # e.g., "train_fold_1", "validation_set_A"
    
    # Quality metrics
    inter_annotator_agreement: Optional[float] = None
    annotation_difficulty: Optional[float] = None
    
    # Custom attributes
    custom_attributes: Dict = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def add_annotation(self, annotation: GroundTruthAnnotation):
        """Add an annotation to this entry."""
        self.annotations.append(annotation)
        self._update_consensus()
    
    def _update_consensus(self):
        """Update consensus annotation based on multiple annotations."""
        if not self.annotations:
            self.consensus_annotation = None
            return
        
        # Simple consensus: use the most recent reviewed annotation
        reviewed_annotations = [a for a in self.annotations if a.status == AnnotationStatus.REVIEWED]
        
        if reviewed_annotations:
            # Use the most recent reviewed annotation
            self.consensus_annotation = max(reviewed_annotations, key=lambda a: a.updated_at)
        elif self.annotations:
            # Use the most recent completed annotation
            completed_annotations = [a for a in self.annotations if a.status == AnnotationStatus.COMPLETED]
            if completed_annotations:
                self.consensus_annotation = max(completed_annotations, key=lambda a: a.updated_at)
    
    def calculate_agreement(self) -> float:
        """Calculate inter-annotator agreement."""
        if len(self.annotations) < 2:
            return 1.0
        
        # Simple agreement calculation based on PII entity overlap
        # This is a simplified version - real implementation would be more sophisticated
        all_entities = []
        for annotation in self.annotations:
            entities = set()
            for entity in annotation.pii_entities:
                entities.add((entity.get('text', ''), entity.get('type', '')))
            all_entities.append(entities)
        
        if not all_entities:
            return 1.0
        
        # Calculate pairwise agreement
        agreements = []
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                intersection = len(all_entities[i] & all_entities[j])
                union = len(all_entities[i] | all_entities[j])
                agreement = intersection / union if union > 0 else 1.0
                agreements.append(agreement)
        
        self.inter_annotator_agreement = sum(agreements) / len(agreements) if agreements else 1.0
        return self.inter_annotator_agreement
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'entry_id': self.entry_id,
            'document_metadata': self.document_metadata.to_dict(),
            'dataset_type': self.dataset_type.value,
            'annotations': [a.to_dict() for a in self.annotations],
            'consensus_annotation': self.consensus_annotation.to_dict() if self.consensus_annotation else None,
            'dataset_version': self.dataset_version,
            'split_assignment': self.split_assignment,
            'inter_annotator_agreement': self.inter_annotator_agreement,
            'annotation_difficulty': self.annotation_difficulty,
            'custom_attributes': self.custom_attributes,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GroundTruthEntry':
        """Create instance from dictionary."""
        entry = cls(
            entry_id=data['entry_id'],
            document_metadata=DocumentMetadata.from_dict(data['document_metadata']),
            dataset_type=DatasetType(data['dataset_type']),
            dataset_version=data.get('dataset_version', '1.0'),
            split_assignment=data.get('split_assignment', ''),
            inter_annotator_agreement=data.get('inter_annotator_agreement'),
            annotation_difficulty=data.get('annotation_difficulty'),
            custom_attributes=data.get('custom_attributes', {}),
            tags=data.get('tags', [])
        )
        
        # Load annotations
        annotations_data = data.get('annotations', [])
        entry.annotations = [GroundTruthAnnotation.from_dict(a) for a in annotations_data]
        
        # Load consensus annotation
        consensus_data = data.get('consensus_annotation')
        if consensus_data:
            entry.consensus_annotation = GroundTruthAnnotation.from_dict(consensus_data)
        
        return entry


class GroundTruthManager:
    """
    Manages ground truth datasets, annotations, and versioning.
    """
    
    def __init__(self, 
                 data_loader: DataLoader = None,
                 storage_path: Union[str, Path] = None):
        """
        Initialize ground truth manager.
        
        Args:
            data_loader: DataLoader instance
            storage_path: Path to store ground truth data
        """
        self.data_loader = data_loader or DataLoader()
        
        if storage_path is None:
            self.storage_path = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/memory/ground_truth")
        else:
            self.storage_path = Path(storage_path)
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.entries_cache: Dict[str, GroundTruthEntry] = {}
        self.annotations_cache: Dict[str, GroundTruthAnnotation] = {}
        
        # Load existing data
        self._load_existing_data()
        
        logger.info(f"GroundTruthManager initialized with storage at: {self.storage_path}")
    
    def create_ground_truth_entry(self, 
                                 document_metadata: DocumentMetadata,
                                 dataset_type: DatasetType,
                                 dataset_version: str = "1.0") -> GroundTruthEntry:
        """Create a new ground truth entry."""
        entry_id = str(uuid.uuid4())
        
        entry = GroundTruthEntry(
            entry_id=entry_id,
            document_metadata=document_metadata,
            dataset_type=dataset_type,
            dataset_version=dataset_version
        )
        
        # Mark document as ground truth in data loader
        self.data_loader.mark_as_ground_truth(
            document_metadata.document_id, 
            category=dataset_type.value
        )
        
        # Cache and save
        self.entries_cache[entry_id] = entry
        self._save_entry(entry)
        
        logger.info(f"Created ground truth entry {entry_id} for document {document_metadata.file_name}")
        return entry
    
    def add_annotation(self, 
                      entry_id: str,
                      annotator_id: str,
                      pii_entities: List[Dict],
                      document_labels: List[str] = None,
                      confidence_scores: List[float] = None) -> GroundTruthAnnotation:
        """Add an annotation to a ground truth entry."""
        entry = self.get_entry(entry_id)
        if not entry:
            raise ValueError(f"Ground truth entry not found: {entry_id}")
        
        annotation_id = str(uuid.uuid4())
        annotation = GroundTruthAnnotation(
            annotation_id=annotation_id,
            document_id=entry.document_metadata.document_id,
            annotator_id=annotator_id,
            pii_entities=pii_entities or [],
            document_labels=document_labels or [],
            confidence_scores=confidence_scores or [],
            status=AnnotationStatus.COMPLETED
        )
        
        entry.add_annotation(annotation)
        entry.calculate_agreement()
        
        # Cache and save
        self.annotations_cache[annotation_id] = annotation
        self._save_entry(entry)
        self._save_annotation(annotation)
        
        logger.info(f"Added annotation {annotation_id} to entry {entry_id}")
        return annotation
    
    def update_annotation_status(self, 
                               annotation_id: str,
                               status: AnnotationStatus,
                               review_notes: str = "",
                               quality_score: float = None):
        """Update annotation status and review information."""
        annotation = self.get_annotation(annotation_id)
        if not annotation:
            raise ValueError(f"Annotation not found: {annotation_id}")
        
        annotation.status = status
        annotation.updated_at = datetime.now()
        annotation.review_notes = review_notes
        
        if quality_score is not None:
            annotation.quality_score = quality_score
        
        # Update the associated entry's consensus
        entry = self.get_entry_by_document_id(annotation.document_id)
        if entry:
            entry._update_consensus()
            self._save_entry(entry)
        
        self._save_annotation(annotation)
        
        logger.info(f"Updated annotation {annotation_id} status to {status.value}")
    
    def create_dataset_split(self, 
                           dataset_version: str,
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15,
                           stratify_by: Optional[str] = None) -> Dict[str, List[str]]:
        """Create train/validation/test splits."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Split ratios must sum to 1.0")
        
        # Get all entries for the dataset version
        entries = [e for e in self.entries_cache.values() if e.dataset_version == dataset_version]
        
        if not entries:
            raise ValueError(f"No entries found for dataset version {dataset_version}")
        
        # Simple random split (could be enhanced with stratification)
        import random
        random.shuffle(entries)
        
        total = len(entries)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_entries = entries[:train_size]
        val_entries = entries[train_size:train_size + val_size]
        test_entries = entries[train_size + val_size:]
        
        # Assign split labels
        for i, entry in enumerate(train_entries):
            entry.split_assignment = f"train_fold_1"
            entry.dataset_type = DatasetType.TRAINING
            self._save_entry(entry)
        
        for i, entry in enumerate(val_entries):
            entry.split_assignment = f"validation_set_A"
            entry.dataset_type = DatasetType.VALIDATION
            self._save_entry(entry)
        
        for i, entry in enumerate(test_entries):
            entry.split_assignment = f"test_set"
            entry.dataset_type = DatasetType.TEST
            self._save_entry(entry)
        
        split_info = {
            'train': [e.entry_id for e in train_entries],
            'validation': [e.entry_id for e in val_entries],
            'test': [e.entry_id for e in test_entries]
        }
        
        # Save split information
        split_file = self.storage_path / f"splits_{dataset_version}.json"
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Created dataset split for version {dataset_version}: "
                   f"train={len(train_entries)}, val={len(val_entries)}, test={len(test_entries)}")
        
        return split_info
    
    def get_entry(self, entry_id: str) -> Optional[GroundTruthEntry]:
        """Get ground truth entry by ID."""
        return self.entries_cache.get(entry_id)
    
    def get_entry_by_document_id(self, document_id: str) -> Optional[GroundTruthEntry]:
        """Get ground truth entry by document ID."""
        for entry in self.entries_cache.values():
            if entry.document_metadata.document_id == document_id:
                return entry
        return None
    
    def get_annotation(self, annotation_id: str) -> Optional[GroundTruthAnnotation]:
        """Get annotation by ID."""
        return self.annotations_cache.get(annotation_id)
    
    def get_entries_by_type(self, dataset_type: DatasetType) -> List[GroundTruthEntry]:
        """Get all entries of a specific dataset type."""
        return [e for e in self.entries_cache.values() if e.dataset_type == dataset_type]
    
    def get_entries_by_version(self, dataset_version: str) -> List[GroundTruthEntry]:
        """Get all entries for a specific dataset version."""
        return [e for e in self.entries_cache.values() if e.dataset_version == dataset_version]
    
    def get_dataset_statistics(self, dataset_version: str = None) -> Dict:
        """Get comprehensive dataset statistics."""
        entries = self.entries_cache.values()
        if dataset_version:
            entries = [e for e in entries if e.dataset_version == dataset_version]
        
        stats = {
            'total_entries': len(list(entries)),
            'by_type': {},
            'by_status': {},
            'annotation_quality': {},
            'agreement_metrics': {}
        }
        
        # Statistics by dataset type
        for dtype in DatasetType:
            type_entries = [e for e in entries if e.dataset_type == dtype]
            stats['by_type'][dtype.value] = len(type_entries)
        
        # Annotation statistics
        all_annotations = []
        for entry in entries:
            all_annotations.extend(entry.annotations)
        
        for status in AnnotationStatus:
            status_annotations = [a for a in all_annotations if a.status == status]
            stats['by_status'][status.value] = len(status_annotations)
        
        # Quality metrics
        quality_scores = [a.quality_score for a in all_annotations if a.quality_score is not None]
        if quality_scores:
            stats['annotation_quality'] = {
                'average_quality': sum(quality_scores) / len(quality_scores),
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores)
            }
        
        # Agreement metrics
        agreements = [e.inter_annotator_agreement for e in entries if e.inter_annotator_agreement is not None]
        if agreements:
            stats['agreement_metrics'] = {
                'average_agreement': sum(agreements) / len(agreements),
                'min_agreement': min(agreements),
                'max_agreement': max(agreements)
            }
        
        return stats
    
    def export_dataset(self, 
                      dataset_version: str,
                      output_path: Union[str, Path],
                      export_format: str = "json") -> bool:
        """Export dataset to file."""
        output_path = Path(output_path)
        entries = self.get_entries_by_version(dataset_version)
        
        if not entries:
            logger.warning(f"No entries found for dataset version {dataset_version}")
            return False
        
        export_data = {
            'dataset_version': dataset_version,
            'export_timestamp': datetime.now().isoformat(),
            'total_entries': len(entries),
            'entries': [entry.to_dict() for entry in entries]
        }
        
        if export_format.lower() == "json":
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Exported {len(entries)} entries to {output_path}")
        return True
    
    def _load_existing_data(self):
        """Load existing ground truth data from storage."""
        try:
            # Load entries
            entries_dir = self.storage_path / "entries"
            if entries_dir.exists():
                for entry_file in entries_dir.glob("*.json"):
                    with open(entry_file, 'r') as f:
                        data = json.load(f)
                        entry = GroundTruthEntry.from_dict(data)
                        self.entries_cache[entry.entry_id] = entry
            
            # Load annotations
            annotations_dir = self.storage_path / "annotations"
            if annotations_dir.exists():
                for annotation_file in annotations_dir.glob("*.json"):
                    with open(annotation_file, 'r') as f:
                        data = json.load(f)
                        annotation = GroundTruthAnnotation.from_dict(data)
                        self.annotations_cache[annotation.annotation_id] = annotation
            
            logger.info(f"Loaded {len(self.entries_cache)} entries and {len(self.annotations_cache)} annotations")
            
        except Exception as e:
            logger.warning(f"Failed to load existing ground truth data: {e}")
    
    def _save_entry(self, entry: GroundTruthEntry):
        """Save ground truth entry to storage."""
        try:
            entries_dir = self.storage_path / "entries"
            entries_dir.mkdir(exist_ok=True)
            
            entry_file = entries_dir / f"{entry.entry_id}.json"
            with open(entry_file, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save entry {entry.entry_id}: {e}")
    
    def _save_annotation(self, annotation: GroundTruthAnnotation):
        """Save annotation to storage."""
        try:
            annotations_dir = self.storage_path / "annotations"
            annotations_dir.mkdir(exist_ok=True)
            
            annotation_file = annotations_dir / f"{annotation.annotation_id}.json"
            with open(annotation_file, 'w') as f:
                json.dump(annotation.to_dict(), f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save annotation {annotation.annotation_id}: {e}")