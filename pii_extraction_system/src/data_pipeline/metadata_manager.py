"""
Metadata Manager Module

Provides comprehensive metadata tagging, categorization, and management capabilities
for documents and processing results.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Set, Any
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
from .data_loader import DocumentMetadata

logger = get_logger(__name__)

class TagType(Enum):
    """Types of tags."""
    SYSTEM = "system"
    USER = "user"
    AUTO_GENERATED = "auto_generated"
    QUALITY = "quality"
    PROCESSING = "processing"
    CONTENT = "content"

class TagScope(Enum):
    """Scope of tag application."""
    DOCUMENT = "document"
    BATCH = "batch"
    EXPERIMENT = "experiment"
    DATASET = "dataset"

@dataclass
class DocumentTag:
    """A single tag applied to a document or entity."""
    
    tag_id: str
    tag_name: str
    tag_value: Optional[str] = None
    tag_type: TagType = TagType.USER
    tag_scope: TagScope = TagScope.DOCUMENT
    
    # Metadata
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    description: str = ""
    confidence: float = 1.0
    
    # Hierarchical tags
    parent_tag: Optional[str] = None
    child_tags: List[str] = field(default_factory=list)
    
    # Additional attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'tag_id': self.tag_id,
            'tag_name': self.tag_name,
            'tag_value': self.tag_value,
            'tag_type': self.tag_type.value,
            'tag_scope': self.tag_scope.value,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'description': self.description,
            'confidence': self.confidence,
            'parent_tag': self.parent_tag,
            'child_tags': self.child_tags,
            'attributes': self.attributes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentTag':
        """Create instance from dictionary."""
        return cls(
            tag_id=data['tag_id'],
            tag_name=data['tag_name'],
            tag_value=data.get('tag_value'),
            tag_type=TagType(data.get('tag_type', 'user')),
            tag_scope=TagScope(data.get('tag_scope', 'document')),
            created_by=data.get('created_by', 'system'),
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            description=data.get('description', ''),
            confidence=data.get('confidence', 1.0),
            parent_tag=data.get('parent_tag'),
            child_tags=data.get('child_tags', []),
            attributes=data.get('attributes', {})
        )

@dataclass
class MetadataSchema:
    """Schema definition for metadata fields."""
    
    schema_id: str
    schema_name: str
    version: str = "1.0"
    
    # Field definitions
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)
    field_constraints: Dict[str, Dict] = field(default_factory=dict)
    
    # Validation rules
    validation_rules: Dict[str, List[str]] = field(default_factory=dict)
    
    def validate_metadata(self, metadata: Dict) -> Tuple[bool, List[str]]:
        """Validate metadata against this schema."""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_type in self.field_types.items():
            if field in metadata:
                value = metadata[field]
                if not self._validate_type(value, expected_type):
                    errors.append(f"Invalid type for field {field}: expected {expected_type}")
        
        # Apply validation rules
        for field, rules in self.validation_rules.items():
            if field in metadata:
                value = metadata[field]
                for rule in rules:
                    if not self._apply_validation_rule(value, rule):
                        errors.append(f"Validation failed for field {field}: {rule}")
        
        return len(errors) == 0, errors
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type."""
        type_mapping = {
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'list': list,
            'dict': dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, assume valid
    
    def _apply_validation_rule(self, value: Any, rule: str) -> bool:
        """Apply a validation rule to a value."""
        # Simple validation rules (can be extended)
        if rule.startswith('min_length:'):
            min_len = int(rule.split(':')[1])
            return len(str(value)) >= min_len
        elif rule.startswith('max_length:'):
            max_len = int(rule.split(':')[1])
            return len(str(value)) <= max_len
        elif rule.startswith('pattern:'):
            import re
            pattern = rule.split(':', 1)[1]
            return bool(re.match(pattern, str(value)))
        
        return True  # Unknown rule, assume valid


class MetadataManager:
    """
    Comprehensive metadata management system for documents and processing results.
    """
    
    def __init__(self, storage_path: Union[str, Path] = None):
        """
        Initialize metadata manager.
        
        Args:
            storage_path: Path to store metadata
        """
        if storage_path is None:
            self.storage_path = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/memory/metadata")
        else:
            self.storage_path = Path(storage_path)
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory caches
        self.tags_cache: Dict[str, DocumentTag] = {}
        self.document_tags: Dict[str, List[str]] = {}  # document_id -> tag_ids
        self.tag_index: Dict[str, Set[str]] = {}  # tag_name -> document_ids
        self.schemas_cache: Dict[str, MetadataSchema] = {}
        
        # Load existing data
        self._load_existing_data()
        
        # Initialize default schemas
        self._initialize_default_schemas()
        
        logger.info(f"MetadataManager initialized with storage at: {self.storage_path}")
    
    def create_tag(self, 
                   tag_name: str,
                   tag_value: Optional[str] = None,
                   tag_type: TagType = TagType.USER,
                   tag_scope: TagScope = TagScope.DOCUMENT,
                   created_by: str = "user",
                   description: str = "",
                   attributes: Dict[str, Any] = None) -> DocumentTag:
        """Create a new tag."""
        tag_id = str(uuid.uuid4())
        
        tag = DocumentTag(
            tag_id=tag_id,
            tag_name=tag_name,
            tag_value=tag_value,
            tag_type=tag_type,
            tag_scope=tag_scope,
            created_by=created_by,
            description=description,
            attributes=attributes or {}
        )
        
        # Cache and save
        self.tags_cache[tag_id] = tag
        self._save_tag(tag)
        
        logger.info(f"Created tag {tag_id}: {tag_name}")
        return tag
    
    def apply_tag_to_document(self, document_id: str, tag_id: str) -> bool:
        """Apply a tag to a document."""
        tag = self.get_tag(tag_id)
        if not tag:
            logger.error(f"Tag not found: {tag_id}")
            return False
        
        # Add to document tags
        if document_id not in self.document_tags:
            self.document_tags[document_id] = []
        
        if tag_id not in self.document_tags[document_id]:
            self.document_tags[document_id].append(tag_id)
        
        # Update tag index
        if tag.tag_name not in self.tag_index:
            self.tag_index[tag.tag_name] = set()
        self.tag_index[tag.tag_name].add(document_id)
        
        # Save changes
        self._save_document_tags()
        self._save_tag_index()
        
        logger.debug(f"Applied tag {tag.tag_name} to document {document_id}")
        return True
    
    def apply_tag_by_name(self, 
                         document_id: str, 
                         tag_name: str,
                         tag_value: Optional[str] = None,
                         create_if_not_exists: bool = True) -> bool:
        """Apply a tag by name to a document, creating it if necessary."""
        # Find existing tag
        existing_tag = self.find_tag_by_name(tag_name, tag_value)
        
        if existing_tag:
            return self.apply_tag_to_document(document_id, existing_tag.tag_id)
        elif create_if_not_exists:
            # Create new tag
            new_tag = self.create_tag(tag_name=tag_name, tag_value=tag_value)
            return self.apply_tag_to_document(document_id, new_tag.tag_id)
        
        return False
    
    def remove_tag_from_document(self, document_id: str, tag_id: str) -> bool:
        """Remove a tag from a document."""
        if document_id in self.document_tags:
            if tag_id in self.document_tags[document_id]:
                self.document_tags[document_id].remove(tag_id)
                
                # Update tag index
                tag = self.get_tag(tag_id)
                if tag and tag.tag_name in self.tag_index:
                    self.tag_index[tag.tag_name].discard(document_id)
                
                self._save_document_tags()
                self._save_tag_index()
                
                logger.debug(f"Removed tag {tag_id} from document {document_id}")
                return True
        
        return False
    
    def get_document_tags(self, document_id: str) -> List[DocumentTag]:
        """Get all tags applied to a document."""
        tag_ids = self.document_tags.get(document_id, [])
        return [self.tags_cache[tag_id] for tag_id in tag_ids if tag_id in self.tags_cache]
    
    def get_documents_by_tag(self, tag_name: str, tag_value: Optional[str] = None) -> Set[str]:
        """Get all documents with a specific tag."""
        if tag_value is None:
            return self.tag_index.get(tag_name, set()).copy()
        
        # Filter by tag value
        documents = set()
        for document_id in self.tag_index.get(tag_name, set()):
            doc_tags = self.get_document_tags(document_id)
            for tag in doc_tags:
                if tag.tag_name == tag_name and tag.tag_value == tag_value:
                    documents.add(document_id)
                    break
        
        return documents
    
    def get_tag(self, tag_id: str) -> Optional[DocumentTag]:
        """Get a tag by ID."""
        return self.tags_cache.get(tag_id)
    
    def find_tag_by_name(self, tag_name: str, tag_value: Optional[str] = None) -> Optional[DocumentTag]:
        """Find a tag by name and optional value."""
        for tag in self.tags_cache.values():
            if tag.tag_name == tag_name and tag.tag_value == tag_value:
                return tag
        return None
    
    def get_all_tag_names(self) -> List[str]:
        """Get all unique tag names."""
        return list(self.tag_index.keys())
    
    def get_tag_statistics(self) -> Dict:
        """Get comprehensive tag statistics."""
        stats = {
            'total_tags': len(self.tags_cache),
            'total_unique_names': len(self.tag_index),
            'by_type': {},
            'by_scope': {},
            'usage_statistics': {}
        }
        
        # Statistics by type and scope
        for tag_type in TagType:
            count = sum(1 for tag in self.tags_cache.values() if tag.tag_type == tag_type)
            stats['by_type'][tag_type.value] = count
        
        for tag_scope in TagScope:
            count = sum(1 for tag in self.tags_cache.values() if tag.tag_scope == tag_scope)
            stats['by_scope'][tag_scope.value] = count
        
        # Usage statistics
        for tag_name, document_ids in self.tag_index.items():
            stats['usage_statistics'][tag_name] = len(document_ids)
        
        return stats
    
    def auto_tag_document(self, document_metadata: DocumentMetadata) -> List[DocumentTag]:
        """Automatically generate tags for a document based on its metadata."""
        auto_tags = []
        
        # File format tag
        format_tag = self.create_tag(
            tag_name="format",
            tag_value=document_metadata.file_extension,
            tag_type=TagType.AUTO_GENERATED,
            description=f"File format: {document_metadata.file_extension}"
        )
        auto_tags.append(format_tag)
        self.apply_tag_to_document(document_metadata.document_id, format_tag.tag_id)
        
        # Content type tag
        content_tag = self.create_tag(
            tag_name="content_type",
            tag_value=document_metadata.content_type,
            tag_type=TagType.AUTO_GENERATED,
            description=f"Content type: {document_metadata.content_type}"
        )
        auto_tags.append(content_tag)
        self.apply_tag_to_document(document_metadata.document_id, content_tag.tag_id)
        
        # Size category tag
        size_mb = document_metadata.file_size_bytes / (1024 * 1024)
        if size_mb < 1:
            size_category = "small"
        elif size_mb < 10:
            size_category = "medium"
        else:
            size_category = "large"
        
        size_tag = self.create_tag(
            tag_name="size_category",
            tag_value=size_category,
            tag_type=TagType.AUTO_GENERATED,
            description=f"File size category: {size_category}"
        )
        auto_tags.append(size_tag)
        self.apply_tag_to_document(document_metadata.document_id, size_tag.tag_id)
        
        # Processing status tag
        status_tag = self.create_tag(
            tag_name="processing_status",
            tag_value=document_metadata.processing_status,
            tag_type=TagType.PROCESSING,
            description=f"Processing status: {document_metadata.processing_status}"
        )
        auto_tags.append(status_tag)
        self.apply_tag_to_document(document_metadata.document_id, status_tag.tag_id)
        
        # Ground truth tag
        if document_metadata.is_ground_truth:
            gt_tag = self.create_tag(
                tag_name="ground_truth",
                tag_value=document_metadata.ground_truth_category,
                tag_type=TagType.SYSTEM,
                description="Document is part of ground truth dataset"
            )
            auto_tags.append(gt_tag)
            self.apply_tag_to_document(document_metadata.document_id, gt_tag.tag_id)
        
        logger.info(f"Auto-generated {len(auto_tags)} tags for document {document_metadata.file_name}")
        return auto_tags
    
    def create_custom_metadata_schema(self, 
                                    schema_name: str,
                                    required_fields: List[str],
                                    optional_fields: List[str] = None,
                                    field_types: Dict[str, str] = None) -> MetadataSchema:
        """Create a custom metadata schema."""
        schema_id = str(uuid.uuid4())
        
        schema = MetadataSchema(
            schema_id=schema_id,
            schema_name=schema_name,
            required_fields=required_fields,
            optional_fields=optional_fields or [],
            field_types=field_types or {}
        )
        
        self.schemas_cache[schema_id] = schema
        self._save_schema(schema)
        
        logger.info(f"Created metadata schema {schema_id}: {schema_name}")
        return schema
    
    def validate_document_metadata(self, 
                                 document_metadata: Dict,
                                 schema_id: str) -> Tuple[bool, List[str]]:
        """Validate document metadata against a schema."""
        schema = self.schemas_cache.get(schema_id)
        if not schema:
            return False, [f"Schema not found: {schema_id}"]
        
        return schema.validate_metadata(document_metadata)
    
    def export_metadata(self, output_path: Union[str, Path]) -> bool:
        """Export all metadata to file."""
        output_path = Path(output_path)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'tags': {tag_id: tag.to_dict() for tag_id, tag in self.tags_cache.items()},
            'document_tags': self.document_tags,
            'tag_index': {name: list(docs) for name, docs in self.tag_index.items()},
            'statistics': self.get_tag_statistics()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Exported metadata to {output_path}")
        return True
    
    def _initialize_default_schemas(self):
        """Initialize default metadata schemas."""
        # Document schema
        doc_schema = MetadataSchema(
            schema_id="default_document",
            schema_name="Default Document Schema",
            required_fields=["document_id", "file_path", "file_name"],
            optional_fields=["tags", "description", "category"],
            field_types={"document_id": "str", "file_path": "str", "file_name": "str"}
        )
        self.schemas_cache[doc_schema.schema_id] = doc_schema
    
    def _load_existing_data(self):
        """Load existing metadata from storage."""
        try:
            # Load tags
            tags_file = self.storage_path / "tags.json"
            if tags_file.exists():
                with open(tags_file, 'r') as f:
                    tags_data = json.load(f)
                    for tag_id, tag_data in tags_data.items():
                        tag = DocumentTag.from_dict(tag_data)
                        self.tags_cache[tag_id] = tag
            
            # Load document tags
            doc_tags_file = self.storage_path / "document_tags.json"
            if doc_tags_file.exists():
                with open(doc_tags_file, 'r') as f:
                    self.document_tags = json.load(f)
            
            # Load tag index
            tag_index_file = self.storage_path / "tag_index.json"
            if tag_index_file.exists():
                with open(tag_index_file, 'r') as f:
                    tag_index_data = json.load(f)
                    self.tag_index = {name: set(docs) for name, docs in tag_index_data.items()}
            
            logger.info(f"Loaded {len(self.tags_cache)} tags and {len(self.document_tags)} document associations")
            
        except Exception as e:
            logger.warning(f"Failed to load existing metadata: {e}")
    
    def _save_tag(self, tag: DocumentTag):
        """Save a tag to storage."""
        try:
            tags_file = self.storage_path / "tags.json"
            
            # Load existing tags
            tags_data = {}
            if tags_file.exists():
                with open(tags_file, 'r') as f:
                    tags_data = json.load(f)
            
            # Add/update tag
            tags_data[tag.tag_id] = tag.to_dict()
            
            # Save back
            with open(tags_file, 'w') as f:
                json.dump(tags_data, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save tag {tag.tag_id}: {e}")
    
    def _save_document_tags(self):
        """Save document tags mapping."""
        try:
            doc_tags_file = self.storage_path / "document_tags.json"
            with open(doc_tags_file, 'w') as f:
                json.dump(self.document_tags, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save document tags: {e}")
    
    def _save_tag_index(self):
        """Save tag index."""
        try:
            tag_index_file = self.storage_path / "tag_index.json"
            tag_index_data = {name: list(docs) for name, docs in self.tag_index.items()}
            with open(tag_index_file, 'w') as f:
                json.dump(tag_index_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tag index: {e}")
    
    def _save_schema(self, schema: MetadataSchema):
        """Save a metadata schema."""
        try:
            schemas_dir = self.storage_path / "schemas"
            schemas_dir.mkdir(exist_ok=True)
            
            schema_file = schemas_dir / f"{schema.schema_id}.json"
            with open(schema_file, 'w') as f:
                json.dump(schema.__dict__, f, indent=2, default=str)
        
        except Exception as e:
            logger.error(f"Failed to save schema {schema.schema_id}: {e}")