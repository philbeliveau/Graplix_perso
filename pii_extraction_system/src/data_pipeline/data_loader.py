"""
Data Loader Module

Handles loading documents from various sources with comprehensive metadata extraction.
Supports batch loading, filtering, and integration with existing document processing pipeline.
"""

import uuid
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

# Import existing components
import sys
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.config import settings
from core.logging_config import get_logger
from utils.document_processor import DocumentProcessor

logger = get_logger(__name__)

@dataclass
class DocumentMetadata:
    """Comprehensive document metadata structure."""
    
    # Basic file information
    document_id: str
    file_path: str
    file_name: str
    file_extension: str
    file_size_bytes: int
    file_hash: str
    
    # Timestamps
    created_at: datetime
    modified_at: datetime
    accessed_at: datetime
    loaded_at: datetime = field(default_factory=datetime.now)
    
    # Processing information
    processing_status: str = "pending"  # pending, processing, completed, failed
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    # Content metadata
    content_type: str = "unknown"
    estimated_pages: int = 0
    estimated_words: int = 0
    language: str = "unknown"
    
    # Custom tags and attributes
    tags: List[str] = field(default_factory=list)
    custom_attributes: Dict = field(default_factory=dict)
    
    # Source information
    source_directory: str = ""
    is_ground_truth: bool = False
    ground_truth_category: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'document_id': self.document_id,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_extension': self.file_extension,
            'file_size_bytes': self.file_size_bytes,
            'file_hash': self.file_hash,
            'created_at': self.created_at.isoformat(),
            'modified_at': self.modified_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'loaded_at': self.loaded_at.isoformat(),
            'processing_status': self.processing_status,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'content_type': self.content_type,
            'estimated_pages': self.estimated_pages,
            'estimated_words': self.estimated_words,
            'language': self.language,
            'tags': self.tags,
            'custom_attributes': self.custom_attributes,
            'source_directory': self.source_directory,
            'is_ground_truth': self.is_ground_truth,
            'ground_truth_category': self.ground_truth_category
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentMetadata':
        """Create instance from dictionary."""
        return cls(
            document_id=data['document_id'],
            file_path=data['file_path'],
            file_name=data['file_name'],
            file_extension=data['file_extension'],
            file_size_bytes=data['file_size_bytes'],
            file_hash=data['file_hash'],
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            modified_at=datetime.fromisoformat(data['modified_at'].replace('Z', '+00:00')),
            accessed_at=datetime.fromisoformat(data['accessed_at'].replace('Z', '+00:00')),
            loaded_at=datetime.fromisoformat(data['loaded_at'].replace('Z', '+00:00')),
            processing_status=data.get('processing_status', 'pending'),
            processing_time=data.get('processing_time'),
            error_message=data.get('error_message'),
            content_type=data.get('content_type', 'unknown'),
            estimated_pages=data.get('estimated_pages', 0),
            estimated_words=data.get('estimated_words', 0),
            language=data.get('language', 'unknown'),
            tags=data.get('tags', []),
            custom_attributes=data.get('custom_attributes', {}),
            source_directory=data.get('source_directory', ''),
            is_ground_truth=data.get('is_ground_truth', False),
            ground_truth_category=data.get('ground_truth_category')
        )


class DataLoader:
    """
    Main data loading class that handles document discovery, metadata extraction,
    and integration with existing processing pipeline.
    """
    
    def __init__(self, 
                 data_directory: Union[str, Path] = None,
                 cache_metadata: bool = True,
                 supported_formats: List[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_directory: Path to data directory (defaults to project data dir)
            cache_metadata: Whether to cache document metadata
            supported_formats: List of supported file extensions
        """
        # Set default data directory
        if data_directory is None:
            self.data_directory = Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data")
        else:
            self.data_directory = Path(data_directory)
        
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")
        
        # Initialize document processor for format support
        self.document_processor = DocumentProcessor()
        
        # Set supported formats
        if supported_formats is None:
            self.supported_formats = self.document_processor.get_supported_formats()
        else:
            self.supported_formats = supported_formats
        
        # Metadata caching
        self.cache_metadata = cache_metadata
        self.metadata_cache: Dict[str, DocumentMetadata] = {}
        self.cache_file = self.data_directory.parent / "memory" / "data_loader_cache.json"
        
        # Load cached metadata if available
        if self.cache_metadata:
            self._load_metadata_cache()
        
        logger.info(f"DataLoader initialized with directory: {self.data_directory}")
        logger.info(f"Supported formats: {self.supported_formats}")
    
    def discover_documents(self, 
                          recursive: bool = True,
                          file_filter: Optional[Callable[[Path], bool]] = None) -> List[Path]:
        """
        Discover all supported documents in the data directory.
        
        Args:
            recursive: Search subdirectories recursively
            file_filter: Optional filter function for files
            
        Returns:
            List of document file paths
        """
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in self.data_directory.glob(pattern):
            if (file_path.is_file() and 
                file_path.suffix.lower() in self.supported_formats):
                
                # Apply custom filter if provided
                if file_filter is None or file_filter(file_path):
                    documents.append(file_path)
        
        logger.info(f"Discovered {len(documents)} documents")
        return sorted(documents)
    
    def load_document_metadata(self, file_path: Union[str, Path]) -> DocumentMetadata:
        """
        Load comprehensive metadata for a single document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentMetadata object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Check cache first
        file_hash = self._calculate_file_hash(file_path)
        cache_key = f"{file_path}:{file_hash}"
        
        if self.cache_metadata and cache_key in self.metadata_cache:
            cached_metadata = self.metadata_cache[cache_key]
            cached_metadata.accessed_at = datetime.now()
            return cached_metadata
        
        # Get file stats
        stat = file_path.stat()
        
        # Create document ID
        document_id = str(uuid.uuid4())
        
        # Create metadata object
        metadata = DocumentMetadata(
            document_id=document_id,
            file_path=str(file_path),
            file_name=file_path.name,
            file_extension=file_path.suffix.lower(),
            file_size_bytes=stat.st_size,
            file_hash=file_hash,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            accessed_at=datetime.fromtimestamp(stat.st_atime),
            source_directory=str(file_path.parent),
            content_type=self._determine_content_type(file_path)
        )
        
        # Add additional metadata based on file type
        self._enrich_metadata(metadata, file_path)
        
        # Cache metadata
        if self.cache_metadata:
            self.metadata_cache[cache_key] = metadata
            self._save_metadata_cache()
        
        return metadata
    
    def load_documents_metadata(self, 
                               file_paths: List[Union[str, Path]],
                               parallel: bool = True) -> List[DocumentMetadata]:
        """
        Load metadata for multiple documents.
        
        Args:
            file_paths: List of document file paths
            parallel: Whether to process in parallel (placeholder for future implementation)
            
        Returns:
            List of DocumentMetadata objects
        """
        metadata_list = []
        
        for file_path in file_paths:
            try:
                metadata = self.load_document_metadata(file_path)
                metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Failed to load metadata for {file_path}: {e}")
                # Create error metadata
                error_metadata = self._create_error_metadata(file_path, str(e))
                metadata_list.append(error_metadata)
        
        return metadata_list
    
    def get_documents_by_format(self, file_extension: str) -> List[DocumentMetadata]:
        """Get all documents of a specific format."""
        documents = self.discover_documents()
        filtered_docs = [d for d in documents if d.suffix.lower() == file_extension.lower()]
        return self.load_documents_metadata(filtered_docs)
    
    def get_documents_by_tag(self, tag: str) -> List[DocumentMetadata]:
        """Get all documents with a specific tag."""
        all_documents = self.discover_documents()
        all_metadata = self.load_documents_metadata(all_documents)
        return [m for m in all_metadata if tag in m.tags]
    
    def get_ground_truth_documents(self, category: Optional[str] = None) -> List[DocumentMetadata]:
        """Get all ground truth documents, optionally filtered by category."""
        all_documents = self.discover_documents()
        all_metadata = self.load_documents_metadata(all_documents)
        
        filtered = [m for m in all_metadata if m.is_ground_truth]
        
        if category:
            filtered = [m for m in filtered if m.ground_truth_category == category]
        
        return filtered
    
    def add_document_tag(self, document_id: str, tag: str) -> bool:
        """Add a tag to a document."""
        for metadata in self.metadata_cache.values():
            if metadata.document_id == document_id:
                if tag not in metadata.tags:
                    metadata.tags.append(tag)
                    self._save_metadata_cache()
                return True
        return False
    
    def mark_as_ground_truth(self, document_id: str, category: str = None) -> bool:
        """Mark a document as ground truth."""
        for metadata in self.metadata_cache.values():
            if metadata.document_id == document_id:
                metadata.is_ground_truth = True
                metadata.ground_truth_category = category
                self._save_metadata_cache()
                return True
        return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about the loaded documents."""
        all_documents = self.discover_documents()
        all_metadata = self.load_documents_metadata(all_documents)
        
        stats = {
            'total_documents': len(all_metadata),
            'by_format': {},
            'total_size_bytes': 0,
            'ground_truth_count': 0,
            'processing_status': {'pending': 0, 'processing': 0, 'completed': 0, 'failed': 0}
        }
        
        for metadata in all_metadata:
            # Format statistics
            ext = metadata.file_extension
            stats['by_format'][ext] = stats['by_format'].get(ext, 0) + 1
            
            # Size statistics
            stats['total_size_bytes'] += metadata.file_size_bytes
            
            # Ground truth statistics
            if metadata.is_ground_truth:
                stats['ground_truth_count'] += 1
            
            # Processing status
            status = metadata.processing_status
            stats['processing_status'][status] = stats['processing_status'].get(status, 0) + 1
        
        stats['total_size_mb'] = round(stats['total_size_bytes'] / (1024 * 1024), 2)
        
        return stats
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _determine_content_type(self, file_path: Path) -> str:
        """Determine content type based on file extension and analysis."""
        ext = file_path.suffix.lower()
        
        content_types = {
            '.pdf': 'document',
            '.docx': 'document',
            '.doc': 'document',
            '.xlsx': 'spreadsheet',
            '.xls': 'spreadsheet',
            '.txt': 'text',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.png': 'image',
            '.tiff': 'image',
            '.bmp': 'image'
        }
        
        return content_types.get(ext, 'unknown')
    
    def _enrich_metadata(self, metadata: DocumentMetadata, file_path: Path):
        """Enrich metadata with additional file-specific information."""
        try:
            # For text files, estimate word count
            if metadata.file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    metadata.estimated_words = len(content.split())
                    metadata.estimated_pages = max(1, len(content) // 2000)  # Rough estimate
            
            # For PDFs, try to get page count from existing processing
            elif metadata.file_extension == '.pdf':
                # This would integrate with existing PDF processing
                metadata.estimated_pages = 1  # Placeholder
            
            # Check if this is a ground truth document based on naming or location
            if 'ground_truth' in file_path.name.lower() or 'gt_' in file_path.name.lower():
                metadata.is_ground_truth = True
                metadata.tags.append('ground_truth')
            
            # Add format-specific tags
            metadata.tags.append(f"format:{metadata.file_extension}")
            metadata.tags.append(f"type:{metadata.content_type}")
            
        except Exception as e:
            logger.warning(f"Failed to enrich metadata for {file_path}: {e}")
    
    def _create_error_metadata(self, file_path: Union[str, Path], error_message: str) -> DocumentMetadata:
        """Create metadata object for files that failed to load."""
        file_path = Path(file_path)
        
        return DocumentMetadata(
            document_id=str(uuid.uuid4()),
            file_path=str(file_path),
            file_name=file_path.name if file_path.exists() else "unknown",
            file_extension=file_path.suffix.lower() if file_path.exists() else "unknown",
            file_size_bytes=0,
            file_hash="",
            created_at=datetime.now(),
            modified_at=datetime.now(),
            accessed_at=datetime.now(),
            processing_status="failed",
            error_message=error_message
        )
    
    def _load_metadata_cache(self):
        """Load metadata cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                for key, data in cache_data.items():
                    self.metadata_cache[key] = DocumentMetadata.from_dict(data)
                
                logger.info(f"Loaded {len(self.metadata_cache)} cached metadata entries")
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")
            self.metadata_cache = {}
    
    def _save_metadata_cache(self):
        """Save metadata cache to disk."""
        try:
            # Ensure directory exists
            self.cache_file.parent.mkdir(exist_ok=True)
            
            # Convert cache to serializable format
            cache_data = {}
            for key, metadata in self.metadata_cache.items():
                cache_data[key] = metadata.to_dict()
            
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
        except Exception as e:
            logger.warning(f"Failed to save metadata cache: {e}")