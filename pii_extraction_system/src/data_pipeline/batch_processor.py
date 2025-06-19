"""
Batch Processor Module

Handles efficient batch processing of multiple documents with parallel execution,
progress monitoring, and comprehensive result aggregation.
"""

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
import json
import threading
from queue import Queue

# Import existing components
import sys
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from core.logging_config import get_logger
from .data_loader import DataLoader, DocumentMetadata
from .format_handlers import FormatHandlerRegistry, ProcessingResult

logger = get_logger(__name__)

@dataclass
class BatchStatus:
    """Real-time batch processing status."""
    
    batch_id: str
    total_documents: int
    processed_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    
    start_time: datetime = field(default_factory=datetime.now)
    current_time: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    
    current_file: str = ""
    processing_speed: float = 0.0  # docs per second
    
    def update_progress(self, document_processed: bool = True, success: bool = True, current_file: str = ""):
        """Update processing progress."""
        if document_processed:
            self.processed_documents += 1
            if success:
                self.successful_documents += 1
            else:
                self.failed_documents += 1
        
        self.current_time = datetime.now()
        self.current_file = current_file
        
        # Calculate processing speed
        elapsed_seconds = (self.current_time - self.start_time).total_seconds()
        if elapsed_seconds > 0:
            self.processing_speed = self.processed_documents / elapsed_seconds
        
        # Estimate completion time
        if self.processing_speed > 0:
            remaining_docs = self.total_documents - self.processed_documents
            remaining_seconds = remaining_docs / self.processing_speed
            self.estimated_completion = self.current_time + datetime.timedelta(seconds=remaining_seconds)
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.processed_documents == 0:
            return 0.0
        return (self.successful_documents / self.processed_documents) * 100
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'batch_id': self.batch_id,
            'total_documents': self.total_documents,
            'processed_documents': self.processed_documents,
            'successful_documents': self.successful_documents,
            'failed_documents': self.failed_documents,
            'start_time': self.start_time.isoformat(),
            'current_time': self.current_time.isoformat(),
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'current_file': self.current_file,
            'processing_speed': self.processing_speed,
            'progress_percentage': self.get_progress_percentage(),
            'success_rate': self.get_success_rate()
        }

@dataclass
class BatchResult:
    """Comprehensive batch processing result."""
    
    batch_id: str
    status: BatchStatus
    results: List[ProcessingResult] = field(default_factory=list)
    
    # Aggregated metrics
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    total_documents_size: int = 0
    
    # Quality metrics
    overall_confidence: float = 0.0
    format_breakdown: Dict[str, int] = field(default_factory=dict)
    error_summary: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    throughput_docs_per_second: float = 0.0
    throughput_mb_per_second: float = 0.0
    
    def finalize(self):
        """Calculate final metrics after processing is complete."""
        if not self.results:
            return
        
        # Basic metrics
        self.total_processing_time = sum(r.processing_time for r in self.results)
        self.average_processing_time = self.total_processing_time / len(self.results)
        
        # Confidence metrics
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            confidences = [r.confidence_score for r in successful_results if r.confidence_score > 0]
            self.overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Format breakdown
        for result in self.results:
            file_path = Path(result.file_path)
            ext = file_path.suffix.lower()
            self.format_breakdown[ext] = self.format_breakdown.get(ext, 0) + 1
        
        # Error summary
        failed_results = [r for r in self.results if not r.success]
        for result in failed_results:
            error_type = self._categorize_error(result.error_message or "unknown")
            self.error_summary[error_type] = self.error_summary.get(error_type, 0) + 1
        
        # Performance metrics
        elapsed_time = (self.status.current_time - self.status.start_time).total_seconds()
        if elapsed_time > 0:
            self.throughput_docs_per_second = len(self.results) / elapsed_time
            
            # Calculate total MB processed
            total_mb = sum(
                Path(r.file_path).stat().st_size / (1024 * 1024) 
                for r in self.results 
                if Path(r.file_path).exists()
            )
            self.throughput_mb_per_second = total_mb / elapsed_time
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error messages into types."""
        error_lower = error_message.lower()
        
        if 'not found' in error_lower or 'file not found' in error_lower:
            return 'file_not_found'
        elif 'permission' in error_lower or 'access' in error_lower:
            return 'permission_denied'
        elif 'password' in error_lower or 'encrypted' in error_lower:
            return 'encryption_error'
        elif 'timeout' in error_lower:
            return 'timeout_error'
        elif 'memory' in error_lower:
            return 'memory_error'
        elif 'format' in error_lower or 'unsupported' in error_lower:
            return 'format_error'
        else:
            return 'processing_error'
    
    def get_summary(self) -> Dict:
        """Get a summary of batch processing results."""
        return {
            'batch_id': self.batch_id,
            'total_documents': len(self.results),
            'successful_documents': sum(1 for r in self.results if r.success),
            'failed_documents': sum(1 for r in self.results if not r.success),
            'total_processing_time': self.total_processing_time,
            'average_processing_time': self.average_processing_time,
            'overall_confidence': self.overall_confidence,
            'throughput_docs_per_second': self.throughput_docs_per_second,
            'throughput_mb_per_second': self.throughput_mb_per_second,
            'format_breakdown': self.format_breakdown,
            'error_summary': self.error_summary,
            'completion_time': self.status.current_time.isoformat()
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'batch_id': self.batch_id,
            'status': self.status.to_dict(),
            'results': [r.to_dict() for r in self.results],
            'summary': self.get_summary()
        }


class BatchProcessor:
    """
    Advanced batch processor for handling multiple documents efficiently.
    Supports parallel processing, progress monitoring, and result aggregation.
    """
    
    def __init__(self, 
                 data_loader: DataLoader = None,
                 format_registry: FormatHandlerRegistry = None,
                 max_workers: int = 4,
                 progress_callback: Optional[Callable[[BatchStatus], None]] = None):
        """
        Initialize batch processor.
        
        Args:
            data_loader: DataLoader instance
            format_registry: Format handler registry
            max_workers: Maximum number of parallel workers
            progress_callback: Optional callback for progress updates
        """
        self.data_loader = data_loader or DataLoader()
        self.format_registry = format_registry or FormatHandlerRegistry()
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        
        # Thread-safe status tracking
        self._status_lock = threading.Lock()
        self._active_batches: Dict[str, BatchStatus] = {}
        
        logger.info(f"BatchProcessor initialized with {max_workers} workers")
    
    def process_directory(self, 
                         directory_path: Union[str, Path] = None,
                         recursive: bool = True,
                         file_filter: Optional[Callable[[Path], bool]] = None,
                         batch_size: Optional[int] = None,
                         **kwargs) -> BatchResult:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Directory to process (defaults to data loader directory)
            recursive: Search recursively
            file_filter: Optional file filter function
            batch_size: Optional batch size for processing
            **kwargs: Additional arguments passed to processors
            
        Returns:
            BatchResult with processing results
        """
        # Discover documents
        if directory_path:
            # Temporarily change data loader directory
            original_dir = self.data_loader.data_directory
            self.data_loader.data_directory = Path(directory_path)
        
        try:
            documents = self.data_loader.discover_documents(recursive=recursive, file_filter=file_filter)
            
            # Process documents
            return self.process_documents(documents, batch_size=batch_size, **kwargs)
            
        finally:
            if directory_path:
                # Restore original directory
                self.data_loader.data_directory = original_dir
    
    def process_documents(self, 
                         documents: List[Union[str, Path]],
                         batch_size: Optional[int] = None,
                         **kwargs) -> BatchResult:
        """
        Process a list of documents.
        
        Args:
            documents: List of document paths
            batch_size: Optional batch size for processing chunks
            **kwargs: Additional arguments passed to processors
            
        Returns:
            BatchResult with processing results
        """
        batch_id = str(uuid.uuid4())
        documents = [Path(d) for d in documents]
        
        # Initialize batch status
        status = BatchStatus(
            batch_id=batch_id,
            total_documents=len(documents)
        )
        
        with self._status_lock:
            self._active_batches[batch_id] = status
        
        # Initialize batch result
        batch_result = BatchResult(batch_id=batch_id, status=status)
        
        try:
            if batch_size and len(documents) > batch_size:
                # Process in chunks
                all_results = []
                for i in range(0, len(documents), batch_size):
                    chunk = documents[i:i + batch_size]
                    chunk_results = self._process_document_chunk(chunk, status, **kwargs)
                    all_results.extend(chunk_results)
                
                batch_result.results = all_results
            else:
                # Process all at once
                batch_result.results = self._process_document_chunk(documents, status, **kwargs)
            
            # Finalize results
            batch_result.finalize()
            
            logger.info(f"Batch {batch_id} completed: {status.successful_documents}/{status.total_documents} successful")
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Batch processing failed for {batch_id}: {e}")
            raise
        finally:
            # Clean up status tracking
            with self._status_lock:
                self._active_batches.pop(batch_id, None)
    
    def _process_document_chunk(self, 
                               documents: List[Path],
                               status: BatchStatus,
                               **kwargs) -> List[ProcessingResult]:
        """Process a chunk of documents in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_doc = {}
            for doc_path in documents:
                future = executor.submit(self._process_single_document, doc_path, **kwargs)
                future_to_doc[future] = doc_path
            
            # Collect results as they complete
            for future in as_completed(future_to_doc):
                doc_path = future_to_doc[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update status
                    with self._status_lock:
                        status.update_progress(
                            document_processed=True,
                            success=result.success,
                            current_file=doc_path.name
                        )
                    
                    # Call progress callback if provided
                    if self.progress_callback:
                        self.progress_callback(status)
                    
                except Exception as e:
                    logger.error(f"Error processing {doc_path}: {e}")
                    
                    # Create error result
                    error_result = ProcessingResult(
                        success=False,
                        document_id=str(uuid.uuid4()),
                        file_path=str(doc_path),
                        processing_time=0.0,
                        error_message=str(e)
                    )
                    results.append(error_result)
                    
                    # Update status
                    with self._status_lock:
                        status.update_progress(
                            document_processed=True,
                            success=False,
                            current_file=doc_path.name
                        )
                    
                    # Call progress callback if provided
                    if self.progress_callback:
                        self.progress_callback(status)
        
        return results
    
    def _process_single_document(self, file_path: Path, **kwargs) -> ProcessingResult:
        """Process a single document."""
        document_id = str(uuid.uuid4())
        
        try:
            # Use format registry to process the file
            result = self.format_registry.process_file(file_path, document_id, **kwargs)
            
            logger.debug(f"Processed {file_path.name}: {'success' if result.success else 'failed'}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            
            return ProcessingResult(
                success=False,
                document_id=document_id,
                file_path=str(file_path),
                processing_time=0.0,
                error_message=str(e)
            )
    
    def process_by_format(self, 
                         file_extension: str,
                         directory_path: Union[str, Path] = None,
                         **kwargs) -> BatchResult:
        """
        Process all documents of a specific format.
        
        Args:
            file_extension: File extension to filter by (e.g., '.pdf')
            directory_path: Directory to search (defaults to data loader directory)
            **kwargs: Additional arguments passed to processors
            
        Returns:
            BatchResult with processing results
        """
        def format_filter(path: Path) -> bool:
            return path.suffix.lower() == file_extension.lower()
        
        return self.process_directory(
            directory_path=directory_path,
            file_filter=format_filter,
            **kwargs
        )
    
    def get_batch_status(self, batch_id: str) -> Optional[BatchStatus]:
        """Get current status of a batch."""
        with self._status_lock:
            return self._active_batches.get(batch_id)
    
    def get_active_batches(self) -> List[BatchStatus]:
        """Get all currently active batches."""
        with self._status_lock:
            return list(self._active_batches.values())
    
    def cancel_batch(self, batch_id: str) -> bool:
        """Cancel an active batch (placeholder for future implementation)."""
        # This would require more sophisticated thread management
        # For now, just remove from tracking
        with self._status_lock:
            if batch_id in self._active_batches:
                del self._active_batches[batch_id]
                return True
        return False
    
    def get_processing_statistics(self) -> Dict:
        """Get overall processing statistics."""
        stats = {
            'active_batches': len(self._active_batches),
            'max_workers': self.max_workers,
            'supported_formats': self.format_registry.get_supported_formats(),
            'handler_info': self.format_registry.get_handler_info()
        }
        
        return stats


class ProgressMonitor:
    """Helper class for monitoring batch processing progress."""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize progress monitor.
        
        Args:
            update_interval: Minimum seconds between progress updates
        """
        self.update_interval = update_interval
        self.last_update = 0.0
        self.progress_history: List[Dict] = []
    
    def __call__(self, status: BatchStatus):
        """Progress callback function."""
        current_time = time.time()
        
        # Rate limit updates
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # Store progress snapshot
        progress_snapshot = status.to_dict()
        self.progress_history.append(progress_snapshot)
        
        # Log progress
        progress_pct = status.get_progress_percentage()
        success_rate = status.get_success_rate()
        
        logger.info(
            f"Batch {status.batch_id}: {progress_pct:.1f}% complete "
            f"({status.processed_documents}/{status.total_documents}), "
            f"Success rate: {success_rate:.1f}%, "
            f"Speed: {status.processing_speed:.2f} docs/sec"
        )
        
        if status.current_file:
            logger.debug(f"Currently processing: {status.current_file}")
    
    def get_progress_history(self) -> List[Dict]:
        """Get full progress history."""
        return self.progress_history.copy()
    
    def clear_history(self):
        """Clear progress history."""
        self.progress_history.clear()