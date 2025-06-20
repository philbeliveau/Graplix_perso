"""
Background document processing for enterprise-scale batch operations.

This module handles parallel processing of thousands of documents with
progress tracking, cost monitoring, and error recovery.
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import multiprocessing

from database.document_store import get_document_db, DocumentRecord
from utils.s3_integration import S3DocumentProcessor, convert_document_to_images
from llm.multimodal_llm_service import llm_service
from llm.cost_tracker import cost_tracker
from core.logging_config import get_logger

logger = get_logger(__name__)


class ProgressTracker:
    """Track processing progress for large batches"""
    
    def __init__(self, batch_id: str, total_documents: int):
        self.batch_id = batch_id
        self.total_documents = total_documents
        self.processed = 0
        self.failed = 0
        self.total_cost = 0.0
        self.start_time = time.time()
        self.db = get_document_db()
    
    def update_progress(self, processed_count: int = 1, cost: float = 0.0, failed: bool = False):
        """Update processing progress"""
        if failed:
            self.failed += 1
        else:
            self.processed += processed_count
        
        self.total_cost += cost
        
        # Calculate metrics
        progress = (self.processed + self.failed) / self.total_documents
        elapsed_time = time.time() - self.start_time
        
        if progress > 0:
            estimated_total_time = elapsed_time / progress
            eta = estimated_total_time - elapsed_time
        else:
            eta = 0
        
        # Log progress every 10 documents or every 30 seconds
        if (self.processed + self.failed) % 10 == 0 or elapsed_time % 30 < 1:
            message = f"Progress: {self.processed + self.failed}/{self.total_documents} ({progress:.1%}), Cost: ${self.total_cost:.2f}, ETA: {eta/60:.1f}m"
            self.db.log_batch_event(self.batch_id, 'info', message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        elapsed_time = time.time() - self.start_time
        progress = (self.processed + self.failed) / self.total_documents if self.total_documents > 0 else 0
        
        if progress > 0:
            estimated_total_time = elapsed_time / progress
            eta_seconds = max(0, estimated_total_time - elapsed_time)
        else:
            eta_seconds = 0
        
        return {
            'batch_id': self.batch_id,
            'total_documents': self.total_documents,
            'processed': self.processed,
            'failed': self.failed,
            'progress': progress,
            'total_cost': self.total_cost,
            'elapsed_time': elapsed_time,
            'eta_seconds': eta_seconds,
            'documents_per_minute': (self.processed + self.failed) / max(elapsed_time / 60, 0.1)
        }


class BackgroundProcessor:
    """Background processor for large document batches"""
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize background processor"""
        # Optimize worker count for I/O-bound tasks
        if max_workers is None:
            max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
        
        self.max_workers = max_workers
        self.db = get_document_db()
        self.active_jobs = {}  # Track active processing jobs
        
        logger.info(f"Background processor initialized with {max_workers} workers")
    
    def start_batch_processing(self, batch_id: str, 
                             model_key: str = "openai/gpt-4o-mini",
                             password: str = "Hubert",
                             max_budget: float = 500.0,
                             progress_callback: Optional[Callable] = None) -> str:
        """Start background processing of a batch"""
        
        # Check if batch exists and is not already processing
        batch_info = self.db.get_batch_info(batch_id)
        if not batch_info:
            raise ValueError(f"Batch {batch_id} not found")
        
        if batch_info.status in ['processing', 'completed']:
            raise ValueError(f"Batch {batch_id} is already {batch_info.status}")
        
        # Mark batch as processing
        self.db.conn.execute("""
            UPDATE batches SET status = 'processing', started_at = ?
            WHERE batch_id = ?
        """, (datetime.now().isoformat(), batch_id))
        self.db.conn.commit()
        
        # Start processing in background thread
        job_id = f"job_{batch_id}"
        processing_thread = threading.Thread(
            target=self._process_batch_worker,
            args=(batch_id, model_key, password, max_budget, progress_callback),
            daemon=True
        )
        
        self.active_jobs[job_id] = {
            'thread': processing_thread,
            'batch_id': batch_id,
            'started_at': datetime.now(),
            'status': 'starting'
        }
        
        processing_thread.start()
        
        self.db.log_batch_event(batch_id, 'info', f"Background processing started with model {model_key}")
        logger.info(f"Started background processing for batch {batch_id}")
        
        return job_id
    
    def _process_batch_worker(self, batch_id: str, model_key: str, password: str, 
                             max_budget: float, progress_callback: Optional[Callable]):
        """Worker function for batch processing"""
        try:
            # Initialize progress tracker
            batch_info = self.db.get_batch_info(batch_id)
            progress_tracker = ProgressTracker(batch_id, batch_info.total_documents)
            
            # Update job status
            job_id = f"job_{batch_id}"
            if job_id in self.active_jobs:
                self.active_jobs[job_id]['status'] = 'processing'
            
            # Process documents in chunks to manage memory and cost
            chunk_size = 50  # Process 50 documents at a time
            total_cost = 0.0
            
            while True:
                # Get next chunk of pending documents
                pending_docs = self.db.get_pending_documents(batch_id, limit=chunk_size)
                
                if not pending_docs:
                    break  # No more pending documents
                
                # Check budget before processing chunk
                if total_cost >= max_budget:
                    self.db.log_batch_event(batch_id, 'warning', f"Budget limit reached: ${total_cost:.2f} >= ${max_budget}")
                    break
                
                # Process chunk in parallel
                chunk_results = self._process_document_chunk(
                    pending_docs, model_key, password, max_budget - total_cost
                )
                
                # Update database with results
                for doc_id, result in chunk_results.items():
                    if result['success']:
                        self.db.update_document_status(
                            doc_id, 'completed', 
                            processing_result=result,
                            cost=result.get('cost', 0),
                            processing_time=result.get('processing_time', 0)
                        )
                        progress_tracker.update_progress(cost=result.get('cost', 0))
                    else:
                        self.db.update_document_status(doc_id, 'failed', processing_result=result)
                        progress_tracker.update_progress(failed=True)
                        
                        # Log specific error
                        error_msg = result.get('error', 'Unknown error')
                        self.db.log_batch_event(batch_id, 'error', f"Document failed: {error_msg}", doc_id)
                
                total_cost = progress_tracker.total_cost
                
                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(progress_tracker.get_stats())
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")
                
                # Brief pause to prevent overwhelming the API
                time.sleep(0.5)
            
            # Mark batch as completed
            final_status = 'completed'
            if progress_tracker.failed > 0:
                final_status = 'completed_with_errors'
            
            self.db.conn.execute("""
                UPDATE batches SET status = ?, completed_at = ?
                WHERE batch_id = ?
            """, (final_status, datetime.now().isoformat(), batch_id))
            self.db.conn.commit()
            
            # Update job status
            if job_id in self.active_jobs:
                self.active_jobs[job_id]['status'] = 'completed'
            
            final_message = f"Batch processing completed: {progress_tracker.processed} successful, {progress_tracker.failed} failed, ${total_cost:.2f} total cost"
            self.db.log_batch_event(batch_id, 'info', final_message)
            logger.info(f"Batch {batch_id} processing completed")
            
        except Exception as e:
            # Mark batch as failed
            self.db.conn.execute("""
                UPDATE batches SET status = 'failed'
                WHERE batch_id = ?
            """, (batch_id,))
            self.db.conn.commit()
            
            # Update job status
            if job_id in self.active_jobs:
                self.active_jobs[job_id]['status'] = 'failed'
            
            error_msg = f"Batch processing failed: {str(e)}"
            self.db.log_batch_event(batch_id, 'error', error_msg)
            logger.error(f"Batch {batch_id} processing failed: {e}")
    
    def _process_document_chunk(self, documents: List[DocumentRecord], 
                               model_key: str, password: str, remaining_budget: float) -> Dict[str, Dict]:
        """Process a chunk of documents in parallel"""
        results = {}
        
        # Use ThreadPoolExecutor for I/O-bound tasks
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(documents))) as executor:
            # Submit all documents for processing
            future_to_doc = {
                executor.submit(self._process_single_document, doc, model_key, password): doc 
                for doc in documents
            }
            
            current_cost = 0.0
            
            # Process completed futures as they finish
            for future in as_completed(future_to_doc):
                doc = future_to_doc[future]
                
                try:
                    result = future.result()
                    current_cost += result.get('cost', 0)
                    
                    # Stop processing if budget exceeded
                    if current_cost > remaining_budget:
                        result['success'] = False
                        result['error'] = f"Budget exceeded: ${current_cost:.2f} > ${remaining_budget:.2f}"
                    
                    results[doc.document_id] = result
                    
                except Exception as e:
                    results[doc.document_id] = {
                        'success': False,
                        'error': str(e),
                        'cost': 0,
                        'processing_time': 0
                    }
        
        return results
    
    def _process_single_document(self, document: DocumentRecord, 
                                model_key: str, password: str) -> Dict[str, Any]:
        """Process a single document"""
        start_time = time.time()
        
        try:
            # Pre-flight budget check
            provider = model_key.split('/')[0] if '/' in model_key else 'openai'
            estimated_cost = 0.03  # Rough estimate per document
            
            if not cost_tracker.can_afford(provider, estimated_cost):
                return {
                    'success': False,
                    'error': 'Insufficient budget for processing',
                    'cost': 0,
                    'processing_time': time.time() - start_time
                }
            
            # For S3 documents, we need to download and process
            if document.s3_key:
                # This would be handled by S3DocumentProcessor
                # For now, simulate processing
                result = self._simulate_s3_document_processing(document, model_key)
            else:
                # Process uploaded document (from session state era)
                result = {
                    'success': False,
                    'error': 'Local document processing not implemented in background processor',
                    'cost': 0
                }
            
            result['processing_time'] = time.time() - start_time
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'cost': 0,
                'processing_time': time.time() - start_time
            }
    
    def _simulate_s3_document_processing(self, document: DocumentRecord, model_key: str) -> Dict[str, Any]:
        """Simulate S3 document processing (placeholder for actual implementation)"""
        
        # In a real implementation, this would:
        # 1. Download document from S3
        # 2. Convert to images using convert_document_to_images
        # 3. Process with LLM using llm_service.extract_pii_from_image
        # 4. Return structured results
        
        # For demonstration, return a simulated successful result
        import random
        
        # Simulate processing time
        time.sleep(random.uniform(1.0, 3.0))
        
        # Simulate random success/failure
        if random.random() > 0.1:  # 90% success rate
            return {
                'success': True,
                'pii_entities': [
                    {
                        'text': 'John Smith',
                        'type': 'person_name',
                        'confidence': 0.95,
                        'start_pos': 10,
                        'end_pos': 20
                    }
                ],
                'cost': random.uniform(0.01, 0.05),
                'method': f'simulated_{model_key}',
                'document_classification': {
                    'domain': 'Business',
                    'difficulty_level': 'Medium'
                }
            }
        else:
            return {
                'success': False,
                'error': 'Simulated processing failure',
                'cost': 0
            }
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a processing job"""
        if job_id not in self.active_jobs:
            return None
        
        job_info = self.active_jobs[job_id]
        batch_id = job_info['batch_id']
        
        # Get current batch info
        batch_info = self.db.get_batch_info(batch_id)
        
        return {
            'job_id': job_id,
            'batch_id': batch_id,
            'status': job_info['status'],
            'started_at': job_info['started_at'].isoformat(),
            'batch_status': batch_info.status if batch_info else 'unknown',
            'total_documents': batch_info.total_documents if batch_info else 0,
            'processed_documents': batch_info.processed_documents if batch_info else 0,
            'failed_documents': batch_info.failed_documents if batch_info else 0,
            'total_cost': batch_info.total_cost if batch_info else 0,
            'is_alive': job_info['thread'].is_alive()
        }
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job"""
        if job_id not in self.active_jobs:
            return False
        
        job_info = self.active_jobs[job_id]
        batch_id = job_info['batch_id']
        
        # Cancel the batch in database
        self.db.cancel_batch(batch_id)
        
        # Update job status
        job_info['status'] = 'cancelled'
        
        logger.info(f"Cancelled job {job_id} for batch {batch_id}")
        return True
    
    def cleanup_completed_jobs(self):
        """Clean up completed job references"""
        completed_jobs = []
        
        for job_id, job_info in self.active_jobs.items():
            if not job_info['thread'].is_alive() or job_info['status'] in ['completed', 'failed', 'cancelled']:
                completed_jobs.append(job_id)
        
        for job_id in completed_jobs:
            del self.active_jobs[job_id]
        
        return len(completed_jobs)


# Global processor instance
_processor_instance = None

def get_background_processor() -> BackgroundProcessor:
    """Get global background processor instance"""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = BackgroundProcessor()
    return _processor_instance