"""
Scalable document storage and management for enterprise-scale processing.

This module provides database persistence to replace session state for handling
10k+ documents without browser crashes or memory issues.
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging

from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BatchInfo:
    """Information about a document processing batch"""
    batch_id: str
    name: str
    status: str  # pending, processing, completed, failed, cancelled
    total_documents: int
    processed_documents: int
    failed_documents: int
    total_cost: float
    estimated_cost: float
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class DocumentRecord:
    """Individual document record in the database"""
    document_id: str
    batch_id: str
    filename: str
    s3_key: Optional[str]
    file_size: int
    document_type: str
    status: str  # pending, processing, completed, failed
    processing_result: Optional[Dict[str, Any]]
    cost: float
    processing_time: float
    created_at: str
    processed_at: Optional[str] = None
    metadata: Dict[str, Any] = None


class DocumentDatabase:
    """Scalable document database for enterprise processing"""
    
    def __init__(self, db_path: str = "data/documents.db"):
        """Initialize document database"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        self._setup_tables()
        logger.info(f"Document database initialized at {self.db_path}")
    
    def _setup_tables(self):
        """Setup database tables"""
        # Batches table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                total_documents INTEGER DEFAULT 0,
                processed_documents INTEGER DEFAULT 0,
                failed_documents INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                estimated_cost REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                metadata TEXT
            )
        """)
        
        # Documents table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                s3_key TEXT,
                file_size INTEGER DEFAULT 0,
                document_type TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                processing_result TEXT,
                cost REAL DEFAULT 0.0,
                processing_time REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                processed_at TEXT,
                metadata TEXT,
                FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
            )
        """)
        
        # Processing logs table for tracking progress
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS processing_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL,
                document_id TEXT,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (batch_id) REFERENCES batches (batch_id)
            )
        """)
        
        # Create indexes for performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_batch_id ON documents(batch_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_batch_id ON processing_logs(batch_id)")
        
        self.conn.commit()
    
    def create_batch(self, name: str, documents: List[Dict[str, Any]], estimated_cost: float = 0.0) -> str:
        """Create a new processing batch"""
        batch_id = str(uuid.uuid4())
        
        # Insert batch record
        self.conn.execute("""
            INSERT INTO batches (batch_id, name, total_documents, estimated_cost, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            batch_id, 
            name, 
            len(documents), 
            estimated_cost,
            datetime.now().isoformat(),
            json.dumps({'source': 's3_batch'})
        ))
        
        # Insert document records
        for doc in documents:
            document_id = str(uuid.uuid4())
            self.conn.execute("""
                INSERT INTO documents (
                    document_id, batch_id, filename, s3_key, file_size, 
                    document_type, created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document_id,
                batch_id,
                doc.get('filename', ''),
                doc.get('s3_key', ''),
                doc.get('file_size', 0),
                doc.get('document_type', 'unknown'),
                datetime.now().isoformat(),
                json.dumps(doc.get('metadata', {}))
            ))
        
        self.conn.commit()
        
        self.log_batch_event(batch_id, 'info', f"Batch created with {len(documents)} documents")
        logger.info(f"Created batch {batch_id} with {len(documents)} documents")
        
        return batch_id
    
    def get_batch_info(self, batch_id: str) -> Optional[BatchInfo]:
        """Get batch information"""
        cursor = self.conn.execute("""
            SELECT * FROM batches WHERE batch_id = ?
        """, (batch_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return BatchInfo(
            batch_id=row['batch_id'],
            name=row['name'],
            status=row['status'],
            total_documents=row['total_documents'],
            processed_documents=row['processed_documents'],
            failed_documents=row['failed_documents'],
            total_cost=row['total_cost'],
            estimated_cost=row['estimated_cost'],
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            metadata=json.loads(row['metadata'] or '{}')
        )
    
    def get_documents_paginated(self, batch_id: str, offset: int = 0, limit: int = 100, 
                               status_filter: Optional[str] = None) -> Tuple[List[DocumentRecord], int]:
        """Get documents with pagination"""
        
        # Build query with optional status filter
        where_clause = "WHERE batch_id = ?"
        params = [batch_id]
        
        if status_filter:
            where_clause += " AND status = ?"
            params.append(status_filter)
        
        # Get total count
        count_cursor = self.conn.execute(f"""
            SELECT COUNT(*) as total FROM documents {where_clause}
        """, params)
        total_count = count_cursor.fetchone()['total']
        
        # Get paginated results
        params.extend([limit, offset])
        cursor = self.conn.execute(f"""
            SELECT * FROM documents {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """, params)
        
        documents = []
        for row in cursor.fetchall():
            documents.append(DocumentRecord(
                document_id=row['document_id'],
                batch_id=row['batch_id'],
                filename=row['filename'],
                s3_key=row['s3_key'],
                file_size=row['file_size'],
                document_type=row['document_type'],
                status=row['status'],
                processing_result=json.loads(row['processing_result'] or '{}'),
                cost=row['cost'],
                processing_time=row['processing_time'],
                created_at=row['created_at'],
                processed_at=row['processed_at'],
                metadata=json.loads(row['metadata'] or '{}')
            ))
        
        return documents, total_count
    
    def update_document_status(self, document_id: str, status: str, 
                              processing_result: Optional[Dict] = None,
                              cost: float = 0.0, processing_time: float = 0.0):
        """Update document processing status"""
        
        update_fields = ["status = ?"]
        params = [status]
        
        if processing_result:
            update_fields.append("processing_result = ?")
            params.append(json.dumps(processing_result))
        
        if cost > 0:
            update_fields.append("cost = ?")
            params.append(cost)
        
        if processing_time > 0:
            update_fields.append("processing_time = ?")
            params.append(processing_time)
        
        if status in ['completed', 'failed']:
            update_fields.append("processed_at = ?")
            params.append(datetime.now().isoformat())
        
        params.append(document_id)
        
        self.conn.execute(f"""
            UPDATE documents 
            SET {', '.join(update_fields)}
            WHERE document_id = ?
        """, params)
        
        # Update batch statistics
        doc_cursor = self.conn.execute("SELECT batch_id FROM documents WHERE document_id = ?", (document_id,))
        batch_id = doc_cursor.fetchone()['batch_id']
        
        self._update_batch_stats(batch_id)
        self.conn.commit()
    
    def _update_batch_stats(self, batch_id: str):
        """Update batch statistics based on document status"""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(cost) as total_cost
            FROM documents 
            WHERE batch_id = ?
        """, (batch_id,))
        
        stats = cursor.fetchone()
        
        # Update batch status
        new_status = 'processing'
        if stats['completed'] + stats['failed'] == stats['total']:
            new_status = 'completed' if stats['failed'] == 0 else 'completed_with_errors'
        
        self.conn.execute("""
            UPDATE batches 
            SET processed_documents = ?, failed_documents = ?, total_cost = ?, status = ?
            WHERE batch_id = ?
        """, (stats['completed'], stats['failed'], stats['total_cost'], new_status, batch_id))
    
    def get_pending_documents(self, batch_id: str, limit: Optional[int] = None) -> List[DocumentRecord]:
        """Get pending documents for processing"""
        query = """
            SELECT * FROM documents 
            WHERE batch_id = ? AND status = 'pending'
            ORDER BY created_at
        """
        params = [batch_id]
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        cursor = self.conn.execute(query, params)
        
        documents = []
        for row in cursor.fetchall():
            documents.append(DocumentRecord(
                document_id=row['document_id'],
                batch_id=row['batch_id'],
                filename=row['filename'],
                s3_key=row['s3_key'],
                file_size=row['file_size'],
                document_type=row['document_type'],
                status=row['status'],
                processing_result=json.loads(row['processing_result'] or '{}'),
                cost=row['cost'],
                processing_time=row['processing_time'],
                created_at=row['created_at'],
                processed_at=row['processed_at'],
                metadata=json.loads(row['metadata'] or '{}')
            ))
        
        return documents
    
    def log_batch_event(self, batch_id: str, level: str, message: str, document_id: Optional[str] = None):
        """Log processing events"""
        self.conn.execute("""
            INSERT INTO processing_logs (batch_id, document_id, level, message, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (batch_id, document_id, level, message, datetime.now().isoformat()))
        
        self.conn.commit()
    
    def get_recent_logs(self, batch_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent processing logs"""
        cursor = self.conn.execute("""
            SELECT * FROM processing_logs 
            WHERE batch_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (batch_id, limit))
        
        logs = []
        for row in cursor.fetchall():
            logs.append({
                'log_id': row['log_id'],
                'batch_id': row['batch_id'],
                'document_id': row['document_id'],
                'level': row['level'],
                'message': row['message'],
                'timestamp': row['timestamp']
            })
        
        return logs
    
    def get_all_batches(self, limit: int = 100) -> List[BatchInfo]:
        """Get all processing batches"""
        cursor = self.conn.execute("""
            SELECT * FROM batches 
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        batches = []
        for row in cursor.fetchall():
            batches.append(BatchInfo(
                batch_id=row['batch_id'],
                name=row['name'],
                status=row['status'],
                total_documents=row['total_documents'],
                processed_documents=row['processed_documents'],
                failed_documents=row['failed_documents'],
                total_cost=row['total_cost'],
                estimated_cost=row['estimated_cost'],
                created_at=row['created_at'],
                started_at=row['started_at'],
                completed_at=row['completed_at'],
                metadata=json.loads(row['metadata'] or '{}')
            ))
        
        return batches
    
    def cancel_batch(self, batch_id: str):
        """Cancel a processing batch"""
        self.conn.execute("""
            UPDATE batches SET status = 'cancelled' WHERE batch_id = ?
        """, (batch_id,))
        
        self.conn.execute("""
            UPDATE documents SET status = 'cancelled' 
            WHERE batch_id = ? AND status = 'pending'
        """, (batch_id,))
        
        self.conn.commit()
        self.log_batch_event(batch_id, 'info', "Batch processing cancelled")
    
    def cleanup_old_batches(self, days_old: int = 30):
        """Clean up old completed batches"""
        cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
        
        cursor = self.conn.execute("""
            SELECT batch_id FROM batches 
            WHERE status IN ('completed', 'failed', 'cancelled') 
            AND created_at < ?
        """, (cutoff_date,))
        
        old_batches = [row['batch_id'] for row in cursor.fetchall()]
        
        for batch_id in old_batches:
            # Delete logs
            self.conn.execute("DELETE FROM processing_logs WHERE batch_id = ?", (batch_id,))
            # Delete documents
            self.conn.execute("DELETE FROM documents WHERE batch_id = ?", (batch_id,))
            # Delete batch
            self.conn.execute("DELETE FROM batches WHERE batch_id = ?", (batch_id,))
        
        self.conn.commit()
        logger.info(f"Cleaned up {len(old_batches)} old batches")
        
        return len(old_batches)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Global database instance
_db_instance = None

def get_document_db() -> DocumentDatabase:
    """Get global document database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DocumentDatabase()
    return _db_instance