"""
Run History and Performance Tracking

This module manages the storage and analysis of processing runs,
including cost tracking, model performance, and historical analysis.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class RunHistoryManager:
    """Manages processing run history and analytics"""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "data" / "run_history.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for run history"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    run_type TEXT NOT NULL,  -- 'single', 'batch'
                    model_used TEXT NOT NULL,
                    total_documents INTEGER NOT NULL,
                    successful_documents INTEGER NOT NULL,
                    total_cost REAL NOT NULL,
                    total_processing_time REAL NOT NULL,
                    average_confidence REAL,
                    total_entities INTEGER NOT NULL,
                    user_id TEXT,
                    settings_json TEXT,
                    summary_json TEXT
                )
            """)
            
            # Documents table (individual document results)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_results (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    document_name TEXT NOT NULL,
                    document_size INTEGER,
                    processing_time REAL NOT NULL,
                    cost REAL NOT NULL,
                    entities_found INTEGER NOT NULL,
                    average_confidence REAL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    entities_json TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (id)
                )
            """)
            
            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    total_runs INTEGER NOT NULL,
                    total_documents INTEGER NOT NULL,
                    total_cost REAL NOT NULL,
                    average_processing_time REAL NOT NULL,
                    average_entities_per_doc REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    UNIQUE(model_name, date)
                )
            """)
            
            # Cost tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cost_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    document_name TEXT,
                    FOREIGN KEY (run_id) REFERENCES runs (id)
                )
            """)
            
            conn.commit()
    
    def save_run(
        self, 
        run_type: str,
        model_used: str,
        document_results: List[Dict[str, Any]],
        settings: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """
        Save a complete processing run
        
        Args:
            run_type: 'single' or 'batch'
            model_used: Model identifier used
            document_results: List of individual document results
            settings: Processing settings used
            user_id: Optional user identifier
        
        Returns:
            Run ID
        """
        run_id = str(uuid.uuid4())
        
        # Calculate aggregated metrics
        total_documents = len(document_results)
        successful_documents = sum(1 for r in document_results if r.get("success", False))
        total_cost = sum(r.get("cost", 0) for r in document_results)
        total_processing_time = sum(r.get("processing_time", 0) for r in document_results)
        total_entities = sum(r.get("entities_found", 0) for r in document_results)
        
        # Calculate average confidence
        confidences = []
        for result in document_results:
            if result.get("success") and result.get("average_confidence"):
                confidences.append(result["average_confidence"])
        average_confidence = sum(confidences) / len(confidences) if confidences else None
        
        # Create summary
        summary = {
            "success_rate": successful_documents / total_documents if total_documents > 0 else 0,
            "cost_per_document": total_cost / total_documents if total_documents > 0 else 0,
            "entities_per_document": total_entities / total_documents if total_documents > 0 else 0,
            "processing_time_per_document": total_processing_time / total_documents if total_documents > 0 else 0,
            "failed_documents": [r["document_name"] for r in document_results if not r.get("success", False)]
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Insert run record
            cursor.execute("""
                INSERT INTO runs (
                    id, run_type, model_used, total_documents, successful_documents,
                    total_cost, total_processing_time, average_confidence, total_entities,
                    user_id, settings_json, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, run_type, model_used, total_documents, successful_documents,
                total_cost, total_processing_time, average_confidence, total_entities,
                user_id, json.dumps(settings), json.dumps(summary)
            ))
            
            # Insert document results
            for result in document_results:
                doc_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO document_results (
                        id, run_id, document_name, document_size, processing_time,
                        cost, entities_found, average_confidence, success, error_message,
                        entities_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, run_id, result.get("document_name", ""),
                    result.get("document_size", 0), result.get("processing_time", 0),
                    result.get("cost", 0), result.get("entities_found", 0),
                    result.get("average_confidence"), result.get("success", False),
                    result.get("error_message"), 
                    json.dumps(result.get("entities", [])),
                    json.dumps(result.get("metadata", {}))
                ))
                
                # Insert cost tracking
                if result.get("success") and "usage" in result:
                    usage = result["usage"]
                    cursor.execute("""
                        INSERT INTO cost_tracking (
                            run_id, model_name, input_tokens, output_tokens, cost, document_name
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        run_id, model_used, usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0), result.get("cost", 0),
                        result.get("document_name", "")
                    ))
            
            conn.commit()
        
        # Update model performance aggregates
        self._update_model_performance(model_used)
        
        return run_id
    
    def _update_model_performance(self, model_name: str):
        """Update daily model performance aggregates"""
        today = datetime.now().date()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Calculate today's metrics for this model
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT r.id) as total_runs,
                    COUNT(dr.id) as total_documents,
                    SUM(dr.cost) as total_cost,
                    AVG(dr.processing_time) as avg_processing_time,
                    AVG(dr.entities_found) as avg_entities,
                    AVG(CASE WHEN dr.success THEN 1.0 ELSE 0.0 END) as success_rate
                FROM runs r
                JOIN document_results dr ON r.id = dr.run_id
                WHERE r.model_used = ? AND DATE(r.timestamp) = ?
            """, (model_name, today))
            
            result = cursor.fetchone()
            if result and result[0] > 0:  # If there are runs today
                cursor.execute("""
                    INSERT OR REPLACE INTO model_performance (
                        model_name, date, total_runs, total_documents, total_cost,
                        average_processing_time, average_entities_per_doc, success_rate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (model_name, today, *result))
                
                conn.commit()
    
    def get_recent_runs(self, limit: int = 20, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent processing runs"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    id, timestamp, run_type, model_used, total_documents,
                    successful_documents, total_cost, total_processing_time,
                    total_entities, summary_json
                FROM runs
            """
            params = []
            
            if user_id:
                query += " WHERE user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            
            runs = []
            for row in cursor.fetchall():
                run_data = {
                    "id": row[0],
                    "timestamp": row[1],
                    "run_type": row[2],
                    "model_used": row[3],
                    "total_documents": row[4],
                    "successful_documents": row[5],
                    "total_cost": row[6],
                    "total_processing_time": row[7],
                    "total_entities": row[8],
                    "summary": json.loads(row[9]) if row[9] else {}
                }
                runs.append(run_data)
            
            return runs
    
    def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific run"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get run info
            cursor.execute("""
                SELECT * FROM runs WHERE id = ?
            """, (run_id,))
            
            run_row = cursor.fetchone()
            if not run_row:
                return None
            
            # Get column names
            columns = [description[0] for description in cursor.description]
            run_data = dict(zip(columns, run_row))
            
            # Parse JSON fields
            run_data["settings"] = json.loads(run_data["settings_json"]) if run_data["settings_json"] else {}
            run_data["summary"] = json.loads(run_data["summary_json"]) if run_data["summary_json"] else {}
            
            # Get document results
            cursor.execute("""
                SELECT * FROM document_results WHERE run_id = ? ORDER BY document_name
            """, (run_id,))
            
            doc_columns = [description[0] for description in cursor.description]
            documents = []
            for doc_row in cursor.fetchall():
                doc_data = dict(zip(doc_columns, doc_row))
                doc_data["entities"] = json.loads(doc_data["entities_json"]) if doc_data["entities_json"] else []
                doc_data["metadata"] = json.loads(doc_data["metadata_json"]) if doc_data["metadata_json"] else {}
                documents.append(doc_data)
            
            run_data["documents"] = documents
            
            return run_data
    
    def get_model_performance_history(
        self, 
        model_name: Optional[str] = None, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get model performance over time"""
        start_date = datetime.now().date() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM model_performance 
                WHERE date >= ?
            """
            params = [start_date]
            
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            
            query += " ORDER BY date DESC, model_name"
            
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            performance_data = []
            for row in cursor.fetchall():
                performance_data.append(dict(zip(columns, row)))
            
            return performance_data
    
    def get_cost_analysis(
        self, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cost analysis for a period"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    model_name,
                    COUNT(*) as total_calls,
                    SUM(cost) as total_cost,
                    AVG(cost) as avg_cost_per_call,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    DATE(timestamp) as date
                FROM cost_tracking
                WHERE timestamp BETWEEN ? AND ?
            """
            params = [start_date, end_date]
            
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            
            query += " GROUP BY model_name, DATE(timestamp) ORDER BY date DESC, model_name"
            
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            cost_data = []
            for row in cursor.fetchall():
                cost_data.append(dict(zip(columns, row)))
            
            # Calculate totals
            total_cost = sum(row["total_cost"] for row in cost_data)
            total_calls = sum(row["total_calls"] for row in cost_data)
            total_input_tokens = sum(row["total_input_tokens"] for row in cost_data)
            total_output_tokens = sum(row["total_output_tokens"] for row in cost_data)
            
            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "summary": {
                    "total_cost": total_cost,
                    "total_calls": total_calls,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "avg_cost_per_call": total_cost / total_calls if total_calls > 0 else 0
                },
                "daily_breakdown": cost_data
            }
    
    def get_model_comparison(self, days: int = 7) -> Dict[str, Any]:
        """Compare models over recent period"""
        start_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    r.model_used,
                    COUNT(DISTINCT r.id) as total_runs,
                    COUNT(dr.id) as total_documents,
                    SUM(dr.cost) as total_cost,
                    AVG(dr.processing_time) as avg_processing_time,
                    AVG(dr.entities_found) as avg_entities,
                    AVG(CASE WHEN dr.success THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(dr.average_confidence) as avg_confidence
                FROM runs r
                JOIN document_results dr ON r.id = dr.run_id
                WHERE r.timestamp >= ?
                GROUP BY r.model_used
                ORDER BY total_documents DESC
            """, (start_date,))
            
            columns = [description[0] for description in cursor.description]
            model_stats = []
            for row in cursor.fetchall():
                stats = dict(zip(columns, row))
                stats["cost_per_document"] = stats["total_cost"] / stats["total_documents"] if stats["total_documents"] > 0 else 0
                model_stats.append(stats)
            
            return {
                "period_days": days,
                "models": model_stats,
                "total_models": len(model_stats)
            }
    
    def export_run_data(self, run_id: str, format: str = "json") -> str:
        """Export run data in specified format"""
        run_data = self.get_run_details(run_id)
        if not run_data:
            raise ValueError(f"Run {run_id} not found")
        
        if format.lower() == "json":
            return json.dumps(run_data, indent=2, default=str)
        elif format.lower() == "csv":
            import pandas as pd
            
            # Create DataFrame from documents
            documents = run_data.get("documents", [])
            df = pd.DataFrame(documents)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def cleanup_old_runs(self, days_to_keep: int = 90):
        """Clean up old run data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get run IDs to delete
            cursor.execute("""
                SELECT id FROM runs WHERE timestamp < ?
            """, (cutoff_date,))
            
            old_run_ids = [row[0] for row in cursor.fetchall()]
            
            if old_run_ids:
                # Delete related records
                for run_id in old_run_ids:
                    cursor.execute("DELETE FROM document_results WHERE run_id = ?", (run_id,))
                    cursor.execute("DELETE FROM cost_tracking WHERE run_id = ?", (run_id,))
                
                # Delete runs
                cursor.execute("DELETE FROM runs WHERE timestamp < ?", (cutoff_date,))
                
                conn.commit()
                
                logger.info(f"Cleaned up {len(old_run_ids)} old runs")
            
            return len(old_run_ids)

# Global instance
run_history_manager = RunHistoryManager()