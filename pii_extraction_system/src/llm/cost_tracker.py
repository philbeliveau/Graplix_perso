"""
Enhanced Cost Tracking System for Multi-LLM Integration

This module provides comprehensive cost tracking, usage monitoring, and analytics
for all LLM API calls with persistent storage and real-time monitoring.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class BudgetCheckResult:
    """Result of a budget check operation"""
    can_afford: bool
    estimated_cost: float
    remaining_daily_budget: float
    remaining_monthly_budget: float
    daily_usage: float
    monthly_usage: float
    daily_limit: float
    monthly_limit: float
    warning_messages: List[str]
    blocking_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class UsageRecord:
    """Single usage record for LLM API call"""
    timestamp: datetime
    session_id: str
    provider: str
    model: str
    task_type: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    actual_cost: Optional[float] = None
    processing_time: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UsageRecord':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class CostTracker:
    """Enhanced cost tracker with persistent storage and analytics"""
    
    def __init__(self, db_path: Optional[str] = None, session_id: Optional[str] = None):
        """
        Initialize cost tracker
        
        Args:
            db_path: Path to SQLite database file
            session_id: Unique session identifier
        """
        self.db_path = db_path or "data/llm_usage.db"
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._lock = threading.Lock()
        
        # Ensure directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # In-memory cache for current session
        self._session_cache: List[UsageRecord] = []
        
        logger.info(f"Cost tracker initialized with session ID: {self.session_id}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        with self._get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    estimated_cost REAL NOT NULL,
                    actual_cost REAL,
                    processing_time REAL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    user_id TEXT,
                    document_id TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON llm_usage(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_session ON llm_usage(session_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_provider ON llm_usage(provider)
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def record_usage(
        self,
        provider: str,
        model: str,
        task_type: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
        actual_cost: Optional[float] = None,
        processing_time: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> str:
        """
        Record LLM usage
        
        Args:
            provider: API provider name
            model: Model name
            task_type: Type of task performed
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            estimated_cost: Estimated cost in USD
            actual_cost: Actual cost if known
            processing_time: Processing time in seconds
            success: Whether the request was successful
            error_message: Error message if failed
            user_id: User identifier
            document_id: Document identifier
            
        Returns:
            Record ID
        """
        record = UsageRecord(
            timestamp=datetime.now(),
            session_id=self.session_id,
            provider=provider,
            model=model,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated_cost=estimated_cost,
            actual_cost=actual_cost,
            processing_time=processing_time,
            success=success,
            error_message=error_message,
            user_id=user_id,
            document_id=document_id
        )
        
        # Add to session cache
        with self._lock:
            self._session_cache.append(record)
        
        # Store in database
        record_id = self._store_record(record)
        
        logger.debug(f"Recorded usage: {provider}/{model} - ${estimated_cost:.6f}")
        return record_id
    
    def _store_record(self, record: UsageRecord) -> str:
        """Store record in database"""
        with self._get_db_connection() as conn:
            cursor = conn.execute('''
                INSERT INTO llm_usage (
                    timestamp, session_id, provider, model, task_type,
                    input_tokens, output_tokens, total_tokens,
                    estimated_cost, actual_cost, processing_time,
                    success, error_message, user_id, document_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.timestamp.isoformat(),
                record.session_id,
                record.provider,
                record.model,
                record.task_type,
                record.input_tokens,
                record.output_tokens,
                record.total_tokens,
                record.estimated_cost,
                record.actual_cost,
                record.processing_time,
                record.success,
                record.error_message,
                record.user_id,
                record.document_id
            ))
            conn.commit()
            return str(cursor.lastrowid)
    
    def get_session_stats(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific session"""
        target_session = session_id or self.session_id
        
        with self._get_db_connection() as conn:
            # Basic stats
            stats = conn.execute('''
                SELECT
                    COUNT(*) as total_requests,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(estimated_cost) as total_estimated_cost,
                    SUM(actual_cost) as total_actual_cost,
                    AVG(processing_time) as avg_processing_time,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_requests
                FROM llm_usage
                WHERE session_id = ?
            ''', (target_session,)).fetchone()
            
            # Provider breakdown
            provider_stats = conn.execute('''
                SELECT
                    provider,
                    COUNT(*) as requests,
                    SUM(estimated_cost) as cost,
                    SUM(total_tokens) as tokens
                FROM llm_usage
                WHERE session_id = ?
                GROUP BY provider
                ORDER BY cost DESC
            ''', (target_session,)).fetchall()
            
            # Model breakdown
            model_stats = conn.execute('''
                SELECT
                    provider || '/' || model as model_name,
                    COUNT(*) as requests,
                    SUM(estimated_cost) as cost,
                    SUM(total_tokens) as tokens,
                    AVG(processing_time) as avg_time
                FROM llm_usage
                WHERE session_id = ?
                GROUP BY provider, model
                ORDER BY cost DESC
            ''', (target_session,)).fetchall()
            
            # Calculate success rate with proper null handling
            success_rate = 0
            if stats:
                successful = stats['successful_requests'] or 0
                total = stats['total_requests'] or 0
                success_rate = (successful / max(total, 1)) * 100 if total > 0 else 0
            
            return {
                'session_id': target_session,
                'summary': dict(stats) if stats else {},
                'by_provider': [dict(row) for row in provider_stats],
                'by_model': [dict(row) for row in model_stats],
                'success_rate': success_rate
            }
    
    def get_daily_costs(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get costs for a specific day"""
        target_date = date or datetime.now()
        date_str = target_date.strftime('%Y-%m-%d')
        
        with self._get_db_connection() as conn:
            daily_stats = conn.execute('''
                SELECT
                    COUNT(*) as total_requests,
                    SUM(estimated_cost) as total_cost,
                    SUM(total_tokens) as total_tokens,
                    provider,
                    SUM(estimated_cost) as provider_cost
                FROM llm_usage
                WHERE DATE(timestamp) = ?
                GROUP BY provider
                ORDER BY provider_cost DESC
            ''', (date_str,)).fetchall()
            
            total_cost = conn.execute('''
                SELECT SUM(estimated_cost) as total
                FROM llm_usage
                WHERE DATE(timestamp) = ?
            ''', (date_str,)).fetchone()
            
            return {
                'date': date_str,
                'total_cost': total_cost['total'] or 0,
                'by_provider': [dict(row) for row in daily_stats]
            }
    
    def get_monthly_costs(self, year: Optional[int] = None, month: Optional[int] = None) -> Dict[str, Any]:
        """Get costs for a specific month"""
        now = datetime.now()
        target_year = year or now.year
        target_month = month or now.month
        
        with self._get_db_connection() as conn:
            monthly_stats = conn.execute('''
                SELECT
                    COUNT(*) as total_requests,
                    SUM(estimated_cost) as total_cost,
                    SUM(total_tokens) as total_tokens,
                    provider
                FROM llm_usage
                WHERE strftime('%Y', timestamp) = ? AND strftime('%m', timestamp) = ?
                GROUP BY provider
                ORDER BY total_cost DESC
            ''', (str(target_year), f"{target_month:02d}")).fetchall()
            
            daily_breakdown = conn.execute('''
                SELECT
                    DATE(timestamp) as date,
                    SUM(estimated_cost) as daily_cost
                FROM llm_usage
                WHERE strftime('%Y', timestamp) = ? AND strftime('%m', timestamp) = ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (str(target_year), f"{target_month:02d}")).fetchall()
            
            total_cost = sum(row['total_cost'] for row in monthly_stats)
            
            return {
                'year': target_year,
                'month': target_month,
                'total_cost': total_cost,
                'by_provider': [dict(row) for row in monthly_stats],
                'daily_breakdown': [dict(row) for row in daily_breakdown]
            }
    
    def get_cost_analysis(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive cost analysis for the last N days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with self._get_db_connection() as conn:
            # Overall trends
            trends = conn.execute('''
                SELECT
                    DATE(timestamp) as date,
                    COUNT(*) as requests,
                    SUM(estimated_cost) as cost,
                    SUM(total_tokens) as tokens
                FROM llm_usage
                WHERE timestamp >= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''', (start_date.isoformat(),)).fetchall()
            
            # Most expensive models
            expensive_models = conn.execute('''
                SELECT
                    provider || '/' || model as model_name,
                    COUNT(*) as requests,
                    SUM(estimated_cost) as total_cost,
                    AVG(estimated_cost) as avg_cost_per_request,
                    SUM(total_tokens) as total_tokens
                FROM llm_usage
                WHERE timestamp >= ?
                GROUP BY provider, model
                ORDER BY total_cost DESC
                LIMIT 10
            ''', (start_date.isoformat(),)).fetchall()
            
            # Usage patterns by hour
            hourly_patterns = conn.execute('''
                SELECT
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as requests,
                    SUM(estimated_cost) as cost
                FROM llm_usage
                WHERE timestamp >= ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''', (start_date.isoformat(),)).fetchall()
            
            # Error analysis
            error_analysis = conn.execute('''
                SELECT
                    provider,
                    model,
                    COUNT(*) as error_count,
                    error_message
                FROM llm_usage
                WHERE timestamp >= ? AND success = 0
                GROUP BY provider, model, error_message
                ORDER BY error_count DESC
            ''', (start_date.isoformat(),)).fetchall()
            
            total_cost = sum(row['cost'] for row in trends)
            total_requests = sum(row['requests'] for row in trends)
            
            return {
                'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                'total_cost': total_cost,
                'total_requests': total_requests,
                'avg_cost_per_request': total_cost / max(total_requests, 1),
                'daily_trends': [dict(row) for row in trends],
                'expensive_models': [dict(row) for row in expensive_models],
                'hourly_patterns': [dict(row) for row in hourly_patterns],
                'error_analysis': [dict(row) for row in error_analysis]
            }
    
    def export_usage_data(
        self,
        output_path: str,
        format: str = 'json',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        session_id: Optional[str] = None
    ):
        """
        Export usage data to file
        
        Args:
            output_path: Output file path
            format: Export format ('json', 'csv')
            start_date: Start date filter
            end_date: End date filter
            session_id: Session ID filter
        """
        # Build query
        where_conditions = []
        params = []
        
        if start_date:
            where_conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())
        
        if end_date:
            where_conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())
        
        if session_id:
            where_conditions.append("session_id = ?")
            params.append(session_id)
        
        where_clause = " WHERE " + " AND ".join(where_conditions) if where_conditions else ""
        
        with self._get_db_connection() as conn:
            rows = conn.execute(f'''
                SELECT * FROM llm_usage
                {where_clause}
                ORDER BY timestamp
            ''', params).fetchall()
            
            data = [dict(row) for row in rows]
            
            if format.lower() == 'json':
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            elif format.lower() == 'csv':
                import csv
                if data:
                    with open(output_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
            
            logger.info(f"Exported {len(data)} records to {output_path}")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics for current session"""
        with self._lock:
            if not self._session_cache:
                return {
                    'session_id': self.session_id,
                    'total_requests': 0,
                    'total_cost': 0,
                    'total_tokens': 0,
                    'success_rate': 0,
                    'recent_activity': []
                }
            
            total_cost = sum(r.estimated_cost for r in self._session_cache)
            total_tokens = sum(r.total_tokens for r in self._session_cache)
            successful = sum(1 for r in self._session_cache if r.success)
            
            # Get recent activity (last 10 requests)
            recent = sorted(self._session_cache, key=lambda x: x.timestamp, reverse=True)[:10]
            
            return {
                'session_id': self.session_id,
                'total_requests': len(self._session_cache),
                'total_cost': total_cost,
                'total_tokens': total_tokens,
                'success_rate': (successful / len(self._session_cache)) * 100,
                'recent_activity': [
                    {
                        'timestamp': r.timestamp.isoformat(),
                        'provider': r.provider,
                        'model': r.model,
                        'cost': r.estimated_cost,
                        'tokens': r.total_tokens,
                        'success': r.success
                    }
                    for r in recent
                ]
            }
    
    def estimate_cost_before_call(
        self,
        provider: str,
        model: str,
        estimated_input_tokens: Optional[int] = None,
        estimated_output_tokens: Optional[int] = None,
        safety_margin: float = 1.1
    ) -> float:
        """
        Estimate cost for an API call before making it
        
        Args:
            provider: API provider name
            model: Model name
            estimated_input_tokens: Estimated input tokens
            estimated_output_tokens: Estimated output tokens
            safety_margin: Safety margin multiplier for estimation
        
        Returns:
            Estimated cost in USD
        """
        # Default token estimates if not provided
        if estimated_input_tokens is None:
            estimated_input_tokens = 800  # Conservative estimate for typical PII extraction prompt
        if estimated_output_tokens is None:
            estimated_output_tokens = 500  # Conservative estimate for typical JSON response
        
        # Model-specific cost per token (per 1000 tokens)
        cost_per_1k_tokens = {
            'openai': {
                'gpt-4o': {'input': 0.0025, 'output': 0.01},
                'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
                'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
                'gpt-4': {'input': 0.03, 'output': 0.06}
            },
            'anthropic': {
                'claude-3-5-sonnet-20241022': {'input': 0.003, 'output': 0.015},
                'claude-3-5-haiku-20241022': {'input': 0.001, 'output': 0.005},
                'claude-3-opus-20240229': {'input': 0.015, 'output': 0.075}
            },
            'google': {
                'gemini-1.5-pro': {'input': 0.0025, 'output': 0.0075},
                'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003}
            },
            'mistral': {
                'mistral-large': {'input': 0.002, 'output': 0.006},
                'mistral-medium': {'input': 0.0015, 'output': 0.0045},
                'mistral-small': {'input': 0.0006, 'output': 0.0018},
                'mistral-tiny': {'input': 0.00025, 'output': 0.00025}
            }
        }
        
        # Get cost information
        provider_costs = cost_per_1k_tokens.get(provider.lower(), {})
        model_costs = provider_costs.get(model, {'input': 0.001, 'output': 0.003})  # Fallback
        
        # Calculate estimated cost
        input_cost = (estimated_input_tokens / 1000) * model_costs['input']
        output_cost = (estimated_output_tokens / 1000) * model_costs['output']
        total_cost = (input_cost + output_cost) * safety_margin
        
        logger.debug(f"Estimated cost for {provider}/{model}: ${total_cost:.6f} (input: {estimated_input_tokens}, output: {estimated_output_tokens})")
        return total_cost
    
    def get_remaining_budget(self, provider: str, budget_config: Optional[Any] = None) -> Dict[str, float]:
        """
        Get remaining budget for a provider
        
        Args:
            provider: Provider name
            budget_config: Budget configuration object (optional)
        
        Returns:
            Dictionary with remaining daily and monthly budgets
        """
        now = datetime.now()
        
        # Get current usage
        daily_costs = self.get_daily_costs(now)
        monthly_costs = self.get_monthly_costs(now.year, now.month)
        
        # Find provider usage
        daily_usage = 0.0
        monthly_usage = 0.0
        
        for provider_data in daily_costs.get('by_provider', []):
            if provider_data.get('provider', '').lower() == provider.lower():
                daily_usage = provider_data.get('provider_cost', 0)
                break
        
        for provider_data in monthly_costs.get('by_provider', []):
            if provider_data.get('provider', '').lower() == provider.lower():
                monthly_usage = provider_data.get('total_cost', 0)
                break
        
        # Get budget limits (use defaults if config not provided)
        if budget_config:
            daily_limit = budget_config.get_daily_limit(provider)
            monthly_limit = budget_config.get_monthly_limit(provider)
        else:
            # Default limits
            daily_limit = 10.0
            monthly_limit = 100.0
        
        return {
            'daily_usage': daily_usage,
            'monthly_usage': monthly_usage,
            'daily_limit': daily_limit,
            'monthly_limit': monthly_limit,
            'remaining_daily': max(0, daily_limit - daily_usage),
            'remaining_monthly': max(0, monthly_limit - monthly_usage)
        }
    
    def can_afford(
        self,
        provider: str,
        estimated_cost: float,
        budget_config: Optional[Any] = None,
        enforce_limits: bool = True
    ) -> BudgetCheckResult:
        """
        Check if we can afford an API call before making it
        
        Args:
            provider: Provider name
            estimated_cost: Estimated cost of the API call
            budget_config: Budget configuration object
            enforce_limits: Whether to enforce budget limits
        
        Returns:
            BudgetCheckResult with detailed information
        """
        budget_info = self.get_remaining_budget(provider, budget_config)
        warning_messages = []
        blocking_reason = None
        
        # Check against daily limit
        remaining_daily = budget_info['remaining_daily']
        daily_usage = budget_info['daily_usage']
        daily_limit = budget_info['daily_limit']
        
        # Check against monthly limit
        remaining_monthly = budget_info['remaining_monthly']
        monthly_usage = budget_info['monthly_usage']
        monthly_limit = budget_info['monthly_limit']
        
        # Determine if we can afford this call
        can_afford_daily = remaining_daily >= estimated_cost
        can_afford_monthly = remaining_monthly >= estimated_cost
        can_afford = can_afford_daily and can_afford_monthly
        
        # Generate warnings
        if budget_config:
            warning_threshold = getattr(budget_config, 'budget_warning_threshold', 0.8)
            
            # Daily warnings
            daily_percentage = (daily_usage + estimated_cost) / daily_limit if daily_limit > 0 else 0
            if daily_percentage >= warning_threshold:
                warning_messages.append(
                    f"Daily budget warning for {provider}: {daily_percentage:.1%} of limit would be used after this call"
                )
            
            # Monthly warnings
            monthly_percentage = (monthly_usage + estimated_cost) / monthly_limit if monthly_limit > 0 else 0
            if monthly_percentage >= warning_threshold:
                warning_messages.append(
                    f"Monthly budget warning for {provider}: {monthly_percentage:.1%} of limit would be used after this call"
                )
        
        # Determine blocking reason if applicable
        if enforce_limits and not can_afford:
            if not can_afford_daily:
                blocking_reason = f"Daily budget exceeded: ${estimated_cost:.4f} requested, ${remaining_daily:.4f} remaining"
            elif not can_afford_monthly:
                blocking_reason = f"Monthly budget exceeded: ${estimated_cost:.4f} requested, ${remaining_monthly:.4f} remaining"
        
        # Override can_afford if enforcement is disabled
        if not enforce_limits:
            can_afford = True
            blocking_reason = None
        
        return BudgetCheckResult(
            can_afford=can_afford,
            estimated_cost=estimated_cost,
            remaining_daily_budget=remaining_daily,
            remaining_monthly_budget=remaining_monthly,
            daily_usage=daily_usage,
            monthly_usage=monthly_usage,
            daily_limit=daily_limit,
            monthly_limit=monthly_limit,
            warning_messages=warning_messages,
            blocking_reason=blocking_reason
        )
    
    def check_emergency_stop(
        self,
        provider: str,
        budget_config: Optional[Any] = None
    ) -> bool:
        """
        Check if emergency stop should be triggered
        
        Args:
            provider: Provider name
            budget_config: Budget configuration object
        
        Returns:
            True if emergency stop should be triggered
        """
        if not budget_config or not getattr(budget_config, 'enable_emergency_stop', False):
            return False
        
        budget_info = self.get_remaining_budget(provider, budget_config)
        emergency_multiplier = getattr(budget_config, 'emergency_stop_multiplier', 1.2)
        
        # Check if usage has exceeded emergency threshold
        daily_emergency_threshold = budget_info['daily_limit'] * emergency_multiplier
        monthly_emergency_threshold = budget_info['monthly_limit'] * emergency_multiplier
        
        daily_exceeded = budget_info['daily_usage'] >= daily_emergency_threshold
        monthly_exceeded = budget_info['monthly_usage'] >= monthly_emergency_threshold
        
        if daily_exceeded or monthly_exceeded:
            logger.critical(f"EMERGENCY STOP triggered for {provider}: Usage exceeded emergency threshold")
            return True
        
        return False


class TokenUsageMonitor:
    """Real-time token usage monitoring"""
    
    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker
        self._daily_limits = {}
        self._monthly_limits = {}
        self._alerts_sent = set()
    
    def set_daily_limit(self, provider: str, limit: float):
        """Set daily cost limit for a provider"""
        self._daily_limits[provider] = limit
    
    def set_monthly_limit(self, provider: str, limit: float):
        """Set monthly cost limit for a provider"""
        self._monthly_limits[provider] = limit
    
    def check_limits(self, provider: str) -> Dict[str, Any]:
        """Check if provider is approaching or exceeding limits"""
        now = datetime.now()
        
        # Check daily limit
        daily_cost = self.cost_tracker.get_daily_costs()
        provider_daily = next((p for p in daily_cost['by_provider'] if p['provider'] == provider), {})
        daily_usage = provider_daily.get('provider_cost', 0)
        
        # Check monthly limit
        monthly_cost = self.cost_tracker.get_monthly_costs()
        provider_monthly = next((p for p in monthly_cost['by_provider'] if p['provider'] == provider), {})
        monthly_usage = provider_monthly.get('total_cost', 0)
        
        alerts = []
        
        # Daily limit checks
        if provider in self._daily_limits:
            daily_limit = self._daily_limits[provider]
            daily_percentage = (daily_usage / daily_limit) * 100
            
            if daily_percentage >= 100:
                alerts.append({
                    'type': 'daily_limit_exceeded',
                    'message': f"Daily limit exceeded for {provider}: ${daily_usage:.2f} / ${daily_limit:.2f}"
                })
            elif daily_percentage >= 80:
                alerts.append({
                    'type': 'daily_limit_warning',
                    'message': f"Daily limit warning for {provider}: {daily_percentage:.1f}% used"
                })
        
        # Monthly limit checks
        if provider in self._monthly_limits:
            monthly_limit = self._monthly_limits[provider]
            monthly_percentage = (monthly_usage / monthly_limit) * 100
            
            if monthly_percentage >= 100:
                alerts.append({
                    'type': 'monthly_limit_exceeded',
                    'message': f"Monthly limit exceeded for {provider}: ${monthly_usage:.2f} / ${monthly_limit:.2f}"
                })
            elif monthly_percentage >= 80:
                alerts.append({
                    'type': 'monthly_limit_warning',
                    'message': f"Monthly limit warning for {provider}: {monthly_percentage:.1f}% used"
                })
        
        return {
            'provider': provider,
            'daily_usage': daily_usage,
            'monthly_usage': monthly_usage,
            'daily_limit': self._daily_limits.get(provider),
            'monthly_limit': self._monthly_limits.get(provider),
            'alerts': alerts
        }


# Global instances
default_cost_tracker = CostTracker()
token_monitor = TokenUsageMonitor(default_cost_tracker)

# Alias for backward compatibility
cost_tracker = default_cost_tracker