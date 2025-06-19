"""Performance monitoring system for the PII extraction system."""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
from pathlib import Path
import statistics

from loguru import logger
from .environment_manager import get_env_manager
from .error_handling import handle_error, ErrorSeverity, ErrorCategory


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str]


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    name: str
    metric: str
    threshold: float
    comparison: str  # '>', '<', '>=', '<=', '=='
    duration_seconds: int
    severity: str
    message: str
    enabled: bool = True


class MetricsCollector:
    """Metrics collection and storage."""
    
    def __init__(self, max_points: int = 10000):
        """Initialize metrics collector."""
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.alerts: List[PerformanceAlert] = []
        self.alert_states: Dict[str, Dict[str, Any]] = {}
        self.collection_interval = 30  # seconds
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Setup default alerts
        self._setup_default_alerts()
    
    def _setup_default_alerts(self) -> None:
        """Setup default performance alerts."""
        env_manager = get_env_manager()
        monitoring_config = env_manager.get_monitoring_config()
        
        default_alerts = [
            PerformanceAlert(
                name="high_cpu_usage",
                metric="system.cpu_percent",
                threshold=85.0,
                comparison=">",
                duration_seconds=300,  # 5 minutes
                severity="warning",
                message="High CPU usage detected"
            ),
            PerformanceAlert(
                name="critical_cpu_usage",
                metric="system.cpu_percent",
                threshold=95.0,
                comparison=">",
                duration_seconds=60,  # 1 minute
                severity="critical",
                message="Critical CPU usage detected"
            ),
            PerformanceAlert(
                name="high_memory_usage",
                metric="system.memory_percent",
                threshold=85.0,
                comparison=">",
                duration_seconds=300,
                severity="warning",
                message="High memory usage detected"
            ),
            PerformanceAlert(
                name="critical_memory_usage",
                metric="system.memory_percent",
                threshold=95.0,
                comparison=">",
                duration_seconds=60,
                severity="critical",
                message="Critical memory usage detected"
            ),
            PerformanceAlert(
                name="low_disk_space",
                metric="system.disk_percent",
                threshold=85.0,
                comparison=">",
                duration_seconds=600,  # 10 minutes
                severity="warning",
                message="Low disk space detected"
            ),
            PerformanceAlert(
                name="slow_processing",
                metric="pii.processing_time_ms",
                threshold=monitoring_config.get("performance_threshold_ms", 5000),
                comparison=">",
                duration_seconds=120,
                severity="warning",
                message="Slow processing detected"
            ),
            PerformanceAlert(
                name="high_error_rate",
                metric="pii.error_rate",
                threshold=0.1,  # 10%
                comparison=">",
                duration_seconds=180,
                severity="critical",
                message="High error rate detected"
            )
        ]
        
        self.alerts.extend(default_alerts)
        
        # Initialize alert states
        for alert in self.alerts:
            self.alert_states[alert.name] = {
                "triggered": False,
                "trigger_time": None,
                "last_notification": None
            }
    
    def add_metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Add a metric point."""
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {}
        )
        
        self.metrics[name].append(point)
        
        # Check alerts for this metric
        self._check_alerts(name, value)
    
    def _check_alerts(self, metric_name: str, current_value: float) -> None:
        """Check if any alerts should be triggered."""
        current_time = datetime.now()
        
        for alert in self.alerts:
            if not alert.enabled or alert.metric != metric_name:
                continue
            
            alert_state = self.alert_states[alert.name]
            
            # Check if threshold is breached
            threshold_breached = self._evaluate_threshold(current_value, alert.threshold, alert.comparison)
            
            if threshold_breached:
                if not alert_state["triggered"]:
                    # Start tracking this potential alert
                    alert_state["trigger_time"] = current_time
                    alert_state["triggered"] = True
                elif alert_state["trigger_time"]:
                    # Check if duration threshold is met
                    duration = (current_time - alert_state["trigger_time"]).total_seconds()
                    if duration >= alert.duration_seconds:
                        self._trigger_alert(alert, current_value)
                        alert_state["last_notification"] = current_time
            else:
                # Reset alert state
                alert_state["triggered"] = False
                alert_state["trigger_time"] = None
    
    def _evaluate_threshold(self, value: float, threshold: float, comparison: str) -> bool:
        """Evaluate if value meets threshold condition."""
        if comparison == ">":
            return value > threshold
        elif comparison == "<":
            return value < threshold
        elif comparison == ">=":
            return value >= threshold
        elif comparison == "<=":
            return value <= threshold
        elif comparison == "==":
            return value == threshold
        else:
            return False
    
    def _trigger_alert(self, alert: PerformanceAlert, current_value: float) -> None:
        """Trigger a performance alert."""
        alert_data = {
            "alert_name": alert.name,
            "metric": alert.metric,
            "current_value": current_value,
            "threshold": alert.threshold,
            "severity": alert.severity,
            "message": alert.message,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.bind(alert=True).warning(f"Performance alert: {alert.message}", **alert_data)
        
        # Send to external alerting systems
        self._send_external_alert(alert_data)
    
    def _send_external_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send alert to external systems."""
        try:
            # TODO: Implement Slack, email, webhook notifications
            logger.info(f"External alert notification: {alert_data['alert_name']}")
        except Exception as e:
            logger.error(f"Failed to send external alert: {e}")
    
    def get_metric_values(self, name: str, since: datetime = None) -> List[MetricPoint]:
        """Get metric values since timestamp."""
        if name not in self.metrics:
            return []
        
        points = list(self.metrics[name])
        
        if since:
            points = [p for p in points if p.timestamp >= since]
        
        return points
    
    def get_metric_statistics(self, name: str, window_minutes: int = 60) -> Dict[str, float]:
        """Get metric statistics for a time window."""
        since = datetime.now() - timedelta(minutes=window_minutes)
        points = self.get_metric_values(name, since)
        
        if not points:
            return {}
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def start_collection(self) -> None:
        """Start automatic metrics collection."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.thread.start()
        logger.info("Performance metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Performance metrics collection stopped")
    
    def _collection_loop(self) -> None:
        """Main collection loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                handle_error(e, context={"component": "metrics_collector"})
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.add_metric("system.cpu_percent", cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.add_metric("system.memory_percent", memory.percent)
        self.add_metric("system.memory_available_gb", memory.available / (1024**3))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.add_metric("system.disk_percent", disk_percent)
        self.add_metric("system.disk_free_gb", disk.free / (1024**3))
        
        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            self.add_metric("system.network_bytes_sent", network.bytes_sent)
            self.add_metric("system.network_bytes_recv", network.bytes_recv)
        except:
            pass
        
        # Process metrics
        try:
            process = psutil.Process()
            self.add_metric("process.cpu_percent", process.cpu_percent())
            self.add_metric("process.memory_mb", process.memory_info().rss / (1024*1024))
            self.add_metric("process.threads", process.num_threads())
        except:
            pass


class PerformanceMonitor:
    """Main performance monitoring system."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.collector = MetricsCollector()
        self.custom_metrics: Dict[str, Callable] = {}
        self.processing_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.env_manager = get_env_manager()
        
        # Start collection if monitoring is enabled
        if self.env_manager.get_monitoring_config().get("enabled", False):
            self.collector.start_collection()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.collector.stop_collection()
    
    def add_custom_metric(self, name: str, collector_func: Callable[[], float]) -> None:
        """Add a custom metric collector."""
        self.custom_metrics[name] = collector_func
        logger.info(f"Added custom metric: {name}")
    
    def record_processing_time(self, duration_ms: float, operation: str = "unknown") -> None:
        """Record processing time."""
        self.processing_times.append(duration_ms)
        self.collector.add_metric("pii.processing_time_ms", duration_ms, {"operation": operation})
    
    def record_error(self, error_type: str) -> None:
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        
        # Calculate error rate
        total_operations = len(self.processing_times)
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / max(total_operations, 1)
        
        self.collector.add_metric("pii.error_rate", error_rate, {"error_type": error_type})
        self.collector.add_metric("pii.error_count", self.error_counts[error_type], {"error_type": error_type})
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "application": {},
            "alerts": []
        }
        
        # System metrics
        for metric_name in ["system.cpu_percent", "system.memory_percent", "system.disk_percent"]:
            stats = self.collector.get_metric_statistics(metric_name, window_minutes=15)
            if stats:
                summary["system"][metric_name.split('.')[1]] = stats
        
        # Application metrics
        processing_stats = self.collector.get_metric_statistics("pii.processing_time_ms", window_minutes=60)
        if processing_stats:
            summary["application"]["processing_time_ms"] = processing_stats
        
        error_rate_stats = self.collector.get_metric_statistics("pii.error_rate", window_minutes=60)
        if error_rate_stats:
            summary["application"]["error_rate"] = error_rate_stats
        
        # Active alerts
        for alert in self.collector.alerts:
            alert_state = self.collector.alert_states[alert.name]
            if alert_state["triggered"]:
                summary["alerts"].append({
                    "name": alert.name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "triggered_at": alert_state["trigger_time"].isoformat() if alert_state["trigger_time"] else None
                })
        
        return summary
    
    def export_metrics(self, output_file: Path, window_hours: int = 24) -> None:
        """Export metrics to file."""
        since = datetime.now() - timedelta(hours=window_hours)
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "window_hours": window_hours,
            "metrics": {}
        }
        
        for metric_name in self.collector.metrics.keys():
            points = self.collector.get_metric_values(metric_name, since)
            export_data["metrics"][metric_name] = [
                {
                    "timestamp": p.timestamp.isoformat(),
                    "value": p.value,
                    "tags": p.tags
                }
                for p in points
            ]
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Metrics exported to {output_file}")
    
    def create_performance_dashboard_data(self) -> Dict[str, Any]:
        """Create data for performance dashboard."""
        dashboard_data = {
            "realtime": {
                "cpu": self._get_latest_metric_value("system.cpu_percent"),
                "memory": self._get_latest_metric_value("system.memory_percent"),
                "disk": self._get_latest_metric_value("system.disk_percent"),
                "processing_time": self._get_latest_metric_value("pii.processing_time_ms"),
                "error_rate": self._get_latest_metric_value("pii.error_rate")
            },
            "trends": {},
            "alerts": []
        }
        
        # Get trend data for the last 6 hours
        for metric_name in ["system.cpu_percent", "system.memory_percent", "pii.processing_time_ms"]:
            since = datetime.now() - timedelta(hours=6)
            points = self.collector.get_metric_values(metric_name, since)
            
            dashboard_data["trends"][metric_name] = [
                {
                    "x": p.timestamp.isoformat(),
                    "y": p.value
                }
                for p in points[-100:]  # Last 100 points
            ]
        
        # Get active alerts
        for alert in self.collector.alerts:
            alert_state = self.collector.alert_states[alert.name]
            if alert_state["triggered"]:
                dashboard_data["alerts"].append({
                    "name": alert.name,
                    "severity": alert.severity,
                    "message": alert.message,
                    "duration": (datetime.now() - alert_state["trigger_time"]).total_seconds() if alert_state["trigger_time"] else 0
                })
        
        return dashboard_data
    
    def _get_latest_metric_value(self, metric_name: str) -> Optional[float]:
        """Get latest value for a metric."""
        if metric_name not in self.collector.metrics:
            return None
        
        points = self.collector.metrics[metric_name]
        return points[-1].value if points else None


# Performance timing decorator
def monitor_performance(operation: str = "unknown"):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                performance_monitor.record_processing_time(duration_ms, operation)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                performance_monitor.record_processing_time(duration_ms, operation)
                performance_monitor.record_error(type(e).__name__)
                raise
        return wrapper
    return decorator


# Context manager for performance monitoring
class PerformanceContext:
    """Context manager for performance monitoring."""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000
        performance_monitor.record_processing_time(duration_ms, self.operation)
        
        if exc_type is not None:
            performance_monitor.record_error(exc_type.__name__)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return performance_monitor


def record_processing_time(duration_ms: float, operation: str = "unknown") -> None:
    """Record processing time."""
    performance_monitor.record_processing_time(duration_ms, operation)


def record_error(error_type: str) -> None:
    """Record an error."""
    performance_monitor.record_error(error_type)