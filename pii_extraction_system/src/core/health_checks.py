"""Comprehensive health check system for the PII extraction system."""

import asyncio
import time
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

from loguru import logger
from .environment_manager import get_env_manager
from .error_handling import handle_error, ErrorSeverity, ErrorCategory


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheckResult]
    summary: Dict[str, Any]


class HealthChecker:
    """Health check executor."""
    
    def __init__(self, name: str, check_func: Callable[[], Dict[str, Any]], timeout: int = 30):
        """Initialize health checker."""
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.last_result: Optional[HealthCheckResult] = None
        self.history: List[HealthCheckResult] = []
        self.max_history = 100
    
    def execute(self) -> HealthCheckResult:
        """Execute health check."""
        start_time = time.time()
        
        try:
            # Execute check with timeout
            result = self._execute_with_timeout()
            response_time = (time.time() - start_time) * 1000
            
            # Determine status based on result
            if result.get("healthy", False):
                status = HealthStatus.HEALTHY
            elif result.get("degraded", False):
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY
            
            # Create result
            health_result = HealthCheckResult(
                name=self.name,
                status=status,
                message=result.get("message", "Check completed"),
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details=result.get("details", {}),
                error=result.get("error")
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            health_result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                error=str(e)
            )
            
            handle_error(e, context={"health_check": self.name})
        
        # Store result
        self.last_result = health_result
        self.history.append(health_result)
        
        # Maintain history limit
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return health_result
    
    def _execute_with_timeout(self) -> Dict[str, Any]:
        """Execute check function with timeout."""
        # For now, simple synchronous execution
        # In production, implement proper timeout handling
        return self.check_func()
    
    def get_availability(self, hours: int = 24) -> float:
        """Get availability percentage for the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_checks = [c for c in self.history if c.timestamp > cutoff]
        
        if not recent_checks:
            return 0.0
        
        healthy_checks = len([c for c in recent_checks if c.status == HealthStatus.HEALTHY])
        return (healthy_checks / len(recent_checks)) * 100


class HealthMonitor:
    """Central health monitoring system."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.checkers: Dict[str, HealthChecker] = {}
        self.env_manager = get_env_manager()
        self._register_default_checks()
    
    def _register_default_checks(self) -> None:
        """Register default health checks."""
        
        # System health checks
        self.register_check("cpu", self._check_cpu_usage)
        self.register_check("memory", self._check_memory_usage)
        self.register_check("disk", self._check_disk_usage)
        
        # Application health checks
        self.register_check("database", self._check_database)
        self.register_check("cache", self._check_cache)
        self.register_check("file_system", self._check_file_system)
        
        # External service checks
        if not self.env_manager.is_development():
            self.register_check("aws_s3", self._check_aws_s3)
            self.register_check("external_apis", self._check_external_apis)
        
        # ML model checks
        self.register_check("ml_models", self._check_ml_models)
    
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]], timeout: int = 30) -> None:
        """Register a health check."""
        self.checkers[name] = HealthChecker(name, check_func, timeout)
        logger.info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if name not in self.checkers:
            logger.warning(f"Health check '{name}' not found")
            return None
        
        return self.checkers[name].execute()
    
    def run_all_checks(self) -> SystemHealth:
        """Run all health checks."""
        results = []
        
        for name, checker in self.checkers.items():
            try:
                result = checker.execute()
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run health check '{name}': {e}")
                results.append(HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {str(e)}",
                    timestamp=datetime.now(),
                    response_time_ms=0,
                    error=str(e)
                ))
        
        # Determine overall status
        overall_status = self._determine_overall_status(results)
        
        # Create summary
        summary = self._create_summary(results)
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            checks=results,
            summary=summary
        )
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system status."""
        if not results:
            return HealthStatus.UNKNOWN
        
        # Count statuses
        status_counts = {}
        for result in results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Determine overall status
        if status_counts.get(HealthStatus.UNHEALTHY, 0) > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts.get(HealthStatus.DEGRADED, 0) > 0:
            return HealthStatus.DEGRADED
        elif status_counts.get(HealthStatus.HEALTHY, 0) > 0:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _create_summary(self, results: List[HealthCheckResult]) -> Dict[str, Any]:
        """Create health check summary."""
        status_counts = {}
        total_response_time = 0
        failed_checks = []
        
        for result in results:
            status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1
            total_response_time += result.response_time_ms
            
            if result.status == HealthStatus.UNHEALTHY:
                failed_checks.append(result.name)
        
        return {
            "total_checks": len(results),
            "status_counts": status_counts,
            "average_response_time_ms": total_response_time / len(results) if results else 0,
            "failed_checks": failed_checks
        }
    
    # System health checks
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent < 70:
                return {"healthy": True, "message": f"CPU usage: {cpu_percent}%", "details": {"cpu_percent": cpu_percent}}
            elif cpu_percent < 90:
                return {"degraded": True, "message": f"High CPU usage: {cpu_percent}%", "details": {"cpu_percent": cpu_percent}}
            else:
                return {"healthy": False, "message": f"Critical CPU usage: {cpu_percent}%", "details": {"cpu_percent": cpu_percent}}
        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent < 80:
                return {"healthy": True, "message": f"Memory usage: {memory_percent}%", "details": {"memory_percent": memory_percent, "available_gb": memory.available / (1024**3)}}
            elif memory_percent < 95:
                return {"degraded": True, "message": f"High memory usage: {memory_percent}%", "details": {"memory_percent": memory_percent, "available_gb": memory.available / (1024**3)}}
            else:
                return {"healthy": False, "message": f"Critical memory usage: {memory_percent}%", "details": {"memory_percent": memory_percent, "available_gb": memory.available / (1024**3)}}
        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            if disk_percent < 80:
                return {"healthy": True, "message": f"Disk usage: {disk_percent:.1f}%", "details": {"disk_percent": disk_percent, "free_gb": disk.free / (1024**3)}}
            elif disk_percent < 95:
                return {"degraded": True, "message": f"High disk usage: {disk_percent:.1f}%", "details": {"disk_percent": disk_percent, "free_gb": disk.free / (1024**3)}}
            else:
                return {"healthy": False, "message": f"Critical disk usage: {disk_percent:.1f}%", "details": {"disk_percent": disk_percent, "free_gb": disk.free / (1024**3)}}
        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            db_config = self.env_manager.get_database_config()
            db_url = db_config.get("url")
            
            if not db_url:
                return {"healthy": False, "message": "Database URL not configured"}
            
            if db_url.startswith("sqlite"):
                # Check SQLite file exists and is accessible
                if ":///" in db_url:
                    db_path = Path(db_url.split("///")[1])
                    if db_path.exists():
                        return {"healthy": True, "message": "SQLite database accessible", "details": {"db_path": str(db_path), "size_mb": db_path.stat().st_size / (1024*1024)}}
                    else:
                        return {"healthy": False, "message": "SQLite database file not found"}
                else:
                    return {"healthy": True, "message": "SQLite in-memory database"}
            
            else:
                # TODO: Implement PostgreSQL/MySQL connection check
                return {"healthy": True, "message": "Database connection check not implemented for non-SQLite"}
        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_cache(self) -> Dict[str, Any]:
        """Check cache system (Redis)."""
        try:
            redis_url = self.env_manager.get_config_value("REDIS_URL")
            
            if not redis_url:
                return {"healthy": True, "message": "Cache not configured (optional)"}
            
            # TODO: Implement Redis connectivity check
            return {"healthy": True, "message": "Cache check not implemented"}
        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_file_system(self) -> Dict[str, Any]:
        """Check file system accessibility."""
        try:
            # Check required directories
            required_dirs = ["data", "logs", "data/models", "data/processed"]
            missing_dirs = []
            
            for dir_name in required_dirs:
                dir_path = Path(dir_name)
                if not dir_path.exists():
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                return {"healthy": False, "message": f"Missing directories: {missing_dirs}", "details": {"missing_dirs": missing_dirs}}
            
            # Check write permissions
            test_file = Path("logs/health_check_test.tmp")
            try:
                test_file.write_text("test")
                test_file.unlink()
                return {"healthy": True, "message": "File system accessible", "details": {"checked_dirs": required_dirs}}
            except Exception:
                return {"healthy": False, "message": "No write permission to logs directory"}
        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_aws_s3(self) -> Dict[str, Any]:
        """Check AWS S3 connectivity."""
        try:
            aws_config = self.env_manager.get_aws_config()
            s3_bucket = aws_config.get("s3_bucket")
            
            if not s3_bucket:
                return {"healthy": True, "message": "S3 not configured (optional)"}
            
            # TODO: Implement S3 connectivity check
            return {"healthy": True, "message": "S3 check not implemented"}
        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API connectivity."""
        try:
            # Check HuggingFace API
            hf_token = self.env_manager.get_config_value("ML_MODELS__HUGGINGFACE_TOKEN")
            
            if not hf_token:
                return {"healthy": True, "message": "External APIs not configured (optional)"}
            
            # TODO: Implement HuggingFace API check
            return {"healthy": True, "message": "External API checks not implemented"}
        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def _check_ml_models(self) -> Dict[str, Any]:
        """Check ML models availability."""
        try:
            model_cache_dir = Path(self.env_manager.get_config_value("ML_MODELS__MODEL_CACHE_DIR", "./data/models"))
            
            if not model_cache_dir.exists():
                return {"degraded": True, "message": "Model cache directory not found", "details": {"model_cache_dir": str(model_cache_dir)}}
            
            # Check for required model files
            model_files = list(model_cache_dir.rglob("*.json")) + list(model_cache_dir.rglob("*.safetensors"))
            
            if len(model_files) == 0:
                return {"degraded": True, "message": "No model files found", "details": {"model_cache_dir": str(model_cache_dir)}}
            
            return {"healthy": True, "message": f"Found {len(model_files)} model files", "details": {"model_files_count": len(model_files), "model_cache_dir": str(model_cache_dir)}}
        
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health."""
        return self.run_all_checks()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        system_health = self.get_system_health()
        
        return {
            "status": system_health.status.value,
            "timestamp": system_health.timestamp.isoformat(),
            "summary": system_health.summary,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms
                }
                for check in system_health.checks
            ]
        }
    
    def export_health_report(self, output_file: Path) -> None:
        """Export comprehensive health report."""
        system_health = self.get_system_health()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "environment": self.env_manager.get_current_environment(),
            "system_health": {
                "status": system_health.status.value,
                "timestamp": system_health.timestamp.isoformat(),
                "summary": system_health.summary
            },
            "checks": [asdict(check) for check in system_health.checks],
            "availability": {
                name: checker.get_availability()
                for name, checker in self.checkers.items()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Health report exported to {output_file}")


# Global health monitor instance
health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    return health_monitor


def get_system_health() -> SystemHealth:
    """Get current system health."""
    return health_monitor.get_system_health()


def health_check_endpoint() -> Dict[str, Any]:
    """Health check endpoint for web services."""
    try:
        health_summary = health_monitor.get_health_summary()
        
        # Return appropriate HTTP status based on health
        if health_summary["status"] == "healthy":
            return {"status": "ok", "health": health_summary}
        elif health_summary["status"] == "degraded":
            return {"status": "warning", "health": health_summary}
        else:
            return {"status": "error", "health": health_summary}
    
    except Exception as e:
        handle_error(e, context={"endpoint": "health_check"})
        return {"status": "error", "message": "Health check failed", "error": str(e)}