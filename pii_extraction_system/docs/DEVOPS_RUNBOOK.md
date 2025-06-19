# DevOps Runbook - PII Extraction System

## Table of Contents

1. [Overview](#overview)
2. [Environment Management](#environment-management)
3. [Deployment Procedures](#deployment-procedures)
4. [Monitoring and Alerting](#monitoring-and-alerting)
5. [Backup and Recovery](#backup-and-recovery)
6. [Troubleshooting](#troubleshooting)
7. [Security Operations](#security-operations)
8. [Performance Management](#performance-management)
9. [Incident Response](#incident-response)
10. [Maintenance Procedures](#maintenance-procedures)

---

## Overview

This runbook provides comprehensive operational procedures for the PII Extraction System. It covers deployment, monitoring, backup/recovery, and incident response procedures.

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Application   │    │    Database     │
│    (Nginx)      │────│   (Streamlit)   │────│   (SQLite/PG)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │      Cache      │              │
         └──────────────│     (Redis)     │──────────────┘
                        └─────────────────┘
                                │
                    ┌─────────────────┐
                    │   Monitoring    │
                    │  (Prometheus/   │
                    │    Grafana)     │
                    └─────────────────┘
```

### Key Components

- **Application Server**: Streamlit dashboard and processing engine
- **Database**: SQLite (dev) / PostgreSQL (prod)
- **Cache**: Redis for session management and caching
- **Storage**: Local filesystem / AWS S3
- **Monitoring**: Prometheus metrics, Grafana dashboards
- **Logging**: Structured logging with Loguru

---

## Environment Management

### Environment Configuration Files

| Environment | Config File | Description |
|-------------|-------------|-------------|
| Development | `.env.development` | Local development with minimal resources |
| Staging | `.env.staging` | Pre-production testing environment |
| Production | `.env.production` | Production environment with full features |

### Environment Switching

```bash
# Set environment
export ENVIRONMENT=development  # or staging, production

# Validate environment configuration
python3 -c "
from src.core.environment_manager import get_env_manager
env_manager = get_env_manager()
result = env_manager.validate_current_environment()
print(f'Environment valid: {result[\"valid\"]}')
if not result['valid']:
    print(f'Missing variables: {result[\"missing_required\"]}')
"
```

### Environment Variables

#### Required Variables (All Environments)
- `ENVIRONMENT`: Environment name (development/staging/production)
- `DATABASE_URL`: Database connection string
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARNING/ERROR)
- `SECRET_KEY`: Application secret key
- `ENCRYPTION_KEY`: Data encryption key

#### Production-Only Requirements
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key
- `DB_PASSWORD`: Database password
- `HUGGINGFACE_TOKEN`: HuggingFace API token

### Configuration Validation

```bash
# Check environment configuration
cd /path/to/pii_extraction_system
python3 -c "
from src.core.environment_manager import get_env_manager
env_manager = get_env_manager()
env_manager.export_environment_info('environment_status.json')
print('Environment status exported to environment_status.json')
"
```

---

## Deployment Procedures

### Pre-Deployment Checklist

- [ ] Environment variables configured
- [ ] Database connections tested
- [ ] Required directories exist
- [ ] Dependencies installed
- [ ] Health checks passing
- [ ] Backup completed

### Local Development Deployment

```bash
# 1. Clone repository
git clone <repository-url>
cd pii_extraction_system

# 2. Install dependencies
pip install poetry
poetry install

# 3. Configure environment
cp .env.example .env.development
# Edit .env.development with appropriate values

# 4. Initialize directories
mkdir -p data/{raw,processed,models} logs

# 5. Start application
poetry shell
streamlit run src/dashboard/main.py --server.port 8502
```

### Docker Deployment

```bash
# Build and start services
docker-compose --profile dev up -d

# Check service health
docker-compose ps
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Deployment

```bash
# 1. Deploy using deployment script
./scripts/deploy.sh production v1.0.0 deploy

# 2. Verify deployment
./scripts/deploy.sh production v1.0.0 health

# 3. Monitor deployment
./scripts/deploy.sh production v1.0.0 monitor
```

### Rollback Procedure

```bash
# Rollback to previous version
./scripts/deploy.sh production v0.9.0 rollback v1.0.0

# Verify rollback
curl -f http://localhost:8501/health
```

---

## Monitoring and Alerting

### Health Checks

#### Manual Health Check
```bash
# Run comprehensive health check
python3 -c "
from src.core.health_checks import get_system_health
health = get_system_health()
print(f'Overall Status: {health.status.value}')
for check in health.checks:
    print(f'{check.name}: {check.status.value} - {check.message}')
"
```

#### Automated Health Monitoring
```bash
# Start health monitoring
python3 -c "
from src.core.health_checks import get_health_monitor
monitor = get_health_monitor()
monitor.export_health_report('health_report.json')
print('Health report exported')
"
```

### Performance Monitoring

#### View Performance Metrics
```bash
# Get performance summary
python3 -c "
from src.core.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
summary = monitor.get_performance_summary()
import json
print(json.dumps(summary, indent=2))
"
```

#### Custom Metrics
```python
from src.core.performance_monitor import monitor_performance, PerformanceContext

# Decorator usage
@monitor_performance("document_processing")
def process_document(doc):
    # Processing logic here
    pass

# Context manager usage
with PerformanceContext("batch_processing"):
    # Batch processing logic here
    pass
```

### Alerting Configuration

Alerts are automatically configured for:
- High CPU usage (>85% for 5 minutes)
- High memory usage (>85% for 5 minutes)
- Low disk space (>85% for 10 minutes)
- Slow processing (>5000ms for 2 minutes)
- High error rate (>10% for 3 minutes)

---

## Backup and Recovery

### Automated Backup

```bash
# Full system backup
./scripts/backup/automated_backup.sh full 30

# Database only backup
./scripts/backup/automated_backup.sh database 7

# With S3 upload
./scripts/backup/automated_backup.sh full 30 my-backup-bucket
```

### Manual Backup

```bash
# Using backup manager directly
cd scripts/backup
python3 backup_manager.py backup --type full --retention-days 30

# List available backups
python3 backup_manager.py list

# Get backup statistics
python3 backup_manager.py stats
```

### Recovery Procedures

#### List Available Backups
```bash
./scripts/backup/disaster_recovery.sh list
```

#### Database Recovery
```bash
# Restore database from specific backup
./scripts/backup/disaster_recovery.sh database database_20231215_143022

# Emergency database recovery (latest backup)
./scripts/backup/disaster_recovery.sh emergency
```

#### Full System Recovery
```bash
# Full recovery from specific backup set
./scripts/backup/disaster_recovery.sh full full_20231215_143022

# Emergency recovery (latest available backups)
./scripts/backup/disaster_recovery.sh emergency
```

#### Recovery Verification
```bash
# Verify database integrity after recovery
python3 -c "
import sqlite3
conn = sqlite3.connect('data/pii_extraction.db')
result = conn.execute('PRAGMA integrity_check;').fetchone()
print(f'Database integrity: {result[0]}')
conn.close()
"

# Verify system health after recovery
python3 -c "
from src.core.health_checks import get_system_health
health = get_system_health()
print(f'System health after recovery: {health.status.value}')
"
```

---

## Troubleshooting

### Common Issues

#### 1. Application Won't Start

**Symptoms**: Application fails to start, connection errors

**Diagnosis**:
```bash
# Check environment configuration
python3 -c "
from src.core.environment_manager import get_env_manager
result = get_env_manager().validate_current_environment()
print(result)
"

# Check required directories
ls -la data/ logs/

# Check dependencies
poetry check
```

**Resolution**:
```bash
# Fix environment configuration
cp .env.example .env.development
# Edit configuration

# Create missing directories
mkdir -p data/{raw,processed,models} logs

# Reinstall dependencies
poetry install --sync
```

#### 2. High CPU/Memory Usage

**Symptoms**: Slow performance, high resource usage

**Diagnosis**:
```bash
# Check system resources
python3 -c "
import psutil
print(f'CPU: {psutil.cpu_percent()}%')
print(f'Memory: {psutil.virtual_memory().percent}%')
print(f'Disk: {psutil.disk_usage(\"/\").percent}%')
"

# Check process metrics
python3 -c "
from src.core.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
summary = monitor.get_performance_summary()
print(summary)
"
```

**Resolution**:
```bash
# Restart application
docker-compose restart pii-extraction-app

# Scale down processing if needed
# Edit configuration to reduce concurrent jobs
# Set PROCESSING__CONCURRENT_JOBS=2 in environment
```

#### 3. Database Connection Issues

**Symptoms**: Database errors, connection timeouts

**Diagnosis**:
```bash
# Test database connection
python3 -c "
from src.core.environment_manager import get_env_manager
db_config = get_env_manager().get_database_config()
print(f'Database URL: {db_config[\"url\"]}')
"

# Check database file (SQLite)
ls -la data/*.db

# Test connectivity
sqlite3 data/pii_extraction.db "SELECT 1;"
```

**Resolution**:
```bash
# For SQLite: Ensure file exists and has proper permissions
chmod 664 data/pii_extraction.db

# For PostgreSQL: Check connection and credentials
psql -h host -U user -d database -c "SELECT 1;"

# Recreate database if corrupted
rm data/pii_extraction.db
# Restart application to recreate
```

#### 4. Processing Errors

**Symptoms**: Document processing failures, extraction errors

**Diagnosis**:
```bash
# Check error logs
tail -f logs/errors.log

# Check processing configuration
python3 -c "
from src.core.environment_manager import get_env_manager
config = get_env_manager().get_config_value
print(f'Max file size: {config(\"PROCESSING__MAX_FILE_SIZE_MB\")} MB')
print(f'Timeout: {config(\"PROCESSING__TIMEOUT_SECONDS\")} seconds')
print(f'OCR Engine: {config(\"PROCESSING__OCR_ENGINE\")}')
"

# Test OCR dependencies
tesseract --version
python3 -c "import easyocr; print('EasyOCR available')"
```

**Resolution**:
```bash
# Install missing OCR dependencies
# macOS
brew install tesseract

# Ubuntu
sudo apt-get install tesseract-ocr

# Check file permissions and sizes
find data/raw -type f -exec ls -lh {} \;

# Increase timeout if needed
# Set PROCESSING__TIMEOUT_SECONDS=600 in environment
```

### Performance Troubleshooting

#### Slow Processing
```bash
# Check processing metrics
python3 -c "
from src.core.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
stats = monitor.collector.get_metric_statistics('pii.processing_time_ms', 60)
print(f'Average processing time: {stats.get(\"mean\", 0):.2f}ms')
print(f'Max processing time: {stats.get(\"max\", 0):.2f}ms')
"

# Optimize configuration
# Reduce concurrent jobs for memory-limited systems
# Enable GPU processing if available
# Use faster OCR models
```

#### Memory Leaks
```bash
# Monitor memory usage over time
python3 -c "
from src.core.performance_monitor import get_performance_monitor
monitor = get_performance_monitor()
points = monitor.collector.get_metric_values('system.memory_percent')
recent_points = points[-10:]  # Last 10 points
for point in recent_points:
    print(f'{point.timestamp}: {point.value:.2f}%')
"

# Restart application if memory usage is continuously increasing
docker-compose restart pii-extraction-app
```

---

## Security Operations

### Security Monitoring

```bash
# Check audit logs
tail -f logs/audit.log

# Verify encryption settings
python3 -c "
from src.core.environment_manager import get_env_manager
security_config = get_env_manager().get_security_config()
print(f'Authentication enabled: {security_config[\"enable_auth\"]}')
"
```

### Access Control

```bash
# Check file permissions
find . -type f -name "*.py" -exec ls -l {} \;
find data/ -type f -exec ls -l {} \;

# Verify environment file security
ls -la .env*
```

### Security Incident Response

1. **Isolate affected systems**
2. **Preserve evidence**
3. **Assess impact**
4. **Contain threat**
5. **Recover systems**
6. **Document incident**

---

## Performance Management

### Performance Baselines

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| CPU Usage | <70% | 70-90% | >90% |
| Memory Usage | <80% | 80-95% | >95% |
| Disk Usage | <80% | 80-95% | >95% |
| Processing Time | <3s | 3-10s | >10s |
| Error Rate | <1% | 1-5% | >5% |

### Performance Optimization

```bash
# Optimize processing configuration
# Adjust concurrent jobs based on system resources
PROCESSING__CONCURRENT_JOBS=4

# Enable GPU processing if available
PROCESSING__EASYOCR_USE_GPU=true

# Optimize ML model settings
ML_MODELS__ENABLED_MODELS=rule_based,ner  # Reduce models for faster processing
```

---

## Incident Response

### Incident Classification

- **P1 - Critical**: Complete system outage, data loss
- **P2 - High**: Major functionality impaired, security incident
- **P3 - Medium**: Minor functionality impaired, performance degraded
- **P4 - Low**: Cosmetic issues, feature requests

### Response Procedures

#### P1 - Critical Incidents

1. **Immediate Response** (0-15 minutes)
   ```bash
   # Check system status
   ./scripts/deploy.sh production latest health
   
   # Emergency recovery if needed
   ./scripts/backup/disaster_recovery.sh emergency
   ```

2. **Investigation** (15-60 minutes)
   ```bash
   # Check logs
   tail -f logs/errors.log logs/pii_extraction.log
   
   # Check system resources
   python3 -c "
   from src.core.health_checks import get_system_health
   health = get_system_health()
   print(health.summary)
   "
   ```

3. **Communication**
   - Notify stakeholders
   - Update status page
   - Document timeline

#### P2 - High Priority Incidents

1. **Assessment** (0-30 minutes)
   ```bash
   # Run health checks
   python3 -c "
   from src.core.health_checks import get_health_monitor
   monitor = get_health_monitor()
   report = monitor.get_health_summary()
   print(report)
   "
   ```

2. **Mitigation** (30-120 minutes)
   - Apply fixes
   - Test solutions
   - Monitor impact

### Post-Incident Activities

1. **Root Cause Analysis**
2. **Documentation**
3. **Process Improvements**
4. **Prevention Measures**

---

## Maintenance Procedures

### Routine Maintenance

#### Daily
- Check system health
- Review error logs
- Verify backup completion

#### Weekly
- Update dependencies
- Clean old logs
- Performance review

#### Monthly
- Security updates
- Capacity planning
- Disaster recovery testing

### Maintenance Scripts

```bash
# Daily health check
./scripts/maintenance/daily_check.sh

# Weekly maintenance
./scripts/maintenance/weekly_maintenance.sh

# Monthly security update
./scripts/maintenance/security_update.sh
```

### Scheduled Downtime

1. **Planning**
   - Schedule maintenance window
   - Notify users
   - Prepare rollback plan

2. **Execution**
   ```bash
   # Create maintenance backup
   ./scripts/backup/automated_backup.sh full 7
   
   # Stop services
   docker-compose down
   
   # Perform maintenance
   # ...
   
   # Start services
   docker-compose up -d
   
   # Verify functionality
   ./scripts/deploy.sh production latest health
   ```

3. **Verification**
   - Test critical functions
   - Monitor performance
   - Confirm user access

---

## Contact Information

### On-Call Rotation
- **Primary**: DevOps Engineer
- **Secondary**: Lead Developer
- **Escalation**: Technical Lead

### Emergency Contacts
- **DevOps Engineer**: devops@company.com
- **Security Team**: security@company.com
- **Management**: management@company.com

### External Vendors
- **AWS Support**: support.aws.com
- **HuggingFace Support**: support@huggingface.co

---

## Quick Reference

### Essential Commands

```bash
# Health check
python3 -c "from src.core.health_checks import get_system_health; print(get_system_health().status.value)"

# Backup
./scripts/backup/automated_backup.sh full 30

# Recovery
./scripts/backup/disaster_recovery.sh emergency

# Performance check
python3 -c "from src.core.performance_monitor import get_performance_monitor; print(get_performance_monitor().get_performance_summary())"

# Environment validation
python3 -c "from src.core.environment_manager import get_env_manager; print(get_env_manager().validate_current_environment())"
```

### Log Locations
- Application logs: `logs/pii_extraction.log`
- Error logs: `logs/errors.log`
- Audit logs: `logs/audit.log`
- Docker logs: `docker-compose logs`

### Configuration Files
- Environment: `.env.{environment}`
- Docker: `docker-compose.yml`
- Nginx: `nginx/nginx.conf`
- Monitoring: `monitoring/prometheus.yml`

This runbook should be reviewed and updated regularly as the system evolves and new procedures are developed.