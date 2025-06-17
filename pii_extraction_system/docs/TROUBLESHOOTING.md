# PII Extraction System - Troubleshooting Guide

## Table of Contents

1. [Quick Diagnostic Tools](#quick-diagnostic-tools)
2. [Common Issues and Solutions](#common-issues-and-solutions)
3. [Error Code Reference](#error-code-reference)
4. [Performance Troubleshooting](#performance-troubleshooting)
5. [Configuration Issues](#configuration-issues)
6. [Integration Problems](#integration-problems)
7. [Security and Privacy Issues](#security-and-privacy-issues)
8. [Dashboard Issues](#dashboard-issues)
9. [Logging and Monitoring](#logging-and-monitoring)
10. [Advanced Debugging](#advanced-debugging)

---

## Quick Diagnostic Tools

### System Health Check

Run this script to quickly assess system health:

```python
# quick_health_check.py
import sys
sys.path.append('.')

from core.pipeline import PIIExtractionPipeline
from extractors.rule_based import RuleBasedExtractor
from utils.document_processor import DocumentProcessor
import requests
import os

def run_health_check():
    print("🔍 PII Extraction System Health Check")
    print("=" * 50)
    
    issues = []
    
    # 1. Check Python environment
    print("✅ Checking Python environment...")
    try:
        import transformers, torch, streamlit
        print("   - Core dependencies: OK")
    except ImportError as e:
        issues.append(f"Missing dependency: {e}")
        print(f"   - Missing dependency: {e}")
    
    # 2. Check core imports
    print("✅ Checking core imports...")
    try:
        pipeline = PIIExtractionPipeline()
        print("   - Pipeline initialization: OK")
    except Exception as e:
        issues.append(f"Pipeline initialization failed: {e}")
        print(f"   - Pipeline failed: {e}")
    
    # 3. Check extractors
    print("✅ Checking extractors...")
    try:
        extractor = RuleBasedExtractor()
        test_result = extractor.extract_pii("Test email: test@example.com")
        if len(test_result.pii_entities) > 0:
            print("   - Rule-based extractor: OK")
        else:
            issues.append("Rule-based extractor not finding PII")
    except Exception as e:
        issues.append(f"Extractor error: {e}")
        print(f"   - Extractor error: {e}")
    
    # 4. Check document processor
    print("✅ Checking document processor...")
    try:
        processor = DocumentProcessor()
        print("   - Document processor: OK")
    except Exception as e:
        issues.append(f"Document processor error: {e}")
        print(f"   - Document processor error: {e}")
    
    # 5. Check data directories
    print("✅ Checking data directories...")
    required_dirs = ['data', 'logs', 'config']
    for dir_name in required_dirs:
        if os.path.exists(f"../{dir_name}"):
            print(f"   - {dir_name}/: OK")
        else:
            issues.append(f"Missing directory: {dir_name}")
            print(f"   - {dir_name}/: MISSING")
    
    # 6. Check dashboard (if running)
    print("✅ Checking dashboard availability...")
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("   - Dashboard: RUNNING")
        else:
            print("   - Dashboard: NOT ACCESSIBLE")
    except:
        print("   - Dashboard: NOT RUNNING")
    
    # Summary
    print("\n" + "=" * 50)
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        print(f"\n🔧 Please resolve {len(issues)} issue(s) before using the system.")
        return False
    else:
        print("✅ ALL CHECKS PASSED - System is healthy!")
        return True

if __name__ == "__main__":
    success = run_health_check()
    sys.exit(0 if success else 1)
```

### Log Analysis Tool

```bash
#!/bin/bash
# analyze_logs.sh

LOG_DIR="../logs"
echo "📊 Analyzing PII Extraction System Logs"
echo "======================================="

# Check if logs directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Logs directory not found at $LOG_DIR"
    exit 1
fi

# Recent errors
echo "🚨 Recent Errors (last 24 hours):"
find $LOG_DIR -name "*.log" -mtime -1 -exec grep -l "ERROR\|CRITICAL" {} \; | while read file; do
    echo "  📁 $file:"
    grep "ERROR\|CRITICAL" "$file" | tail -5 | sed 's/^/    /'
    echo
done

# Performance issues
echo "⚡ Performance Issues:"
find $LOG_DIR -name "*.log" -mtime -1 -exec grep -l "slow\|timeout\|performance" {} \; | while read file; do
    echo "  📁 $file:"
    grep "slow\|timeout\|performance" "$file" | tail -3 | sed 's/^/    /'
    echo
done

# Memory usage
echo "💾 Memory Warnings:"
find $LOG_DIR -name "*.log" -mtime -1 -exec grep -l "memory\|oom" {} \; | while read file; do
    echo "  📁 $file:"
    grep "memory\|oom" "$file" | tail -3 | sed 's/^/    /'
    echo
done

echo "✅ Log analysis complete"
```

---

## Common Issues and Solutions

### 1. Installation and Setup Issues

#### Problem: "ModuleNotFoundError: No module named 'core'"

**Symptoms:**
- Import errors when running the system
- Python can't find the core modules

**Diagnosis:**
```bash
# Check current directory
pwd
# Should be in: pii_extraction_system/src/

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

**Solutions:**
```bash
# Solution 1: Run from correct directory
cd pii_extraction_system/src
python your_script.py

# Solution 2: Add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Solution 3: Install as package
pip install -e .

# Solution 4: Use relative imports in scripts
import sys
sys.path.append('.')
```

#### Problem: "Failed to download model"

**Symptoms:**
- Hugging Face model downloads fail
- Network timeouts during initialization

**Diagnosis:**
```bash
# Test internet connectivity
curl -I https://huggingface.co

# Check available disk space
df -h

# Test manual model download
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')"
```

**Solutions:**
```bash
# Solution 1: Download models manually
mkdir -p models/cache
cd models/cache
git lfs clone https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english

# Solution 2: Use different model
# Edit config to use a smaller model:
# ner_model: "dslim/bert-base-NER"

# Solution 3: Increase timeout
export HF_HUB_DOWNLOAD_TIMEOUT=300

# Solution 4: Use local models
# Set model path to local directory in config
```

### 2. Processing Issues

#### Problem: "OCR not working" / "Tesseract not found"

**Symptoms:**
- Image processing fails
- Error: "tesseract is not installed"

**Diagnosis:**
```bash
# Check if Tesseract is installed
tesseract --version

# Check if Python wrapper is installed
python -c "import pytesseract; print('OK')"
```

**Solutions:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra

# macOS
brew install tesseract

# Windows (use installer from GitHub releases)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki

# Install Python wrapper
pip install pytesseract

# Configure path if needed (Windows)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### Problem: "Poor PII extraction quality"

**Symptoms:**
- Many false positives
- Missing obvious PII
- Low confidence scores

**Diagnosis:**
```python
# Test with known PII examples
from extractors.rule_based import RuleBasedExtractor

extractor = RuleBasedExtractor()
test_text = "Email: john.doe@company.com, Phone: (555) 123-4567"
result = extractor.extract_pii(test_text)

print(f"Found {len(result.pii_entities)} entities:")
for entity in result.pii_entities:
    print(f"  - {entity.pii_type}: {entity.text} (confidence: {entity.confidence})")
```

**Solutions:**
```yaml
# Solution 1: Adjust confidence thresholds
# In config file:
extractors:
  rule_based:
    confidence_threshold: 0.6  # Lower for more results
  ner:
    confidence_threshold: 0.7

# Solution 2: Enable more extractors
ml_models:
  enabled_models: ["rule_based", "ner", "dictionary"]

# Solution 3: Add custom patterns
extractors:
  rule_based:
    custom_patterns:
      company_email:
        pattern: "\\b[A-Za-z0-9._%+-]+@company\\.com\\b"
        confidence: 0.9
```

### 3. Performance Issues

#### Problem: "Slow processing speed"

**Symptoms:**
- Long processing times
- High CPU/memory usage
- Timeouts

**Diagnosis:**
```python
# Profile processing time
import time
from core.pipeline import PIIExtractionPipeline

pipeline = PIIExtractionPipeline()
start_time = time.time()

# Test with sample document
result = pipeline.extract_from_file("test_document.txt")

processing_time = time.time() - start_time
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Entities found: {len(result.pii_entities)}")
```

**Solutions:**
```yaml
# Solution 1: Optimize configuration
processing:
  max_concurrent_jobs: 4  # Adjust based on CPU cores
  chunk_size: 500        # Smaller chunks for large documents
  timeout_seconds: 300   # Increase timeout

# Solution 2: Use GPU acceleration
ml_models:
  device: "cuda"  # If GPU available
  batch_size: 32  # Larger batches for GPU

# Solution 3: Disable heavy extractors for speed
ml_models:
  enabled_models: ["rule_based"]  # Fastest option
```

```bash
# Solution 4: System optimization
# Increase available memory
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Use SSD storage
# Move models to SSD if possible
mv models/cache /fast/ssd/path/models/cache
ln -s /fast/ssd/path/models/cache models/cache
```

### 4. Memory Issues

#### Problem: "Out of memory" / "CUDA out of memory"

**Symptoms:**
- Process killed by system
- CUDA OOM errors
- Gradual memory increase

**Diagnosis:**
```bash
# Monitor memory usage
top -p $(pgrep -f python)

# Check GPU memory (if applicable)
nvidia-smi

# Python memory profiling
pip install memory-profiler
python -m memory_profiler your_script.py
```

**Solutions:**
```python
# Solution 1: Reduce batch size
# In config:
ml_models:
  batch_size: 8  # Reduce from default 16

# Solution 2: Process in chunks
def process_large_text(text, chunk_size=1000):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    all_entities = []
    
    for chunk in chunks:
        result = pipeline.extract_from_text(chunk)
        all_entities.extend(result.pii_entities)
    
    return all_entities

# Solution 3: Clear model cache periodically
import torch
torch.cuda.empty_cache()  # For GPU
```

---

## Error Code Reference

### System Errors (E001-E099)

| Code | Error | Description | Solution |
|------|-------|-------------|----------|
| E001 | ConfigurationError | Invalid configuration file | Check YAML syntax and required fields |
| E002 | DependencyError | Missing required dependency | Install missing packages |
| E003 | InitializationError | Component initialization failed | Check logs for specific component error |
| E004 | ResourceError | Insufficient system resources | Increase memory/disk space |

### Processing Errors (E100-E199)

| Code | Error | Description | Solution |
|------|-------|-------------|----------|
| E101 | DocumentNotFound | File does not exist | Verify file path and permissions |
| E102 | UnsupportedFormat | File format not supported | Use PDF, DOCX, or image files |
| E103 | ProcessingTimeout | Document processing timed out | Increase timeout or reduce document size |
| E104 | OCRError | OCR processing failed | Check Tesseract installation |
| E105 | ExtractionError | PII extraction failed | Check extractor configuration |

### Model Errors (E200-E299)

| Code | Error | Description | Solution |
|------|-------|-------------|----------|
| E201 | ModelLoadError | Failed to load ML model | Check model path and internet connection |
| E202 | ModelNotFound | Model file not found | Download model or check path |
| E203 | GPUError | GPU processing error | Use CPU or check CUDA installation |
| E204 | ModelCorrupted | Model file corrupted | Re-download model |

### Storage Errors (E300-E399)

| Code | Error | Description | Solution |
|------|-------|-------------|----------|
| E301 | StorageError | Storage operation failed | Check permissions and disk space |
| E302 | S3Error | AWS S3 operation failed | Check credentials and bucket access |
| E303 | DatabaseError | Database operation failed | Check database connection |
| E304 | PermissionError | Insufficient permissions | Check file/directory permissions |

---

## Performance Troubleshooting

### CPU Performance

```bash
# Monitor CPU usage
htop

# Check CPU-bound processes
ps aux --sort=-%cpu | head -10

# Optimize CPU usage
# Set CPU affinity for better performance
taskset -c 0-3 python your_script.py
```

### Memory Performance

```python
# Memory profiling
import tracemalloc

tracemalloc.start()

# Your code here
result = pipeline.extract_from_file("large_document.pdf")

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

### Disk I/O Performance

```bash
# Monitor disk I/O
iotop

# Check disk usage
df -h
du -sh data/

# Optimize disk performance
# Use tmpfs for temporary files
sudo mount -t tmpfs -o size=2G tmpfs /tmp/pii_temp
```

### Network Performance

```bash
# Test model download speed
curl -w "@curl-format.txt" -o /dev/null -s "https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english/resolve/main/config.json"

# Check S3 transfer speed
aws s3 cp test_file.txt s3://your-bucket/test_file.txt --cli-read-timeout 0
```

---

## Configuration Issues

### YAML Syntax Errors

```python
# Validate YAML configuration
import yaml

def validate_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration file is valid")
        return config
    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error: {e}")
        return None
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        return None

# Usage
config = validate_config("config/production.yaml")
```

### Environment Variables

```bash
# Check required environment variables
#!/bin/bash
required_vars=("AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "PII_S3_BUCKET")

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Missing environment variable: $var"
    else
        echo "✅ $var is set"
    fi
done
```

---

## Integration Problems

### Dashboard Integration

```python
# Test dashboard backend connection
import requests
import json

def test_dashboard_api():
    try:
        # Test if pipeline is accessible from dashboard
        from core.pipeline import PIIExtractionPipeline
        pipeline = PIIExtractionPipeline()
        
        # Test processing
        test_text = "Email: test@example.com"
        result = pipeline.extract_from_file("test.txt")
        
        print("✅ Dashboard can access backend")
        return True
    except Exception as e:
        print(f"❌ Dashboard integration error: {e}")
        return False

test_dashboard_api()
```

### Database Integration

```python
# Test database connectivity
def test_database_connection():
    try:
        from core.config import settings
        # Add database connection test based on your implementation
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False
```

---

## Security and Privacy Issues

### Audit Log Issues

```python
# Verify audit logging
import os
from datetime import datetime

def check_audit_logs():
    audit_dir = "../logs/audit"
    
    if not os.path.exists(audit_dir):
        print(f"❌ Audit log directory not found: {audit_dir}")
        return False
    
    # Check recent audit entries
    log_files = [f for f in os.listdir(audit_dir) if f.endswith('.log')]
    
    if not log_files:
        print("❌ No audit log files found")
        return False
    
    latest_log = max(log_files, key=lambda f: os.path.getctime(os.path.join(audit_dir, f)))
    
    with open(os.path.join(audit_dir, latest_log), 'r') as f:
        lines = f.readlines()
    
    recent_entries = [line for line in lines if datetime.now().strftime('%Y-%m-%d') in line]
    
    print(f"✅ Found {len(recent_entries)} recent audit entries")
    return True

check_audit_logs()
```

### Privacy Compliance

```python
# Check privacy settings
def verify_privacy_compliance():
    from core.config import settings
    
    checks = {
        "Audit logging enabled": settings.privacy.enable_audit_logging,
        "GDPR mode enabled": settings.privacy.enable_gdpr_mode,
        "Data retention configured": settings.privacy.data_retention_days > 0,
        "Default redaction method set": bool(settings.privacy.default_redaction_method)
    }
    
    for check, status in checks.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {check}: {status}")
    
    return all(checks.values())

verify_privacy_compliance()
```

---

## Dashboard Issues

### Streamlit Common Problems

```python
# dashboard_debug.py
import streamlit as st
import sys
import os

# Add debug information to dashboard
def add_debug_info():
    st.sidebar.markdown("## Debug Information")
    
    # Python path
    st.sidebar.text("Python Path:")
    for path in sys.path[:5]:  # Show first 5 paths
        st.sidebar.text(f"  {path}")
    
    # Current directory
    st.sidebar.text(f"Current Dir: {os.getcwd()}")
    
    # Environment variables
    st.sidebar.text("Key Environment Variables:")
    env_vars = ["PYTHONPATH", "PII_CONFIG_PATH", "AWS_DEFAULT_REGION"]
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        st.sidebar.text(f"  {var}: {value}")

# Add to your dashboard main page
if st.sidebar.checkbox("Show Debug Info"):
    add_debug_info()
```

### Upload Issues

```python
# Test file upload functionality
def test_file_upload():
    import tempfile
    import os
    
    # Create test file
    test_content = "Test document with PII: john.doe@example.com"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file_path = f.name
    
    try:
        # Test processing
        from core.pipeline import PIIExtractionPipeline
        pipeline = PIIExtractionPipeline()
        result = pipeline.extract_from_file(test_file_path)
        
        print(f"✅ File upload test successful: {len(result.pii_entities)} entities found")
        return True
    except Exception as e:
        print(f"❌ File upload test failed: {e}")
        return False
    finally:
        # Clean up
        os.unlink(test_file_path)

test_file_upload()
```

---

## Logging and Monitoring

### Log Level Configuration

```python
# Adjust logging levels for debugging
import logging

def set_debug_logging():
    # Set root logger to DEBUG
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Set specific loggers
    logging.getLogger("core.pipeline").setLevel(logging.DEBUG)
    logging.getLogger("extractors").setLevel(logging.DEBUG)
    logging.getLogger("utils").setLevel(logging.DEBUG)
    
    # Add console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add to root logger
    logging.getLogger().addHandler(console_handler)
    
    print("✅ Debug logging enabled")

# Use when debugging
set_debug_logging()
```

### Custom Monitoring

```python
# Custom monitoring script
import time
import psutil
import logging

class SystemMonitor:
    def __init__(self, log_file="system_monitor.log"):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def monitor_system(self, duration=300, interval=30):
        """Monitor system for specified duration."""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            self.logger.info(f"CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%")
            
            # Alert on high usage
            if cpu_percent > 90:
                self.logger.warning(f"High CPU usage: {cpu_percent}%")
            if memory_percent > 90:
                self.logger.warning(f"High memory usage: {memory_percent}%")
            if disk_percent > 90:
                self.logger.warning(f"High disk usage: {disk_percent}%")
            
            time.sleep(interval)
    
    def check_processes(self):
        """Check for PII extraction processes."""
        pii_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'pii' in cmdline.lower() or 'streamlit' in cmdline.lower():
                        pii_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cmdline': cmdline
                        })
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        
        self.logger.info(f"Found {len(pii_processes)} PII-related processes")
        for proc in pii_processes:
            self.logger.info(f"  PID {proc['pid']}: {proc['cmdline']}")
        
        return pii_processes

# Usage
monitor = SystemMonitor()
monitor.check_processes()
```

---

## Advanced Debugging

### Memory Leak Detection

```python
# Memory leak detection
import gc
import tracemalloc
from pympler import tracker

def detect_memory_leaks():
    # Start memory tracking
    tr = tracker.SummaryTracker()
    tracemalloc.start()
    
    # Your code here - repeat the operation multiple times
    from core.pipeline import PIIExtractionPipeline
    
    for i in range(5):
        pipeline = PIIExtractionPipeline()
        # Simulate processing
        test_text = "Test email: test@example.com"
        with open('temp_test.txt', 'w') as f:
            f.write(test_text)
        result = pipeline.extract_from_file('temp_test.txt')
        
        # Force garbage collection
        del pipeline
        gc.collect()
        
        # Check memory
        current, peak = tracemalloc.get_traced_memory()
        print(f"Iteration {i+1}: Current={current/1024/1024:.1f}MB, Peak={peak/1024/1024:.1f}MB")
    
    # Show memory diff
    tr.print_diff()
    
    # Cleanup
    import os
    os.remove('temp_test.txt')
    tracemalloc.stop()

# Run memory leak detection
detect_memory_leaks()
```

### Performance Profiling

```python
# Performance profiling
import cProfile
import pstats
from io import StringIO

def profile_extraction():
    pr = cProfile.Profile()
    pr.enable()
    
    # Your code to profile
    from core.pipeline import PIIExtractionPipeline
    pipeline = PIIExtractionPipeline()
    
    # Create test document
    test_content = """
    Dear John Smith,
    Contact us at support@company.com or call (555) 123-4567.
    Employee ID: EMP-2024-001
    SSN: 123-45-6789
    """
    
    with open('profile_test.txt', 'w') as f:
        f.write(test_content)
    
    result = pipeline.extract_from_file('profile_test.txt')
    
    pr.disable()
    
    # Analyze results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("Top 20 functions by cumulative time:")
    print(s.getvalue())
    
    # Cleanup
    import os
    os.remove('profile_test.txt')

# Run profiling
profile_extraction()
```

### Network Debugging

```python
# Network connectivity testing
import requests
import socket
import time

def test_network_connectivity():
    """Test various network connections."""
    
    tests = [
        ("Hugging Face", "https://huggingface.co", 5),
        ("AWS S3", "https://s3.amazonaws.com", 5),
        ("PyPI", "https://pypi.org", 5)
    ]
    
    for name, url, timeout in tests:
        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                print(f"✅ {name}: OK ({duration:.2f}s)")
            else:
                print(f"⚠️ {name}: HTTP {response.status_code} ({duration:.2f}s)")
        except requests.exceptions.Timeout:
            print(f"❌ {name}: TIMEOUT")
        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: CONNECTION ERROR")
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")

def test_dns_resolution():
    """Test DNS resolution for key domains."""
    domains = ["huggingface.co", "s3.amazonaws.com", "pypi.org"]
    
    for domain in domains:
        try:
            ip = socket.gethostbyname(domain)
            print(f"✅ {domain} -> {ip}")
        except socket.gaierror as e:
            print(f"❌ {domain}: DNS ERROR - {e}")

# Run network tests
test_network_connectivity()
test_dns_resolution()
```

---

## Getting Additional Help

### Collecting System Information

```bash
#!/bin/bash
# collect_debug_info.sh

echo "PII Extraction System Debug Information"
echo "======================================="
date
echo

echo "System Information:"
uname -a
echo

echo "Python Information:"
python --version
pip --version
echo

echo "Disk Space:"
df -h
echo

echo "Memory Information:"
free -h
echo

echo "Environment Variables:"
env | grep -E "(PII_|AWS_|PYTHON)" | sort
echo

echo "Python Packages:"
pip list | grep -E "(torch|transformers|streamlit|pandas)"
echo

echo "Running Processes:"
ps aux | grep -E "(python|streamlit)" | grep -v grep
echo

echo "Network Connectivity:"
curl -I https://huggingface.co 2>/dev/null | head -1
curl -I https://pypi.org 2>/dev/null | head -1
echo

echo "Log File Sizes:"
find ../logs -name "*.log" -exec du -h {} \; 2>/dev/null | sort -h
echo

echo "Recent Errors:"
find ../logs -name "*.log" -mtime -1 -exec grep -l "ERROR" {} \; | head -3 | while read file; do
    echo "=== $file ==="
    tail -5 "$file"
    echo
done
```

### Creating Support Tickets

When reporting issues, include:

1. **System Information**:
   - Operating system and version
   - Python version
   - Hardware specifications (CPU, RAM, GPU)

2. **Environment Details**:
   - Installation method (pip, Docker, etc.)
   - Configuration files (sanitized)
   - Environment variables (non-sensitive)

3. **Error Information**:
   - Complete error messages
   - Stack traces
   - Relevant log files
   - Steps to reproduce

4. **Context**:
   - What you were trying to do
   - What you expected to happen
   - What actually happened
   - When the issue started occurring

### Community Resources

- **Documentation**: Check the API documentation and user guide
- **GitHub Issues**: Search existing issues and create new ones
- **Error Logs**: Always include relevant log files
- **Minimal Examples**: Provide minimal code that reproduces the issue

This troubleshooting guide should help you diagnose and resolve most common issues with the PII Extraction System. For persistent problems, use the diagnostic tools and information collection scripts to gather details before seeking additional support.