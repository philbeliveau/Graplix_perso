# PII Extraction System - Deployment Guide

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Local Development Deployment](#local-development-deployment)
3. [Docker Deployment](#docker-deployment)
4. [AWS Production Deployment](#aws-production-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Environment Configuration](#environment-configuration)
7. [Security Configuration](#security-configuration)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Backup and Recovery](#backup-and-recovery)
10. [Troubleshooting](#troubleshooting)

---

## Pre-Deployment Checklist

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 20 GB free space
- **Network**: Stable internet connection for model downloads
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), or Windows 10+

#### Recommended Production Requirements
- **CPU**: 8+ cores, 3.0 GHz
- **RAM**: 32 GB
- **Storage**: 100 GB SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for ML acceleration)
- **Network**: High-bandwidth connection for cloud storage

### Dependencies Verification

```bash
# Check Python version (3.8+ required)
python --version

# Check pip and virtual environment tools
pip --version
python -m venv --help

# Check Docker (if using containerized deployment)
docker --version
docker-compose --version

# Check available disk space
df -h

# Check memory
free -h
```

### Pre-Deployment Testing

```bash
# Clone repository
git clone <repository-url>
cd pii_extraction_system

# Run system validation
cd src
python ../validation_test.py
```

Expected output should show all components passing validation tests.

---

## Local Development Deployment

### Step 1: Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or using Poetry (recommended)
pip install poetry
poetry install
poetry shell
```

### Step 2: Configuration

```bash
# Copy default configuration
cp config/default.yaml config/development.yaml

# Edit configuration as needed
nano config/development.yaml
```

### Step 3: Initialize Data Directories

```bash
# Create necessary directories
mkdir -p data/documents/{original,processed}
mkdir -p data/models/cache
mkdir -p logs/{audit,performance,errors}
mkdir -p temp/uploads
```

### Step 4: Start Services

```bash
# Start the dashboard (development mode)
cd src/dashboard
streamlit run main.py --server.port 8501 --server.address 0.0.0.0

# In another terminal, test the API
cd src
python -c "
from core.pipeline import PIIExtractionPipeline
pipeline = PIIExtractionPipeline()
print('✅ Pipeline ready for development')
"
```

### Step 5: Verify Installation

- Open browser to `http://localhost:8501`
- Test document upload and processing
- Check logs in `logs/` directory

---

## Docker Deployment

### Step 1: Build Docker Images

```bash
# Build main application image
docker build -t pii-extraction:latest .

# Build with specific tag
docker build -t pii-extraction:v1.0.0 .

# Build development image (with debugging tools)
docker build -f Dockerfile.dev -t pii-extraction:dev .
```

### Step 2: Docker Compose Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale api=3

# Stop services
docker-compose down
```

### Docker Compose Configuration (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  api:
    build: .
    image: pii-extraction:latest
    ports:
      - "8000:8000"
    environment:
      - PII_CONFIG_PATH=/app/config/production.yaml
      - PII_LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    depends_on:
      - redis
      - postgres
    restart: unless-stopped

  dashboard:
    build: .
    image: pii-extraction:latest
    command: streamlit run src/dashboard/main.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - PII_CONFIG_PATH=/app/config/production.yaml
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    depends_on:
      - api
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: pii_extraction
      POSTGRES_USER: pii_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api
      - dashboard
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
```

### Step 3: Environment Variables

```bash
# Create .env file
cat > .env << EOF
# Database
DB_PASSWORD=secure_password_here

# AWS Configuration (if using S3)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-west-2

# Application Configuration
PII_S3_BUCKET=your-pii-bucket
PII_LOG_LEVEL=INFO
PII_MAX_WORKERS=4

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
EOF

# Set proper permissions
chmod 600 .env
```

---

## AWS Production Deployment

### Architecture Overview

```
Internet Gateway
    ↓
Application Load Balancer
    ↓
ECS Fargate Cluster
├── API Service (Auto Scaling)
├── Dashboard Service
└── Worker Service (Batch Processing)
    ↓
├── RDS PostgreSQL (Multi-AZ)
├── ElastiCache Redis
├── S3 Buckets (Documents & Models)
└── CloudWatch (Monitoring)
```

### Step 1: Infrastructure as Code (Terraform)

```hcl
# main.tf
provider "aws" {
  region = var.aws_region
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "pii-extraction-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  
  tags = var.common_tags
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "pii-extraction-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = var.common_tags
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "pii-extraction-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
  
  tags = var.common_tags
}

# RDS PostgreSQL
resource "aws_db_instance" "main" {
  identifier     = "pii-extraction-db"
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = "db.t3.medium"
  
  allocated_storage     = 100
  max_allocated_storage = 500
  storage_type          = "gp2"
  storage_encrypted     = true
  
  db_name  = "pii_extraction"
  username = "pii_user"
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "pii-extraction-final-snapshot"
  
  tags = var.common_tags
}

# S3 Buckets
resource "aws_s3_bucket" "documents" {
  bucket = "${var.project_name}-documents-${random_id.bucket_suffix.hex}"
  
  tags = var.common_tags
}

resource "aws_s3_bucket_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ECS Task Definition
resource "aws_ecs_task_definition" "api" {
  family                   = "pii-extraction-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = 2048
  memory                   = 4096
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn
  
  container_definitions = jsonencode([
    {
      name  = "api"
      image = "${aws_ecr_repository.main.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "PII_CONFIG_PATH"
          value = "/app/config/production.yaml"
        },
        {
          name  = "PII_S3_BUCKET"
          value = aws_s3_bucket.documents.bucket
        },
        {
          name  = "DATABASE_URL"
          value = "postgresql://${aws_db_instance.main.username}:${var.db_password}@${aws_db_instance.main.endpoint}/${aws_db_instance.main.db_name}"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.main.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "api"
        }
      }
      
      essential = true
    }
  ])
  
  tags = var.common_tags
}
```

### Step 2: Deploy with Terraform

```bash
# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var-file="production.tfvars"

# Apply deployment
terraform apply -var-file="production.tfvars"

# Get outputs
terraform output
```

### Step 3: ECS Service Configuration

```bash
# Build and push Docker image to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

docker build -t pii-extraction .
docker tag pii-extraction:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/pii-extraction:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/pii-extraction:latest

# Create ECS service
aws ecs create-service \
  --cluster pii-extraction-cluster \
  --service-name pii-extraction-api \
  --task-definition pii-extraction-api:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-12345],assignPublicIp=DISABLED}"
```

---

## Kubernetes Deployment

### Step 1: Prepare Kubernetes Manifests

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: pii-extraction

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pii-config
  namespace: pii-extraction
data:
  production.yaml: |
    data_source:
      source_type: s3
      s3_bucket: ${S3_BUCKET}
    
    ml_models:
      enabled_models: ["rule_based", "ner", "layout_aware"]
    
    logging:
      level: INFO
      audit_enabled: true

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: pii-secrets
  namespace: pii-extraction
type: Opaque
data:
  db-password: <base64-encoded-password>
  aws-access-key: <base64-encoded-key>
  aws-secret-key: <base64-encoded-secret>

---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pii-extraction-api
  namespace: pii-extraction
  labels:
    app: pii-extraction-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pii-extraction-api
  template:
    metadata:
      labels:
        app: pii-extraction-api
    spec:
      containers:
      - name: api
        image: pii-extraction:latest
        ports:
        - containerPort: 8000
        env:
        - name: PII_CONFIG_PATH
          value: "/app/config/production.yaml"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: pii-secrets
              key: db-password
        - name: AWS_ACCESS_KEY_ID
          valueFrom:
            secretKeyRef:
              name: pii-secrets
              key: aws-access-key
        - name: AWS_SECRET_ACCESS_KEY
          valueFrom:
            secretKeyRef:
              name: pii-secrets
              key: aws-secret-key
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: models
          mountPath: /app/models
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: pii-config
      - name: models
        emptyDir: {}

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: pii-extraction-api-service
  namespace: pii-extraction
spec:
  selector:
    app: pii-extraction-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pii-extraction-ingress
  namespace: pii-extraction
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - pii.yourdomain.com
    secretName: pii-extraction-tls
  rules:
  - host: pii.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pii-extraction-api-service
            port:
              number: 80
```

### Step 2: Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n pii-extraction
kubectl get services -n pii-extraction
kubectl get ingress -n pii-extraction

# View logs
kubectl logs -f deployment/pii-extraction-api -n pii-extraction

# Scale deployment
kubectl scale deployment pii-extraction-api --replicas=5 -n pii-extraction
```

---

## Environment Configuration

### Production Configuration (`config/production.yaml`)

```yaml
# Data Source Configuration
data_source:
  source_type: "s3"
  s3_bucket: "${PII_S3_BUCKET}"
  s3_region: "${AWS_DEFAULT_REGION}"
  local_path: "./data"

# ML Models Configuration
ml_models:
  enabled_models: ["rule_based", "ner", "layout_aware"]
  model_cache_dir: "./models/cache"
  ner_model: "dbmdz/bert-large-cased-finetuned-conll03-english"
  device: "cuda"  # or "cpu"
  max_length: 512
  batch_size: 16

# Processing Configuration
processing:
  max_concurrent_jobs: 8
  chunk_size: 1000
  timeout_seconds: 600
  enable_parallel: true

# Privacy and Compliance
privacy:
  enable_audit_logging: true
  audit_log_path: "/app/logs/audit/audit.log"
  enable_gdpr_mode: true
  enable_law25_mode: true
  default_redaction_method: "mask"
  data_retention_days: 90

# Storage Configuration
storage:
  use_s3: true
  s3_bucket: "${PII_S3_BUCKET}"
  s3_region: "${AWS_DEFAULT_REGION}"
  local_fallback: true
  encryption_enabled: true

# Database Configuration
database:
  url: "${DATABASE_URL}"
  pool_size: 20
  max_overflow: 30
  echo: false

# Security Configuration
security:
  secret_key: "${SECRET_KEY}"
  jwt_secret: "${JWT_SECRET}"
  session_timeout: 3600
  max_login_attempts: 5
  password_min_length: 12

# Logging Configuration
logging:
  level: "INFO"
  format: "json"
  log_file: "/app/logs/pii_extraction.log"
  max_file_size: "100MB"
  backup_count: 10
  audit_enabled: true
  performance_tracking: true

# Monitoring Configuration
monitoring:
  enable_metrics: true
  metrics_port: 9090
  health_check_port: 8080
  alert_on_errors: true
  performance_threshold_ms: 5000

# Dashboard Configuration
dashboard:
  port: 8501
  host: "0.0.0.0"
  max_upload_size: "200MB"
  session_timeout: 1800
  enable_authentication: true
```

---

## Security Configuration

### SSL/TLS Configuration

```bash
# Generate self-signed certificates (development only)
openssl req -x509 -newkey rsa:4096 -keyout private.key -out certificate.crt -days 365 -nodes

# For production, use Let's Encrypt or your certificate authority
certbot certonly --webroot -w /var/www/html -d yourdomain.com
```

### Nginx SSL Configuration

```nginx
# nginx/nginx.conf
server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/certificate.crt;
    ssl_certificate_key /etc/nginx/ssl/private.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_session_cache shared:SSL:1m;
    ssl_session_timeout 5m;
    
    location / {
        proxy_pass http://dashboard:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    location /api/ {
        proxy_pass http://api:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### IAM Roles and Policies (AWS)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-pii-bucket",
        "arn:aws:s3:::your-pii-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:Encrypt",
        "kms:GenerateDataKey"
      ],
      "Resource": "arn:aws:kms:region:account:key/key-id"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:region:account:*"
    }
  ]
}
```

---

## Monitoring and Logging

### CloudWatch Configuration (AWS)

```yaml
# cloudwatch-config.json
{
  "agent": {
    "metrics_collection_interval": 60,
    "run_as_user": "cwagent"
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/app/logs/pii_extraction.log",
            "log_group_name": "/aws/ecs/pii-extraction/app",
            "log_stream_name": "{instance_id}/app"
          },
          {
            "file_path": "/app/logs/audit/audit.log",
            "log_group_name": "/aws/ecs/pii-extraction/audit",
            "log_stream_name": "{instance_id}/audit"
          }
        ]
      }
    }
  },
  "metrics": {
    "namespace": "PII/Extraction",
    "metrics_collected": {
      "cpu": {
        "measurement": [
          "cpu_usage_idle",
          "cpu_usage_iowait",
          "cpu_usage_user",
          "cpu_usage_system"
        ],
        "metrics_collection_interval": 60
      },
      "mem": {
        "measurement": [
          "mem_used_percent"
        ],
        "metrics_collection_interval": 60
      },
      "disk": {
        "measurement": [
          "used_percent"
        ],
        "metrics_collection_interval": 60,
        "resources": [
          "*"
        ]
      }
    }
  }
}
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pii-extraction'
    static_configs:
      - targets: ['api:9090']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Alert Rules

```yaml
# alert_rules.yml
groups:
- name: pii-extraction
  rules:
  - alert: HighErrorRate
    expr: rate(pii_extraction_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate in PII extraction"
      description: "Error rate is {{ $value }} per second"

  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 90%"

  - alert: ServiceDown
    expr: up{job="pii-extraction"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "PII Extraction service is down"
      description: "Service has been down for more than 1 minute"
```

---

## Backup and Recovery

### Database Backup Strategy

```bash
#!/bin/bash
# backup_database.sh

DB_HOST="your-db-host"
DB_NAME="pii_extraction"
DB_USER="pii_user"
BACKUP_DIR="/backups/database"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME -F c -b -v -f "$BACKUP_DIR/pii_extraction_$DATE.backup"

# Compress backup
gzip "$BACKUP_DIR/pii_extraction_$DATE.backup"

# Upload to S3
aws s3 cp "$BACKUP_DIR/pii_extraction_$DATE.backup.gz" s3://your-backup-bucket/database/

# Clean up old local backups (keep 7 days)
find $BACKUP_DIR -name "*.backup.gz" -mtime +7 -delete

echo "Backup completed: pii_extraction_$DATE.backup.gz"
```

### S3 Data Backup

```bash
#!/bin/bash
# backup_s3_data.sh

SOURCE_BUCKET="your-pii-bucket"
BACKUP_BUCKET="your-backup-bucket"
DATE=$(date +%Y%m%d)

# Sync data to backup bucket
aws s3 sync s3://$SOURCE_BUCKET s3://$BACKUP_BUCKET/data/$DATE/ --delete

# Create lifecycle policy for old backups
aws s3api put-bucket-lifecycle-configuration \
  --bucket $BACKUP_BUCKET \
  --lifecycle-configuration file://lifecycle-policy.json

echo "S3 data backup completed for $DATE"
```

### Recovery Procedures

```bash
# Database Recovery
#!/bin/bash
# restore_database.sh

BACKUP_FILE="$1"
DB_HOST="your-db-host"
DB_NAME="pii_extraction"
DB_USER="pii_user"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Download backup from S3 if needed
if [[ $BACKUP_FILE == s3://* ]]; then
    aws s3 cp $BACKUP_FILE ./temp_backup.backup.gz
    BACKUP_FILE="./temp_backup.backup.gz"
fi

# Decompress backup
gunzip $BACKUP_FILE

# Restore database
pg_restore -h $DB_HOST -U $DB_USER -d $DB_NAME -v "${BACKUP_FILE%.gz}"

echo "Database restoration completed"
```

---

## Troubleshooting

### Common Deployment Issues

#### 1. Docker Build Failures

```bash
# Problem: Docker build fails due to dependency issues
# Solution: Clear cache and rebuild
docker system prune -f
docker build --no-cache -t pii-extraction:latest .

# Problem: Out of disk space during build
# Solution: Clean up Docker resources
docker system df
docker system prune --volumes -f
```

#### 2. Container Startup Issues

```bash
# Check container logs
docker logs pii-extraction-api

# Debug inside container
docker exec -it pii-extraction-api /bin/bash

# Check resource usage
docker stats

# Verify environment variables
docker exec pii-extraction-api env
```

#### 3. AWS Deployment Issues

```bash
# Check ECS service status
aws ecs describe-services --cluster pii-extraction-cluster --services pii-extraction-api

# View ECS task logs
aws logs get-log-events --log-group-name /aws/ecs/pii-extraction --log-stream-name <stream-name>

# Check security group rules
aws ec2 describe-security-groups --group-ids sg-xxxxxxxxx

# Verify IAM permissions
aws sts get-caller-identity
aws iam simulate-principal-policy --policy-source-arn <role-arn> --action-names s3:GetObject --resource-arns <resource-arn>
```

#### 4. Performance Issues

```bash
# Check system resources
htop
iotop
df -h

# Monitor application metrics
curl http://localhost:9090/metrics

# Check database connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

# Analyze slow queries
SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;
```

### Health Checks

```python
# health_check.py
import requests
import sys
import time

def check_api_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        return response.status_code == 200
    except:
        return False

def check_dashboard_health():
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        return response.status_code == 200
    except:
        return False

def check_database_health():
    # Add database connectivity check
    pass

if __name__ == "__main__":
    checks = {
        "API": check_api_health(),
        "Dashboard": check_dashboard_health(),
        "Database": check_database_health()
    }
    
    all_healthy = all(checks.values())
    
    for service, healthy in checks.items():
        status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
        print(f"{service}: {status}")
    
    sys.exit(0 if all_healthy else 1)
```

### Disaster Recovery Plan

1. **Data Recovery Priority**:
   - Critical: Database (user data, configurations)
   - Important: Processed documents and results
   - Medium: ML model cache
   - Low: Temporary files and logs

2. **Recovery Time Objectives**:
   - Database: RTO 1 hour, RPO 15 minutes
   - Application: RTO 30 minutes
   - Full system: RTO 2 hours

3. **Recovery Steps**:
   1. Assess the scope of the incident
   2. Restore database from latest backup
   3. Restore S3 data if affected
   4. Redeploy application services
   5. Verify system functionality
   6. Monitor for issues

This deployment guide provides comprehensive instructions for deploying the PII Extraction System in various environments, from local development to production-scale cloud deployments.