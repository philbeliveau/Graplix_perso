#!/bin/bash
# Automated backup script for PII Extraction System
# DevOps Engineer - Deployment Configuration

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BACKUP_TYPE=${1:-full}
RETENTION_DAYS=${2:-30}
S3_BUCKET=${3:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if running as root (not recommended for security)
check_user() {
    if [ "$EUID" -eq 0 ]; then
        warning "Running as root. Consider using a dedicated backup user."
    fi
}

# Check disk space
check_disk_space() {
    log "Checking available disk space..."
    
    AVAILABLE_SPACE=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    REQUIRED_SPACE=1048576  # 1GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        error "Insufficient disk space. Available: ${AVAILABLE_SPACE}KB, Required: ${REQUIRED_SPACE}KB"
        exit 1
    fi
    
    success "Disk space check passed. Available: $((AVAILABLE_SPACE / 1024))MB"
}

# Check dependencies
check_dependencies() {
    log "Checking backup dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 is not installed"
        exit 1
    fi
    
    # Check if backup manager is available
    if [ ! -f "$SCRIPT_DIR/backup_manager.py" ]; then
        error "Backup manager script not found: $SCRIPT_DIR/backup_manager.py"
        exit 1
    fi
    
    # Check AWS CLI if S3 backup is requested
    if [ -n "$S3_BUCKET" ]; then
        if ! command -v aws &> /dev/null; then
            warning "AWS CLI not installed. S3 backup may fail."
        else
            # Test AWS credentials
            if ! aws sts get-caller-identity &> /dev/null; then
                warning "AWS credentials not configured properly. S3 backup may fail."
            fi
        fi
    fi
    
    success "Dependencies check passed"
}

# Pre-backup tasks
pre_backup_tasks() {
    log "Performing pre-backup tasks..."
    
    cd "$PROJECT_DIR"
    
    # Create backup directory if it doesn't exist
    mkdir -p backups
    
    # Stop any running services (optional)
    # docker-compose down 2>/dev/null || true
    
    # Flush any pending writes
    sync
    
    success "Pre-backup tasks completed"
}

# Post-backup tasks
post_backup_tasks() {
    log "Performing post-backup tasks..."
    
    # Restart services if they were stopped
    # docker-compose up -d 2>/dev/null || true
    
    # Clean up old backups
    python3 "$SCRIPT_DIR/backup_manager.py" cleanup
    
    # Send notification (if configured)
    send_notification_if_configured
    
    success "Post-backup tasks completed"
}

# Send notification if configured
send_notification_if_configured() {
    SLACK_WEBHOOK_URL=${SLACK_WEBHOOK_URL:-""}
    EMAIL_RECIPIENT=${EMAIL_RECIPIENT:-""}
    
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        send_slack_notification
    fi
    
    if [ -n "$EMAIL_RECIPIENT" ]; then
        send_email_notification
    fi
}

# Send Slack notification
send_slack_notification() {
    log "Sending Slack notification..."
    
    HOSTNAME=$(hostname)
    TIMESTAMP=$(date +'%Y-%m-%d %H:%M:%S')
    
    PAYLOAD=$(cat <<EOF
{
    "text": "✅ PII Extraction System Backup Completed",
    "attachments": [
        {
            "color": "good",
            "fields": [
                {"title": "Server", "value": "$HOSTNAME", "short": true},
                {"title": "Backup Type", "value": "$BACKUP_TYPE", "short": true},
                {"title": "Timestamp", "value": "$TIMESTAMP", "short": true},
                {"title": "Status", "value": "Completed Successfully", "short": true}
            ]
        }
    ]
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' \
         --data "$PAYLOAD" \
         "$SLACK_WEBHOOK_URL" &> /dev/null || warning "Failed to send Slack notification"
}

# Send email notification
send_email_notification() {
    log "Sending email notification..."
    
    HOSTNAME=$(hostname)
    TIMESTAMP=$(date +'%Y-%m-%d %H:%M:%S')
    SUBJECT="PII Extraction System Backup Completed - $HOSTNAME"
    
    BODY=$(cat <<EOF
PII Extraction System Backup Report

Server: $HOSTNAME
Backup Type: $BACKUP_TYPE
Timestamp: $TIMESTAMP
Status: Completed Successfully
Retention: $RETENTION_DAYS days

This is an automated notification from the PII Extraction System backup service.
EOF
)
    
    echo "$BODY" | mail -s "$SUBJECT" "$EMAIL_RECIPIENT" 2>/dev/null || warning "Failed to send email notification"
}

# Error handler
handle_error() {
    local exit_code=$?
    error "Backup failed with exit code: $exit_code"
    
    # Send failure notification
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        send_failure_notification
    fi
    
    exit $exit_code
}

# Send failure notification
send_failure_notification() {
    HOSTNAME=$(hostname)
    TIMESTAMP=$(date +'%Y-%m-%d %H:%M:%S')
    
    PAYLOAD=$(cat <<EOF
{
    "text": "❌ PII Extraction System Backup Failed",
    "attachments": [
        {
            "color": "danger",
            "fields": [
                {"title": "Server", "value": "$HOSTNAME", "short": true},
                {"title": "Backup Type", "value": "$BACKUP_TYPE", "short": true},
                {"title": "Timestamp", "value": "$TIMESTAMP", "short": true},
                {"title": "Status", "value": "Failed", "short": true}
            ]
        }
    ]
}
EOF
)
    
    curl -X POST -H 'Content-type: application/json' \
         --data "$PAYLOAD" \
         "$SLACK_WEBHOOK_URL" &> /dev/null || true
}

# Main backup function
perform_backup() {
    log "Starting $BACKUP_TYPE backup with $RETENTION_DAYS days retention..."
    
    # Build command
    BACKUP_CMD="python3 $SCRIPT_DIR/backup_manager.py backup --type $BACKUP_TYPE --retention-days $RETENTION_DAYS"
    
    if [ -n "$S3_BUCKET" ]; then
        BACKUP_CMD="$BACKUP_CMD --s3-bucket $S3_BUCKET"
        log "S3 backup enabled to bucket: $S3_BUCKET"
    fi
    
    # Execute backup
    if $BACKUP_CMD; then
        success "$BACKUP_TYPE backup completed successfully"
        return 0
    else
        error "$BACKUP_TYPE backup failed"
        return 1
    fi
}

# Generate backup report
generate_backup_report() {
    log "Generating backup report..."
    
    REPORT_FILE="backups/backup_report_$(date +'%Y%m%d_%H%M%S').json"
    
    python3 "$SCRIPT_DIR/backup_manager.py" stats > "$REPORT_FILE"
    
    success "Backup report generated: $REPORT_FILE"
}

# Health check before backup
health_check() {
    log "Performing health check before backup..."
    
    # Check if the system is healthy enough for backup
    if [ -f "$PROJECT_DIR/src/core/health_checks.py" ]; then
        python3 -c "
from src.core.health_checks import get_system_health
health = get_system_health()
if health.status.value in ['unhealthy']:
    print('UNHEALTHY')
    exit(1)
else:
    print('HEALTHY')
    exit(0)
" 2>/dev/null || warning "Health check script not available or failed"
    fi
    
    success "Health check passed"
}

# Lock file management to prevent concurrent backups
acquire_lock() {
    LOCK_FILE="/tmp/pii_backup.lock"
    
    if [ -f "$LOCK_FILE" ]; then
        LOCK_PID=$(cat "$LOCK_FILE")
        if ps -p "$LOCK_PID" > /dev/null 2>&1; then
            error "Another backup process is already running (PID: $LOCK_PID)"
            exit 1
        else
            warning "Stale lock file found. Removing..."
            rm -f "$LOCK_FILE"
        fi
    fi
    
    echo $$ > "$LOCK_FILE"
    log "Acquired backup lock"
}

release_lock() {
    LOCK_FILE="/tmp/pii_backup.lock"
    if [ -f "$LOCK_FILE" ]; then
        rm -f "$LOCK_FILE"
        log "Released backup lock"
    fi
}

# Cleanup function
cleanup() {
    release_lock
    log "Cleanup completed"
}

# Display usage
usage() {
    echo "Usage: $0 [backup_type] [retention_days] [s3_bucket]"
    echo ""
    echo "Parameters:"
    echo "  backup_type     : database|files|models|full (default: full)"
    echo "  retention_days  : Number of days to retain backups (default: 30)"
    echo "  s3_bucket      : S3 bucket for cloud backup (optional)"
    echo ""
    echo "Environment Variables:"
    echo "  SLACK_WEBHOOK_URL : Slack webhook for notifications"
    echo "  EMAIL_RECIPIENT   : Email address for notifications"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Full backup with 30 days retention"
    echo "  $0 database 7                        # Database backup with 7 days retention"
    echo "  $0 full 30 my-backup-bucket         # Full backup with S3 upload"
}

# Main execution
main() {
    log "PII Extraction System Automated Backup Script"
    log "============================================="
    
    # Parse arguments
    if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        usage
        exit 0
    fi
    
    # Validate backup type
    case $BACKUP_TYPE in
        database|files|models|full)
            ;;
        *)
            error "Invalid backup type: $BACKUP_TYPE"
            usage
            exit 1
            ;;
    esac
    
    # Set error handler
    trap handle_error ERR
    trap cleanup EXIT
    
    # Acquire lock
    acquire_lock
    
    # Perform checks
    check_user
    check_dependencies
    check_disk_space
    health_check
    
    # Execute backup workflow
    pre_backup_tasks
    perform_backup
    post_backup_tasks
    generate_backup_report
    
    success "Backup workflow completed successfully!"
}

# Run main function
main "$@"