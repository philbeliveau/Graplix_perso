#!/bin/bash
# Disaster Recovery Script for PII Extraction System
# DevOps Engineer - Deployment Configuration

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RECOVERY_TYPE=${1:-""}
BACKUP_ID=${2:-""}
TARGET_DIR=${3:-""}

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

# Display banner
display_banner() {
    echo -e "${RED}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                     DISASTER RECOVERY SCRIPT                     ║"
    echo "║                   PII EXTRACTION SYSTEM                          ║"
    echo "║                                                                   ║"
    echo "║  ⚠️  WARNING: This script will modify system data ⚠️             ║"
    echo "║  Make sure you understand the implications before proceeding      ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Display usage
usage() {
    echo "Usage: $0 <recovery_type> [backup_id] [target_dir]"
    echo ""
    echo "Recovery Types:"
    echo "  list                    : List available backups"
    echo "  database <backup_id>    : Restore database from backup"
    echo "  files <backup_id>       : Restore files from backup"
    echo "  models <backup_id>      : Restore models from backup"
    echo "  full <backup_id>        : Full system recovery"
    echo "  emergency              : Emergency recovery with latest backups"
    echo ""
    echo "Parameters:"
    echo "  backup_id    : Specific backup ID to restore from"
    echo "  target_dir   : Target directory for file restoration (optional)"
    echo ""
    echo "Examples:"
    echo "  $0 list                                    # List all available backups"
    echo "  $0 database database_20231215_143022       # Restore specific database backup"
    echo "  $0 files files_20231215_143022 /tmp/restore # Restore files to specific directory"
    echo "  $0 emergency                               # Emergency recovery"
    echo ""
    echo "Pre-requisites:"
    echo "  - Backup files must be available locally or in S3"
    echo "  - Sufficient disk space for restoration"
    echo "  - Proper permissions to write to target locations"
}

# Confirm dangerous operations
confirm_operation() {
    local operation=$1
    
    echo -e "${YELLOW}"
    echo "You are about to perform: $operation"
    echo "This operation may overwrite existing data."
    echo -e "${NC}"
    
    read -p "Are you sure you want to continue? (yes/no): " confirmation
    
    case $confirmation in
        yes|YES|y|Y)
            log "Operation confirmed by user"
            ;;
        *)
            log "Operation cancelled by user"
            exit 0
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "Checking recovery prerequisites..."
    
    # Check if running with appropriate permissions
    if [ ! -w "$PROJECT_DIR" ]; then
        error "No write permission to project directory: $PROJECT_DIR"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python3 is not installed"
        exit 1
    fi
    
    # Check backup manager
    if [ ! -f "$SCRIPT_DIR/backup_manager.py" ]; then
        error "Backup manager script not found: $SCRIPT_DIR/backup_manager.py"
        exit 1
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df "$PROJECT_DIR" | awk 'NR==2 {print $4}')
    REQUIRED_SPACE=2097152  # 2GB in KB
    
    if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
        warning "Low disk space. Available: $((AVAILABLE_SPACE / 1024))MB, Recommended: $((REQUIRED_SPACE / 1024))MB"
        read -p "Continue anyway? (yes/no): " continue_low_space
        if [ "$continue_low_space" != "yes" ]; then
            exit 1
        fi
    fi
    
    success "Prerequisites check passed"
}

# Create recovery point before restoration
create_recovery_point() {
    log "Creating recovery point before restoration..."
    
    RECOVERY_POINT_DIR="recovery_points/$(date +'%Y%m%d_%H%M%S')"
    mkdir -p "$RECOVERY_POINT_DIR"
    
    # Backup current state
    if [ -d "data" ]; then
        cp -r data "$RECOVERY_POINT_DIR/" 2>/dev/null || true
    fi
    
    if [ -d "config" ]; then
        cp -r config "$RECOVERY_POINT_DIR/" 2>/dev/null || true
    fi
    
    # Create recovery info
    cat > "$RECOVERY_POINT_DIR/recovery_info.json" << EOF
{
    "created_at": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "recovery_type": "$RECOVERY_TYPE",
    "backup_id": "$BACKUP_ID",
    "note": "Pre-recovery backup created automatically"
}
EOF
    
    success "Recovery point created: $RECOVERY_POINT_DIR"
    echo "RECOVERY_POINT=$RECOVERY_POINT_DIR" > /tmp/pii_recovery_point.env
}

# List available backups
list_backups() {
    log "Listing available backups..."
    
    cd "$PROJECT_DIR"
    python3 "$SCRIPT_DIR/backup_manager.py" list
}

# Restore database
restore_database() {
    local backup_id=$1
    local target_path=${2:-"data/pii_extraction_restored.db"}
    
    log "Restoring database from backup: $backup_id"
    
    confirm_operation "Database restoration from backup $backup_id"
    create_recovery_point
    
    # Stop services that might be using the database
    log "Stopping services..."
    pkill -f "streamlit" 2>/dev/null || true
    sleep 2
    
    cd "$PROJECT_DIR"
    
    if python3 "$SCRIPT_DIR/backup_manager.py" restore --type database --backup-id "$backup_id" --target "$target_path"; then
        success "Database restored successfully to: $target_path"
        
        # Verify database integrity
        verify_database_integrity "$target_path"
        
        log "Database restoration completed"
        return 0
    else
        error "Database restoration failed"
        return 1
    fi
}

# Restore files
restore_files() {
    local backup_id=$1
    local target_dir=${2:-"restored_files"}
    
    log "Restoring files from backup: $backup_id"
    
    confirm_operation "Files restoration from backup $backup_id to $target_dir"
    create_recovery_point
    
    cd "$PROJECT_DIR"
    
    if python3 "$SCRIPT_DIR/backup_manager.py" restore --type files --backup-id "$backup_id" --target "$target_dir"; then
        success "Files restored successfully to: $target_dir"
        
        # Set proper permissions
        find "$target_dir" -type d -exec chmod 755 {} \;
        find "$target_dir" -type f -exec chmod 644 {} \;
        
        log "Files restoration completed"
        return 0
    else
        error "Files restoration failed"
        return 1
    fi
}

# Restore models
restore_models() {
    local backup_id=$1
    local target_dir=${2:-"data/models"}
    
    log "Restoring models from backup: $backup_id"
    
    confirm_operation "Models restoration from backup $backup_id"
    create_recovery_point
    
    cd "$PROJECT_DIR"
    
    # Create target directory
    mkdir -p "$target_dir"
    
    if python3 "$SCRIPT_DIR/backup_manager.py" restore --type models --backup-id "$backup_id" --target "$target_dir"; then
        success "Models restored successfully to: $target_dir"
        
        log "Models restoration completed"
        return 0
    else
        error "Models restoration failed"
        return 1
    fi
}

# Full system recovery
full_system_recovery() {
    local backup_prefix=$1
    
    log "Performing full system recovery..."
    
    confirm_operation "FULL SYSTEM RECOVERY - This will restore database, files, and models"
    create_recovery_point
    
    # Stop all services
    log "Stopping all services..."
    pkill -f "streamlit" 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    sleep 5
    
    cd "$PROJECT_DIR"
    
    # Find latest backups if backup_prefix provided
    if [ -n "$backup_prefix" ]; then
        DATABASE_BACKUP=$(python3 "$SCRIPT_DIR/backup_manager.py" list | grep "^$backup_prefix.*database" | head -1 | awk '{print $1}')
        FILES_BACKUP=$(python3 "$SCRIPT_DIR/backup_manager.py" list | grep "^$backup_prefix.*files" | head -1 | awk '{print $1}')
        MODELS_BACKUP=$(python3 "$SCRIPT_DIR/backup_manager.py" list | grep "^$backup_prefix.*models" | head -1 | awk '{print $1}')
    else
        # Use latest available backups
        DATABASE_BACKUP=$(python3 "$SCRIPT_DIR/backup_manager.py" list | grep "database.*completed" | head -1 | awk '{print $1}')
        FILES_BACKUP=$(python3 "$SCRIPT_DIR/backup_manager.py" list | grep "files.*completed" | head -1 | awk '{print $1}')
        MODELS_BACKUP=$(python3 "$SCRIPT_DIR/backup_manager.py" list | grep "models.*completed" | head -1 | awk '{print $1}')
    fi
    
    local recovery_failed=false
    
    # Restore database
    if [ -n "$DATABASE_BACKUP" ]; then
        log "Restoring database: $DATABASE_BACKUP"
        if ! python3 "$SCRIPT_DIR/backup_manager.py" restore --type database --backup-id "$DATABASE_BACKUP"; then
            error "Database restoration failed"
            recovery_failed=true
        fi
    else
        warning "No database backup found"
    fi
    
    # Restore files
    if [ -n "$FILES_BACKUP" ]; then
        log "Restoring files: $FILES_BACKUP"
        if ! python3 "$SCRIPT_DIR/backup_manager.py" restore --type files --backup-id "$FILES_BACKUP" --target restored_files; then
            error "Files restoration failed"
            recovery_failed=true
        else
            # Move restored files to proper locations
            if [ -d "restored_files" ]; then
                rsync -av restored_files/ . --exclude="restored_files"
                rm -rf restored_files
            fi
        fi
    else
        warning "No files backup found"
    fi
    
    # Restore models
    if [ -n "$MODELS_BACKUP" ]; then
        log "Restoring models: $MODELS_BACKUP"
        if ! python3 "$SCRIPT_DIR/backup_manager.py" restore --type models --backup-id "$MODELS_BACKUP"; then
            error "Models restoration failed"
            recovery_failed=true
        fi
    else
        warning "No models backup found"
    fi
    
    if [ "$recovery_failed" = true ]; then
        error "Full system recovery completed with errors"
        return 1
    else
        success "Full system recovery completed successfully"
        return 0
    fi
}

# Emergency recovery
emergency_recovery() {
    log "EMERGENCY RECOVERY MODE ACTIVATED"
    
    echo -e "${RED}"
    echo "╔═══════════════════════════════════════════════════════════════════╗"
    echo "║                        EMERGENCY RECOVERY                        ║"
    echo "║                                                                   ║"
    echo "║  This will restore the system using the latest available         ║"
    echo "║  backups. Current data may be overwritten.                       ║"
    echo "╚═══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    confirm_operation "EMERGENCY RECOVERY - Restore from latest backups"
    
    # Perform full recovery with latest backups
    full_system_recovery ""
    
    # Restart services
    log "Restarting services..."
    docker-compose up -d 2>/dev/null || true
    
    # Verify system health
    verify_system_health
    
    success "Emergency recovery completed"
}

# Verify database integrity
verify_database_integrity() {
    local db_path=$1
    
    log "Verifying database integrity: $db_path"
    
    if [ ! -f "$db_path" ]; then
        warning "Database file not found: $db_path"
        return 1
    fi
    
    # Check if SQLite database
    if file "$db_path" | grep -q SQLite; then
        if sqlite3 "$db_path" "PRAGMA integrity_check;" | grep -q "ok"; then
            success "Database integrity check passed"
            return 0
        else
            error "Database integrity check failed"
            return 1
        fi
    else
        warning "Cannot verify integrity of non-SQLite database"
        return 0
    fi
}

# Verify system health after recovery
verify_system_health() {
    log "Verifying system health after recovery..."
    
    # Check if health check script is available
    if [ -f "$PROJECT_DIR/src/core/health_checks.py" ]; then
        cd "$PROJECT_DIR"
        
        if python3 -c "
from src.core.health_checks import get_system_health
health = get_system_health()
print(f'System Status: {health.status.value}')
if health.status.value == 'healthy':
    exit(0)
else:
    exit(1)
" 2>/dev/null; then
            success "System health verification passed"
        else
            warning "System health verification failed or degraded"
        fi
    else
        warning "Health check script not available"
    fi
}

# Rollback to recovery point
rollback_recovery() {
    if [ ! -f "/tmp/pii_recovery_point.env" ]; then
        error "No recovery point information found"
        exit 1
    fi
    
    source /tmp/pii_recovery_point.env
    
    if [ ! -d "$RECOVERY_POINT" ]; then
        error "Recovery point directory not found: $RECOVERY_POINT"
        exit 1
    fi
    
    log "Rolling back to recovery point: $RECOVERY_POINT"
    
    confirm_operation "Rollback to recovery point $RECOVERY_POINT"
    
    # Stop services
    pkill -f "streamlit" 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    
    # Restore from recovery point
    if [ -d "$RECOVERY_POINT/data" ]; then
        rm -rf data
        cp -r "$RECOVERY_POINT/data" .
    fi
    
    if [ -d "$RECOVERY_POINT/config" ]; then
        rm -rf config
        cp -r "$RECOVERY_POINT/config" .
    fi
    
    success "Rollback completed"
    
    # Clean up
    rm -f /tmp/pii_recovery_point.env
}

# Generate recovery report
generate_recovery_report() {
    local recovery_type=$1
    local backup_id=$2
    local status=$3
    
    REPORT_FILE="recovery_reports/recovery_report_$(date +'%Y%m%d_%H%M%S').json"
    mkdir -p recovery_reports
    
    cat > "$REPORT_FILE" << EOF
{
    "recovery_timestamp": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "recovery_type": "$recovery_type",
    "backup_id": "$backup_id",
    "status": "$status",
    "project_dir": "$PROJECT_DIR",
    "script_version": "1.0.0"
}
EOF
    
    log "Recovery report generated: $REPORT_FILE"
}

# Error handler
handle_error() {
    local exit_code=$?
    error "Recovery operation failed with exit code: $exit_code"
    
    # Generate failure report
    generate_recovery_report "$RECOVERY_TYPE" "$BACKUP_ID" "failed"
    
    # Suggest rollback if recovery point exists
    if [ -f "/tmp/pii_recovery_point.env" ]; then
        echo ""
        warning "A recovery point was created before this operation."
        warning "You can rollback using: $0 rollback"
    fi
    
    exit $exit_code
}

# Main execution
main() {
    display_banner
    
    # Parse arguments
    if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        usage
        exit 0
    fi
    
    RECOVERY_TYPE=$1
    BACKUP_ID=$2
    TARGET_DIR=$3
    
    # Set error handler
    trap handle_error ERR
    
    # Check prerequisites
    check_prerequisites
    
    # Execute recovery operation
    case $RECOVERY_TYPE in
        list)
            list_backups
            ;;
        database)
            if [ -z "$BACKUP_ID" ]; then
                error "Backup ID required for database recovery"
                usage
                exit 1
            fi
            restore_database "$BACKUP_ID" "$TARGET_DIR"
            generate_recovery_report "database" "$BACKUP_ID" "success"
            ;;
        files)
            if [ -z "$BACKUP_ID" ]; then
                error "Backup ID required for files recovery"
                usage
                exit 1
            fi
            restore_files "$BACKUP_ID" "$TARGET_DIR"
            generate_recovery_report "files" "$BACKUP_ID" "success"
            ;;
        models)
            if [ -z "$BACKUP_ID" ]; then
                error "Backup ID required for models recovery"
                usage
                exit 1
            fi
            restore_models "$BACKUP_ID" "$TARGET_DIR"
            generate_recovery_report "models" "$BACKUP_ID" "success"
            ;;
        full)
            full_system_recovery "$BACKUP_ID"
            generate_recovery_report "full" "$BACKUP_ID" "success"
            ;;
        emergency)
            emergency_recovery
            generate_recovery_report "emergency" "latest" "success"
            ;;
        rollback)
            rollback_recovery
            ;;
        *)
            error "Invalid recovery type: $RECOVERY_TYPE"
            usage
            exit 1
            ;;
    esac
    
    success "Recovery operation completed successfully!"
    
    echo ""
    log "Next steps:"
    log "1. Verify that the restored data is correct"
    log "2. Test system functionality"
    log "3. Restart any stopped services if needed"
    log "4. Monitor system health"
}

# Run main function
main "$@"