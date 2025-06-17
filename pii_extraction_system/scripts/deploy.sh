#!/bin/bash
# Deployment script for PII Extraction System
# Agent 5: DevOps & CI/CD Specialist

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV=${1:-development}
VERSION=${2:-latest}

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

# Validate environment
validate_environment() {
    log "Validating deployment environment: $ENV"
    
    case $ENV in
        development|staging|production)
            log "Environment $ENV is valid"
            ;;
        *)
            error "Invalid environment: $ENV. Must be development, staging, or production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running"
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Poetry is installed (for local builds)
    if ! command -v poetry &> /dev/null; then
        warning "Poetry is not installed. Some features may not work."
    fi
    
    success "Prerequisites check passed"
}

# Build application
build_application() {
    log "Building application for $ENV environment..."
    
    cd "$PROJECT_DIR"
    
    # Build Docker image
    docker build --target production -t "pii-extraction-system:$VERSION" .
    
    # Tag for environment
    docker tag "pii-extraction-system:$VERSION" "pii-extraction-system:$ENV"
    
    success "Application built successfully"
}

# Run tests
run_tests() {
    log "Running tests..."
    
    cd "$PROJECT_DIR"
    
    # Run tests in Docker container
    docker-compose --profile test run --rm pii-extraction-test
    
    if [ $? -eq 0 ]; then
        success "All tests passed"
    else
        error "Tests failed"
        exit 1
    fi
}

# Security scan
run_security_scan() {
    log "Running security scan..."
    
    cd "$PROJECT_DIR"
    
    # Run security scanning
    docker-compose --profile security run --rm security-scanner
    
    # Check for critical vulnerabilities
    if [ -f "security-reports/bandit.json" ]; then
        critical_issues=$(jq '.results | length' security-reports/bandit.json 2>/dev/null || echo "0")
        if [ "$critical_issues" -gt 0 ]; then
            warning "Security scan found $critical_issues issues. Review security-reports/"
        else
            success "Security scan passed"
        fi
    fi
}

# Deploy to environment
deploy_to_environment() {
    log "Deploying to $ENV environment..."
    
    cd "$PROJECT_DIR"
    
    case $ENV in
        development)
            deploy_development
            ;;
        staging)
            deploy_staging
            ;;
        production)
            deploy_production
            ;;
    esac
}

# Development deployment
deploy_development() {
    log "Starting development environment..."
    
    # Start development services
    docker-compose --profile dev up -d
    
    # Wait for services to be ready
    wait_for_health "http://localhost:8502/health" 60
    
    success "Development environment is running at http://localhost:8502"
}

# Staging deployment
deploy_staging() {
    log "Deploying to staging environment..."
    
    # Start staging services with monitoring
    docker-compose --profile monitoring up -d
    
    # Wait for services to be ready
    wait_for_health "http://localhost:8501/health" 120
    
    success "Staging environment is running at http://localhost:8501"
    log "Monitoring available at:"
    log "  - Grafana: http://localhost:3000 (admin/admin123)"
    log "  - Prometheus: http://localhost:9090"
    log "  - MLflow: http://localhost:5000"
}

# Production deployment
deploy_production() {
    log "Deploying to production environment..."
    
    # Additional checks for production
    if [ "$VERSION" == "latest" ]; then
        error "Cannot deploy 'latest' to production. Specify a version tag."
        exit 1
    fi
    
    # Deploy with full monitoring and security
    docker-compose --profile monitoring up -d
    
    # Wait for services to be ready
    wait_for_health "http://localhost:8501/health" 180
    
    # Run smoke tests
    run_smoke_tests
    
    success "Production deployment completed successfully"
}

# Wait for service health
wait_for_health() {
    local url=$1
    local timeout=${2:-60}
    local counter=0
    
    log "Waiting for service to be healthy: $url"
    
    while [ $counter -lt $timeout ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            success "Service is healthy"
            return 0
        fi
        
        sleep 5
        counter=$((counter + 5))
        log "Waiting... ($counter/$timeout seconds)"
    done
    
    error "Service did not become healthy within $timeout seconds"
    return 1
}

# Smoke tests for production
run_smoke_tests() {
    log "Running smoke tests..."
    
    # Test main application
    if curl -f -s "http://localhost:8501/health" > /dev/null; then
        success "Main application is responding"
    else
        error "Main application is not responding"
        exit 1
    fi
    
    # Test basic API endpoints
    if curl -f -s "http://localhost:8501/api/status" > /dev/null; then
        success "API endpoints are responding"
    else
        warning "API endpoints may not be fully ready"
    fi
}

# Rollback function
rollback() {
    local previous_version=$1
    
    if [ -z "$previous_version" ]; then
        error "Previous version not specified for rollback"
        exit 1
    fi
    
    warning "Rolling back to version: $previous_version"
    
    # Stop current deployment
    docker-compose down
    
    # Deploy previous version
    VERSION="$previous_version" deploy_to_environment
    
    success "Rollback completed"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Remove unused Docker images
    docker image prune -f
    
    # Clean up old containers
    docker container prune -f
    
    success "Cleanup completed"
}

# Health check function
health_check() {
    log "Performing health check..."
    
    # Check application health
    if curl -f -s "http://localhost:8501/health" > /dev/null; then
        success "Application is healthy"
    else
        error "Application health check failed"
        return 1
    fi
    
    # Check database connectivity (if applicable)
    # Add database health checks here
    
    # Check external services
    # Add external service checks here
    
    success "Health check passed"
}

# Main deployment function
main() {
    log "Starting deployment script for PII Extraction System"
    log "Environment: $ENV"
    log "Version: $VERSION"
    
    validate_environment
    check_prerequisites
    
    case "${3:-deploy}" in
        build)
            build_application
            ;;
        test)
            run_tests
            ;;
        security)
            run_security_scan
            ;;
        deploy)
            build_application
            run_tests
            run_security_scan
            deploy_to_environment
            ;;
        rollback)
            rollback "$4"
            ;;
        cleanup)
            cleanup
            ;;
        health)
            health_check
            ;;
        *)
            echo "Usage: $0 <environment> <version> [action]"
            echo "  environment: development|staging|production"
            echo "  version: version tag (e.g., v1.0.0) or latest"
            echo "  action: build|test|security|deploy|rollback|cleanup|health"
            exit 1
            ;;
    esac
    
    success "Deployment script completed successfully"
}

# Handle script interruption
trap 'error "Script interrupted"; exit 1' INT TERM

# Run main function
main "$@"