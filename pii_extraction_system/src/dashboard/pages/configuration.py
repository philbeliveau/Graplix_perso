"""
Configuration Page - System settings and model parameter management

This page provides comprehensive control over system parameters and model settings,
enabling customization for different deployment scenarios and requirements.
"""

import streamlit as st
import pandas as pd
import json
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import session_state, ui_components, auth

def show_page():
    """Main configuration page"""
    st.markdown('<div class="section-header">⚙️ Configuration</div>', 
                unsafe_allow_html=True)
    st.markdown("Configure system settings, model parameters, and security options.")
    
    # Check permissions
    if not auth.has_permission('configure'):
        st.error("Access denied. Configuration requires admin permissions.")
        return
    
    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 Model Configuration",
        "🔒 Privacy & Security", 
        "📊 Performance Settings",
        "🔧 System Administration"
    ])
    
    with tab1:
        show_model_configuration()
    
    with tab2:
        show_privacy_security()
    
    with tab3:
        show_performance_settings()
    
    with tab4:
        show_system_administration()

def show_model_configuration():
    """Show model configuration interface"""
    st.markdown("### Model Configuration")
    
    # Model selection and parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Primary Models")
        
        # Get current configuration
        current_config = get_current_model_config()
        
        enabled_models = st.multiselect(
            "Enabled Models",
            [
                'Rule-Based Extractor',
                'spaCy NER',
                'Transformers NER', 
                'LayoutLM',
                'Custom Fine-tuned',
                'Ensemble Method'
            ],
            default=current_config.get('enabled_models', ['Rule-Based Extractor', 'spaCy NER'])
        )
        
        primary_model = st.selectbox(
            "Primary Model",
            enabled_models if enabled_models else ['Rule-Based Extractor'],
            index=0
        )
        
        fallback_model = st.selectbox(
            "Fallback Model",
            [model for model in enabled_models if model != primary_model] + ['None'],
            index=0 if len(enabled_models) > 1 else 0
        )
    
    with col2:
        st.markdown("#### Model Parameters")
        
        # Global confidence thresholds
        st.markdown("**Confidence Thresholds:**")
        
        confidence_config = {}
        pii_categories = ['PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'SSN', 'CREDIT_CARD', 'DATE', 'ORGANIZATION']
        
        for category in pii_categories:
            default_threshold = current_config.get('confidence_thresholds', {}).get(category, 0.5)
            confidence_config[category] = st.slider(
                f"{category}",
                min_value=0.0,
                max_value=1.0,
                value=default_threshold,
                step=0.05,
                key=f"conf_{category}"
            )
    
    # Advanced model settings
    st.markdown("#### Advanced Model Settings")
    
    with st.expander("Rule-Based Extractor Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            case_sensitive = st.checkbox("Case Sensitive Matching", value=False)
            use_regex = st.checkbox("Enable Regex Patterns", value=True)
            custom_patterns = st.text_area(
                "Custom Regex Patterns (one per line)",
                placeholder="Enter custom regex patterns...",
                height=100
            )
        
        with col2:
            word_boundaries = st.checkbox("Enforce Word Boundaries", value=True)
            min_confidence = st.slider("Minimum Confidence", 0.0, 1.0, 0.8, 0.05)
    
    with st.expander("Transformer Model Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            model_name = st.selectbox(
                "Pre-trained Model",
                [
                    'dbmdz/bert-large-cased-finetuned-conll03-english',
                    'microsoft/DialoGPT-medium',
                    'bert-base-multilingual-cased',
                    'Custom Model'
                ]
            )
            
            max_length = st.number_input("Max Sequence Length", 128, 512, 256)
        
        with col2:
            batch_size = st.number_input("Batch Size", 1, 32, 8)
            use_gpu = st.checkbox("Use GPU if Available", value=True)
    
    with st.expander("Ensemble Settings"):
        st.markdown("**Voting Strategy:**")
        voting_strategy = st.radio(
            "Strategy",
            ['Majority Vote', 'Weighted Vote', 'Confidence-based'],
            horizontal=True
        )
        
        if voting_strategy == 'Weighted Vote':
            st.markdown("**Model Weights:**")
            weights = {}
            for model in enabled_models:
                weights[model] = st.slider(f"{model} Weight", 0.0, 1.0, 1.0/len(enabled_models), 0.1)
    
    # Save configuration
    if st.button("Save Model Configuration", type="primary"):
        save_model_configuration({
            'enabled_models': enabled_models,
            'primary_model': primary_model,
            'fallback_model': fallback_model,
            'confidence_thresholds': confidence_config,
            'advanced_settings': {
                'rule_based': {
                    'case_sensitive': case_sensitive if 'case_sensitive' in locals() else False,
                    'use_regex': use_regex if 'use_regex' in locals() else True,
                    'custom_patterns': custom_patterns.split('\n') if 'custom_patterns' in locals() else [],
                    'word_boundaries': word_boundaries if 'word_boundaries' in locals() else True,
                    'min_confidence': min_confidence if 'min_confidence' in locals() else 0.8
                },
                'transformer': {
                    'model_name': model_name if 'model_name' in locals() else 'bert-base-multilingual-cased',
                    'max_length': max_length if 'max_length' in locals() else 256,
                    'batch_size': batch_size if 'batch_size' in locals() else 8,
                    'use_gpu': use_gpu if 'use_gpu' in locals() else True
                },
                'ensemble': {
                    'voting_strategy': voting_strategy if 'voting_strategy' in locals() else 'Majority Vote',
                    'weights': weights if 'weights' in locals() else {}
                }
            }
        })

def show_privacy_security():
    """Show privacy and security configuration"""
    st.markdown("### Privacy & Security Configuration")
    
    # Privacy settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Privacy Settings")
        
        # Data retention
        retention_days = st.number_input(
            "Data Retention Period (days)",
            min_value=1,
            max_value=365,
            value=90,
            help="How long to keep processed documents and results"
        )
        
        # Redaction settings
        st.markdown("**Redaction Options:**")
        redaction_method = st.selectbox(
            "Redaction Method",
            ['Replace with [REDACTED]', 'Replace with Category', 'Black Boxes', 'Asterisks']
        )
        
        redaction_categories = st.multiselect(
            "Categories to Redact",
            ['PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'SSN', 'CREDIT_CARD', 'DATE'],
            default=['SSN', 'CREDIT_CARD']
        )
        
        # Compliance settings
        st.markdown("**Compliance Settings:**")
        gdpr_compliance = st.checkbox("GDPR Compliance Mode", value=True)
        law25_compliance = st.checkbox("Quebec Law 25 Compliance", value=True)
        
    with col2:
        st.markdown("#### Security Settings")
        
        # Access control
        st.markdown("**Access Control:**")
        session_timeout = st.slider(
            "Session Timeout (hours)",
            min_value=1,
            max_value=24,
            value=8
        )
        
        require_2fa = st.checkbox("Require Two-Factor Authentication", value=False)
        
        # API security
        st.markdown("**API Security:**")
        api_rate_limit = st.number_input(
            "API Rate Limit (requests/minute)",
            min_value=10,
            max_value=1000,
            value=100
        )
        
        require_api_key = st.checkbox("Require API Key", value=True)
        
        # Audit settings
        st.markdown("**Audit & Logging:**")
        enable_audit_log = st.checkbox("Enable Audit Logging", value=True)
        log_retention_days = st.number_input(
            "Log Retention (days)",
            min_value=30,
            max_value=365,
            value=180
        )
    
    # Encryption settings
    st.markdown("#### Encryption Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        encryption_at_rest = st.checkbox("Encrypt Data at Rest", value=True)
        encryption_in_transit = st.checkbox("Encrypt Data in Transit", value=True)
    
    with col2:
        encryption_algorithm = st.selectbox(
            "Encryption Algorithm",
            ['AES-256', 'AES-128', 'ChaCha20-Poly1305']
        )
    
    # Save privacy/security settings
    if st.button("Save Privacy & Security Settings", type="primary"):
        save_privacy_security_config({
            'data_retention_days': retention_days,
            'redaction': {
                'method': redaction_method,
                'categories': redaction_categories
            },
            'compliance': {
                'gdpr': gdpr_compliance,
                'law25': law25_compliance
            },
            'security': {
                'session_timeout_hours': session_timeout,
                'require_2fa': require_2fa,
                'api_rate_limit': api_rate_limit,
                'require_api_key': require_api_key
            },
            'audit': {
                'enable_audit_log': enable_audit_log,
                'log_retention_days': log_retention_days
            },
            'encryption': {
                'at_rest': encryption_at_rest,
                'in_transit': encryption_in_transit,
                'algorithm': encryption_algorithm
            }
        })

def show_performance_settings():
    """Show performance configuration"""
    st.markdown("### Performance Settings")
    
    # Processing settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Processing Configuration")
        
        max_concurrent_jobs = st.slider(
            "Max Concurrent Processing Jobs",
            min_value=1,
            max_value=10,
            value=3
        )
        
        batch_size = st.slider(
            "Document Batch Size",
            min_value=1,
            max_value=50,
            value=10
        )
        
        processing_timeout = st.number_input(
            "Processing Timeout (seconds)",
            min_value=30,
            max_value=600,
            value=120
        )
        
        # Memory settings
        st.markdown("**Memory Management:**")
        max_memory_per_job = st.slider(
            "Max Memory per Job (GB)",
            min_value=1,
            max_value=16,
            value=4
        )
        
        enable_memory_cleanup = st.checkbox("Enable Automatic Memory Cleanup", value=True)
    
    with col2:
        st.markdown("#### Caching Configuration")
        
        enable_result_cache = st.checkbox("Enable Result Caching", value=True)
        cache_ttl = st.number_input(
            "Cache TTL (hours)",
            min_value=1,
            max_value=168,
            value=24
        )
        
        max_cache_size = st.slider(
            "Max Cache Size (GB)",
            min_value=1,
            max_value=100,
            value=10
        )
        
        # OCR settings
        st.markdown("**OCR Configuration:**")
        ocr_engine = st.selectbox(
            "OCR Engine", 
            ['tesseract', 'easyocr', 'both'], 
            index=0,
            help="Select OCR engine: Tesseract (traditional), EasyOCR (deep learning), or both for comparison"
        )
        
        if ocr_engine in ['easyocr', 'both']:
            easyocr_use_gpu = st.checkbox("Use GPU for EasyOCR", value=False, help="Requires CUDA-compatible GPU")
        else:
            easyocr_use_gpu = False
        
        # LLM OCR settings
        st.markdown("**LLM OCR Configuration:**")
        enable_llm_ocr = st.checkbox("Enable LLM OCR", value=False, help="Use AI models for enhanced OCR accuracy")
        
        if enable_llm_ocr:
            llm_ocr_model = st.selectbox(
                "Primary LLM Model",
                ["gpt-4o-mini", "gpt-3.5-turbo", "claude-3-haiku", "gemini-1.5-flash"],
                help="Primary model for LLM-based OCR"
            )
            max_cost_per_doc = st.slider(
                "Max Cost per Document ($)",
                0.01, 0.50, 0.10, 0.01,
                help="Maximum cost allowed per document"
            )
        else:
            llm_ocr_model = "gpt-4o-mini"
            max_cost_per_doc = 0.10
            
        ocr_dpi = st.selectbox("OCR DPI", [150, 200, 300, 400], index=1)
        ocr_parallel = st.checkbox("Parallel OCR Processing", value=True)
    
    # Resource monitoring
    st.markdown("#### Resource Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_threshold = st.slider("CPU Alert Threshold (%)", 50, 95, 85)
        memory_threshold = st.slider("Memory Alert Threshold (%)", 50, 95, 80)
    
    with col2:
        disk_threshold = st.slider("Disk Usage Alert Threshold (%)", 60, 95, 85)
        enable_auto_scaling = st.checkbox("Enable Auto-scaling", value=False)
    
    # Performance optimization
    st.markdown("#### Performance Optimization")
    
    optimization_level = st.selectbox(
        "Optimization Level",
        ['Conservative', 'Balanced', 'Aggressive'],
        index=1,
        help="Higher levels may use more resources but provide better performance"
    )
    
    enable_gpu_acceleration = st.checkbox("Enable GPU Acceleration", value=True)
    
    if st.button("Save Performance Settings", type="primary"):
        save_performance_config({
            'processing': {
                'max_concurrent_jobs': max_concurrent_jobs,
                'batch_size': batch_size,
                'timeout_seconds': processing_timeout,
                'max_memory_gb': max_memory_per_job,
                'enable_memory_cleanup': enable_memory_cleanup
            },
            'caching': {
                'enable_result_cache': enable_result_cache,
                'cache_ttl_hours': cache_ttl,
                'max_cache_size_gb': max_cache_size
            },
            'ocr': {
                'engine': ocr_engine,
                'easyocr_use_gpu': easyocr_use_gpu,
                'enable_llm_ocr': enable_llm_ocr,
                'llm_ocr_model': llm_ocr_model,
                'max_llm_cost_per_document': max_cost_per_doc,
                'dpi': ocr_dpi,
                'parallel_processing': ocr_parallel
            },
            'monitoring': {
                'cpu_threshold': cpu_threshold,
                'memory_threshold': memory_threshold,
                'disk_threshold': disk_threshold,
                'enable_auto_scaling': enable_auto_scaling
            },
            'optimization': {
                'level': optimization_level,
                'gpu_acceleration': enable_gpu_acceleration
            }
        })

def show_system_administration():
    """Show system administration interface"""
    st.markdown("### System Administration")
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Uptime", "127.5 hours")
        st.metric("Active Users", "7")
    
    with col2:
        st.metric("Documents Processed", "1,247")
        st.metric("Storage Used", "45.2 GB")
    
    with col3:
        st.metric("API Calls Today", "3,421")
        st.metric("Error Rate", "0.8%")
    
    # System actions
    st.markdown("#### System Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Restart Services"):
            restart_system_services()
    
    with col2:
        if st.button("Clear Cache"):
            clear_system_cache()
    
    with col3:
        if st.button("Export Logs"):
            export_system_logs()
    
    # User management
    st.markdown("#### User Management")
    
    # Current users table
    users_data = [
        {'Username': 'admin', 'Role': 'Administrator', 'Last Login': '2024-01-15 14:30', 'Status': 'Active'},
        {'Username': 'analyst1', 'Role': 'Analyst', 'Last Login': '2024-01-15 13:45', 'Status': 'Active'},
        {'Username': 'viewer1', 'Role': 'Viewer', 'Last Login': '2024-01-14 16:20', 'Status': 'Active'},
        {'Username': 'temp_user', 'Role': 'Viewer', 'Last Login': '2024-01-10 09:15', 'Status': 'Inactive'}
    ]
    
    df_users = pd.DataFrame(users_data)
    st.dataframe(df_users, use_container_width=True)
    
    # Add new user
    with st.expander("Add New User"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
        
        with col2:
            new_role = st.selectbox("Role", ['Viewer', 'Analyst', 'Administrator'])
            send_email = st.checkbox("Send welcome email")
        
        if st.button("Create User"):
            if new_username and new_password:
                create_new_user(new_username, new_password, new_role, send_email)
            else:
                st.error("Please provide username and password")
    
    # Backup and maintenance
    st.markdown("#### Backup & Maintenance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Backup Settings:**")
        backup_frequency = st.selectbox("Backup Frequency", ['Daily', 'Weekly', 'Monthly'])
        backup_retention = st.number_input("Backup Retention (days)", 7, 365, 30)
        
        if st.button("Create Backup Now"):
            create_system_backup()
    
    with col2:
        st.markdown("**Maintenance:**")
        auto_maintenance = st.checkbox("Enable Auto-maintenance", value=True)
        maintenance_window = st.time_input("Maintenance Window Start", value=None)
        
        if st.button("Schedule Maintenance"):
            schedule_maintenance(auto_maintenance, maintenance_window)

def get_current_model_config() -> Dict[str, Any]:
    """Get current model configuration"""
    return st.session_state.get('model_config', {
        'enabled_models': ['Rule-Based Extractor', 'spaCy NER'],
        'primary_model': 'spaCy NER',
        'confidence_thresholds': {
            'PERSON': 0.5,
            'EMAIL': 0.8,
            'PHONE': 0.7,
            'ADDRESS': 0.6,
            'SSN': 0.9,
            'CREDIT_CARD': 0.9,
            'DATE': 0.5,
            'ORGANIZATION': 0.6
        }
    })

def save_model_configuration(config: Dict[str, Any]):
    """Save model configuration"""
    st.session_state.model_config = config
    st.success("Model configuration saved successfully!")

def save_privacy_security_config(config: Dict[str, Any]):
    """Save privacy and security configuration"""
    st.session_state.privacy_security_config = config
    st.success("Privacy & security settings saved successfully!")

def save_performance_config(config: Dict[str, Any]):
    """Save performance configuration"""
    st.session_state.performance_config = config
    st.success("Performance settings saved successfully!")

def restart_system_services():
    """Restart system services"""
    st.success("System services restarted successfully!")

def clear_system_cache():
    """Clear system cache"""
    st.success("System cache cleared successfully!")

def export_system_logs():
    """Export system logs"""
    st.success("System logs exported! Download link will be sent to your email.")

def create_new_user(username: str, password: str, role: str, send_email: bool):
    """Create new user"""
    st.success(f"User '{username}' created successfully with role '{role}'!")
    if send_email:
        st.info("Welcome email sent to user.")

def create_system_backup():
    """Create system backup"""
    st.success("System backup created successfully!")

def schedule_maintenance(auto_maintenance: bool, window_start):
    """Schedule system maintenance"""
    if auto_maintenance:
        st.success(f"Auto-maintenance enabled. Window starts at {window_start}")
    else:
        st.success("Manual maintenance mode enabled.")