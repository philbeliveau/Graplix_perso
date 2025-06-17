"""
Session State Management for PII Extraction Dashboard

This module manages Streamlit session state for maintaining user data,
uploaded documents, and application state across page navigation.
"""

import streamlit as st
from typing import Dict, Any, Optional
from datetime import datetime

def initialize_session_state():
    """Initialize all required session state variables"""
    
    # Authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = 'viewer'
    
    # Document processing state
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = {}
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}
    if 'current_document' not in st.session_state:
        st.session_state.current_document = None
    
    # Model and processing state
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []
    if 'model_comparison_results' not in st.session_state:
        st.session_state.model_comparison_results = {}
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.5
    
    # Analysis and feedback state
    if 'error_annotations' not in st.session_state:
        st.session_state.error_annotations = {}
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {}
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []
    
    # System monitoring state
    if 'system_status' not in st.session_state:
        st.session_state.system_status = {
            'processing_queue': 0,
            'active_models': [],
            'system_health': 'healthy',
            'last_update': datetime.now()
        }
    
    # Configuration state
    if 'app_config' not in st.session_state:
        st.session_state.app_config = {
            'pii_categories': [
                'PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'SSN', 
                'CREDIT_CARD', 'DATE', 'ORGANIZATION'
            ],
            'supported_formats': ['pdf', 'docx', 'jpg', 'png'],
            'max_file_size_mb': 50,
            'batch_size': 10
        }

def get_session_value(key: str, default: Any = None) -> Any:
    """Get value from session state with default fallback"""
    return st.session_state.get(key, default)

def set_session_value(key: str, value: Any) -> None:
    """Set value in session state"""
    st.session_state[key] = value

def update_session_dict(key: str, update_dict: Dict[str, Any]) -> None:
    """Update a dictionary in session state"""
    if key not in st.session_state:
        st.session_state[key] = {}
    st.session_state[key].update(update_dict)

def clear_session_data(keys: Optional[list] = None) -> None:
    """Clear specific session state keys or all data"""
    if keys is None:
        # Clear all non-auth data
        keys_to_clear = [
            'uploaded_documents', 'processing_results', 'current_document',
            'model_comparison_results', 'error_annotations', 'feedback_data'
        ]
    else:
        keys_to_clear = keys
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def add_uploaded_document(file_id: str, file_info: Dict[str, Any]) -> None:
    """Add uploaded document to session state"""
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = {}
    st.session_state.uploaded_documents[file_id] = {
        **file_info,
        'upload_time': datetime.now(),
        'status': 'uploaded'
    }

def update_document_status(file_id: str, status: str, **kwargs) -> None:
    """Update document processing status"""
    if file_id in st.session_state.uploaded_documents:
        st.session_state.uploaded_documents[file_id]['status'] = status
        st.session_state.uploaded_documents[file_id].update(kwargs)

def get_document_info(file_id: str) -> Optional[Dict[str, Any]]:
    """Get document information by file ID"""
    return st.session_state.uploaded_documents.get(file_id)

def store_processing_results(file_id: str, results: Dict[str, Any]) -> None:
    """Store PII extraction results for a document"""
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}
    st.session_state.processing_results[file_id] = {
        **results,
        'processed_time': datetime.now()
    }

def get_processing_results(file_id: str) -> Optional[Dict[str, Any]]:
    """Get processing results for a document"""
    return st.session_state.processing_results.get(file_id)

def add_error_annotation(file_id: str, annotation: Dict[str, Any]) -> None:
    """Add error annotation for feedback"""
    if 'error_annotations' not in st.session_state:
        st.session_state.error_annotations = {}
    if file_id not in st.session_state.error_annotations:
        st.session_state.error_annotations[file_id] = []
    
    st.session_state.error_annotations[file_id].append({
        **annotation,
        'timestamp': datetime.now()
    })

def get_error_annotations(file_id: str) -> list:
    """Get error annotations for a document"""
    return st.session_state.error_annotations.get(file_id, [])

def update_system_status(status_data: Dict[str, Any]) -> None:
    """Update system status information"""
    if 'system_status' not in st.session_state:
        st.session_state.system_status = {}
    
    st.session_state.system_status.update({
        **status_data,
        'last_update': datetime.now()
    })

def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    return st.session_state.get('system_status', {
        'processing_queue': 0,
        'active_models': [],
        'system_health': 'unknown',
        'last_update': datetime.now()
    })