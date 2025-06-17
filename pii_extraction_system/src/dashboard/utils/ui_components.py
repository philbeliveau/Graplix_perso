"""
UI Components for PII Extraction Dashboard

This module provides reusable UI components and utilities for consistent
dashboard interface design and functionality.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

def show_system_status():
    """Display system status indicators"""
    status = st.session_state.get('system_status', {})
    health = status.get('system_health', 'unknown')
    queue_size = status.get('processing_queue', 0)
    active_models = status.get('active_models', [])
    
    # Health indicator
    if health == 'healthy':
        st.markdown('🟢 **System Healthy**')
    elif health == 'warning':
        st.markdown('🟡 **System Warning**')
    elif health == 'error':
        st.markdown('🔴 **System Error**')
    else:
        st.markdown('⚪ **Status Unknown**')
    
    # Queue status
    if queue_size > 0:
        st.markdown(f"📋 Queue: {queue_size} documents")
    else:
        st.markdown("📋 Queue: Empty")
    
    # Active models
    if active_models:
        st.markdown(f"🤖 Models: {len(active_models)} active")
    else:
        st.markdown("🤖 Models: None active")

def create_metric_card(title: str, value: str, delta: Optional[str] = None, 
                      help_text: Optional[str] = None):
    """Create a metric display card"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.metric(
            label=title,
            value=value,
            delta=delta,
            help=help_text
        )

def create_progress_bar(label: str, progress: float, color: str = "#1f77b4"):
    """Create a custom progress bar"""
    st.markdown(f"**{label}**")
    progress_html = f"""
    <div style="background-color: #e0e0e0; border-radius: 10px; padding: 3px;">
        <div style="background-color: {color}; width: {progress*100}%; 
                    height: 20px; border-radius: 7px; text-align: center; 
                    line-height: 20px; color: white; font-size: 12px;">
            {progress*100:.1f}%
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

def display_pii_highlights(text: str, pii_entities: List[Dict], 
                          categories_colors: Optional[Dict] = None) -> str:
    """Generate HTML with PII entities highlighted"""
    if not categories_colors:
        categories_colors = {
            'PERSON': '#ff9999',
            'EMAIL': '#99ccff', 
            'PHONE': '#ffcc99',
            'ADDRESS': '#99ff99',
            'SSN': '#ff99ff',
            'CREDIT_CARD': '#ffff99',
            'DATE': '#cc99ff',
            'ORGANIZATION': '#99ffcc'
        }
    
    # Sort entities by start position (reverse order for replacement)
    sorted_entities = sorted(pii_entities, key=lambda x: x.get('start', 0), reverse=True)
    
    highlighted_text = text
    for entity in sorted_entities:
        start = entity.get('start', 0)
        end = entity.get('end', 0)
        entity_type = entity.get('type', 'UNKNOWN')
        confidence = entity.get('confidence', 0.0)
        
        color = categories_colors.get(entity_type, '#cccccc')
        
        # Create highlighted span with tooltip
        highlighted_span = f"""
        <span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; 
                     border: 1px solid #ccc;" 
              title="{entity_type} (Confidence: {confidence:.2f})">
            {text[start:end]}
        </span>
        """
        
        highlighted_text = (highlighted_text[:start] + highlighted_span + 
                          highlighted_text[end:])
    
    return highlighted_text

def create_confidence_histogram(confidences: List[float], title: str = "Confidence Distribution"):
    """Create confidence score histogram"""
    fig = px.histogram(
        x=confidences,
        bins=20,
        title=title,
        labels={'x': 'Confidence Score', 'y': 'Count'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(
        xaxis_range=[0, 1],
        showlegend=False,
        height=300
    )
    return fig

def create_pii_category_chart(pii_counts: Dict[str, int]):
    """Create PII category distribution chart"""
    if not pii_counts:
        return go.Figure().add_annotation(text="No PII detected", 
                                        x=0.5, y=0.5, showarrow=False)
    
    fig = px.bar(
        x=list(pii_counts.keys()),
        y=list(pii_counts.values()),
        title="PII Categories Detected",
        labels={'x': 'PII Category', 'y': 'Count'},
        color=list(pii_counts.values()),
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig

def create_performance_metrics_chart(metrics: Dict[str, float]):
    """Create performance metrics radar chart"""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance Metrics',
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Model Performance Metrics",
        height=400
    )
    return fig

def create_model_comparison_table(comparison_data: List[Dict]):
    """Create model comparison table"""
    if not comparison_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(comparison_data)
    return df

def show_file_uploader(key: str, allowed_types: List[str], 
                      max_size_mb: int = 50, multiple: bool = False):
    """Enhanced file uploader with validation"""
    uploaded_files = st.file_uploader(
        "Choose files" if multiple else "Choose file",
        type=allowed_types,
        key=key,
        accept_multiple_files=multiple,
        help=f"Supported formats: {', '.join(allowed_types)}. Max size: {max_size_mb}MB"
    )
    
    if uploaded_files:
        if not multiple:
            uploaded_files = [uploaded_files]
        
        valid_files = []
        for file in uploaded_files:
            # Check file size
            if file.size > max_size_mb * 1024 * 1024:
                st.error(f"File {file.name} is too large. Maximum size: {max_size_mb}MB")
                continue
            
            # Check file type
            file_extension = file.name.split('.')[-1].lower()
            if file_extension not in [t.lower() for t in allowed_types]:
                st.error(f"File {file.name} has unsupported format")
                continue
            
            valid_files.append(file)
        
        return valid_files if multiple else (valid_files[0] if valid_files else None)
    
    return [] if multiple else None

def show_processing_status(status: str, progress: Optional[float] = None):
    """Show processing status with optional progress"""
    status_icons = {
        'uploaded': '📄',
        'processing': '⚙️',
        'completed': '✅',
        'error': '❌',
        'queued': '⏳'
    }
    
    icon = status_icons.get(status, '❓')
    st.markdown(f"{icon} **Status:** {status.capitalize()}")
    
    if progress is not None and 0 <= progress <= 1:
        st.progress(progress)

def create_error_analysis_chart(error_data: Dict[str, List]):
    """Create error analysis visualization"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('False Positives', 'False Negatives'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # False positives
    fp_categories = error_data.get('false_positives', {})
    if fp_categories:
        fig.add_trace(
            go.Bar(x=list(fp_categories.keys()), y=list(fp_categories.values()),
                   name='False Positives', marker_color='red'),
            row=1, col=1
        )
    
    # False negatives  
    fn_categories = error_data.get('false_negatives', {})
    if fn_categories:
        fig.add_trace(
            go.Bar(x=list(fn_categories.keys()), y=list(fn_categories.values()),
                   name='False Negatives', marker_color='orange'),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=False)
    return fig

def export_data(data: Any, filename: str, format_type: str = 'json'):
    """Create download button for data export"""
    if format_type == 'json':
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, indent=2, default=str)
        else:
            data_str = str(data)
        mime_type = 'application/json'
    elif format_type == 'csv':
        if isinstance(data, pd.DataFrame):
            data_str = data.to_csv(index=False)
        else:
            data_str = str(data)
        mime_type = 'text/csv'
    else:
        data_str = str(data)
        mime_type = 'text/plain'
    
    st.download_button(
        label=f"Download {format_type.upper()}",
        data=data_str,
        file_name=f"{filename}.{format_type}",
        mime=mime_type
    )

def show_annotation_interface(text: str, existing_annotations: List[Dict] = None):
    """Show interface for manual annotation of PII entities"""
    st.markdown("### Manual Annotation")
    st.markdown("Click and drag to select text, then choose PII category:")
    
    # Categories for annotation
    categories = ['PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'SSN', 
                 'CREDIT_CARD', 'DATE', 'ORGANIZATION', 'OTHER']
    
    selected_category = st.selectbox("PII Category", categories)
    
    # Text area for annotation (simplified - in production use more sophisticated tools)
    st.text_area("Document Text", text, height=300)
    
    # Show existing annotations
    if existing_annotations:
        st.markdown("### Existing Annotations")
        for i, annotation in enumerate(existing_annotations):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.text(f"{annotation.get('text', '')}")
            with col2:
                st.text(annotation.get('type', ''))
            with col3:
                if st.button(f"Remove", key=f"remove_{i}"):
                    existing_annotations.pop(i)
                    st.rerun()
    
    return existing_annotations or []