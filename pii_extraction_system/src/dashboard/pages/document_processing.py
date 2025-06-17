"""
Document Processing Page - Primary interface for individual document processing

This page provides the main interface for uploading documents, processing them
for PII extraction, and viewing results with interactive highlighting.
"""

import streamlit as st
import uuid
import io
from PIL import Image
import PyPDF2
from docx import Document
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import session_state, ui_components, auth
from core.pipeline import PIIExtractionPipeline
from utils.document_processor import DocumentProcessor

def show_page():
    """Main document processing page"""
    st.markdown('<div class="section-header">📄 Document Processing</div>', 
                unsafe_allow_html=True)
    st.markdown("Upload and process individual documents for PII extraction.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Two column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        show_upload_section()
    
    with col2:
        show_document_list()
    
    # Full width results section
    show_processing_results()

def show_upload_section():
    """Document upload interface"""
    st.markdown("### Upload Documents")
    
    # Get configuration
    config = st.session_state.get('app_config', {})
    supported_formats = config.get('supported_formats', ['pdf', 'docx', 'xlsx', 'xls', 'jpg', 'png'])
    max_size_mb = config.get('max_file_size_mb', 50)
    
    # Password input for protected documents
    st.markdown("#### Document Password (if required)")
    st.text_input(
        "Enter password for protected documents",
        type="password",
        value="Hubert",
        help="Leave blank for unprotected documents. Default password 'Hubert' is pre-filled.",
        key="document_password"
    )
    
    # Processing options
    st.markdown("#### Processing Options")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        processing_model = st.selectbox(
            "Select Processing Model",
            ['Ensemble Method (All Models)', 'Rule-Based Extractor', 'spaCy NER', 'Transformers NER', 'Layout-Aware NER'],
            help="Choose which PII extraction model(s) to use"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Only show PII entities with confidence above this threshold"
        )
    
    # Store processing options in session state
    st.session_state.processing_model = processing_model
    st.session_state.confidence_threshold = confidence_threshold
    
    # File uploader
    uploaded_files = ui_components.show_file_uploader(
        key="document_uploader",
        allowed_types=supported_formats,
        max_size_mb=max_size_mb,
        multiple=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file is not None:
                # Check if file is already processed to avoid duplication
                file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                if file_key not in st.session_state.get('processed_files', set()):
                    if 'processed_files' not in st.session_state:
                        st.session_state.processed_files = set()
                    st.session_state.processed_files.add(file_key)
                    process_uploaded_file(uploaded_file)

def process_uploaded_file(uploaded_file):
    """Process and store uploaded file"""
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Extract file info
    file_info = {
        'name': uploaded_file.name,
        'size': uploaded_file.size,
        'type': uploaded_file.type,
        'file_id': file_id
    }
    
    # Store file data
    file_data = uploaded_file.read()
    
    # Add to session state
    session_state.add_uploaded_document(file_id, file_info)
    
    # Store file content for processing
    if 'file_data' not in st.session_state:
        st.session_state.file_data = {}
    st.session_state.file_data[file_id] = file_data
    
    st.success(f"✅ Uploaded: {uploaded_file.name}")
    st.info("📝 Document uploaded successfully. Use the 'View' button to select it, then click 'Process' to analyze for PII.")

def show_document_list():
    """Display uploaded documents"""
    st.markdown("### Uploaded Documents")
    
    uploaded_docs = st.session_state.get('uploaded_documents', {})
    
    if not uploaded_docs:
        st.info("No documents uploaded yet.")
        return
    
    for file_id, doc_info in uploaded_docs.items():
        with st.expander(f"📄 {doc_info['name']}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Size:** {doc_info['size']:,} bytes")
                st.write(f"**Type:** {doc_info['type']}")
                st.write(f"**Status:** {doc_info.get('status', 'uploaded')}")
                
                if doc_info.get('status') == 'processing':
                    st.spinner("Processing...")
                elif doc_info.get('status') == 'completed':
                    st.success("✅ Processing completed")
                elif doc_info.get('status') == 'error':
                    st.error("❌ Processing failed")
            
            with col2:
                if auth.has_permission('write'):
                    if st.button("Process", key=f"btn_process_{file_id}"):
                        process_document(file_id)
                
                if st.button("View", key=f"btn_view_{file_id}"):
                    st.session_state.current_document = file_id
                    st.rerun()
                
                if auth.has_permission('delete'):
                    if st.button("Delete", key=f"btn_delete_{file_id}"):
                        delete_document(file_id)

def process_document(file_id: str):
    """Process document for PII extraction"""
    doc_info = session_state.get_document_info(file_id)
    if not doc_info:
        st.error("Document not found")
        return
    
    # Update status
    session_state.update_document_status(file_id, 'processing')
    
    try:
        # Get file data
        file_data = st.session_state.file_data.get(file_id)
        if not file_data:
            raise ValueError("File data not found")
        
        # Save file temporarily for processing
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{doc_info['name']}") as tmp_file:
            tmp_file.write(file_data)
            tmp_file_path = tmp_file.name
        
        try:
            # Get processing options from session state
            processing_model = st.session_state.get('processing_model', 'Ensemble Method (All Models)')
            
            # Map UI model names to internal model names
            model_mapping = {
                'Ensemble Method (All Models)': ["rule_based", "ner", "layout_aware"],
                'Rule-Based Extractor': ["rule_based"],
                'spaCy NER': ["ner"],
                'Transformers NER': ["ner"],
                'Layout-Aware NER': ["layout_aware"]
            }
            
            enabled_models = model_mapping.get(processing_model, ["rule_based", "ner", "layout_aware"])
            
            # Use actual PII pipeline with selected models
            pipeline = PIIExtractionPipeline(models=enabled_models)
            extraction_result = pipeline.extract_from_file(tmp_file_path)
            
            # Get confidence threshold from session state
            confidence_threshold = st.session_state.get('confidence_threshold', 0.5)
            
            # Convert PII entities to expected format and filter by confidence
            pii_results = []
            for entity in extraction_result.pii_entities:
                if entity.confidence >= confidence_threshold:
                    pii_results.append({
                        'type': entity.pii_type.upper(),
                        'text': entity.text,
                        'start': entity.start_pos,
                        'end': entity.end_pos,
                        'confidence': entity.confidence,
                        'context': entity.context,
                        'extractor': entity.extractor
                    })
            
            # Extract text content for display
            doc_processor = DocumentProcessor()
            password = st.session_state.get('document_password', '')
            processed_doc = doc_processor.process_document(tmp_file_path, password if password else None)
            text_content = processed_doc.get('raw_text', '')
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
        # Store results
        results = {
            'text_content': text_content,
            'pii_entities': pii_results,
            'processing_method': 'real_pipeline',
            'processing_time': extraction_result.processing_time,
            'confidence_scores': extraction_result.confidence_scores,
            'confidence_threshold': confidence_threshold,
            'total_entities': len(pii_results),
            'selected_model': processing_model,
            'enabled_models': enabled_models,
            'total_entities_before_filter': len(extraction_result.pii_entities)
        }
        
        session_state.store_processing_results(file_id, results)
        session_state.update_document_status(file_id, 'completed')
        
        st.success(f"✅ Processing completed for {doc_info['name']}")
        st.rerun()
        
    except Exception as e:
        session_state.update_document_status(file_id, 'error', error_message=str(e))
        st.error(f"❌ Processing failed: {str(e)}")

# Removed mock functions - now using real PII pipeline

def show_processing_results():
    """Display processing results for current document"""
    current_doc_id = st.session_state.get('current_document')
    if not current_doc_id:
        return
    
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Processing Results</div>', 
                unsafe_allow_html=True)
    
    # Get document info and results
    doc_info = session_state.get_document_info(current_doc_id)
    results = session_state.get_processing_results(current_doc_id)
    
    if not doc_info:
        st.error("Document not found")
        return
    
    if not results:
        st.info(f"No processing results for {doc_info['name']}. Please process the document first.")
        return
    
    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document View", "📋 PII Entities", "📊 Statistics", "⚙️ Export"])
    
    with tab1:
        show_document_view(results)
    
    with tab2:
        show_pii_entities(results)
    
    with tab3:
        show_statistics(results)
    
    with tab4:
        show_export_options(current_doc_id, results)

def show_document_view(results: Dict[str, Any]):
    """Show document with PII highlighting"""
    st.markdown("### Document with PII Highlighting")
    
    text_content = results.get('text_content', '')
    pii_entities = results.get('pii_entities', [])
    
    if not text_content:
        st.warning("No text content available")
        return
    
    # Confidence threshold filter
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=results.get('confidence_threshold', 0.5),
        step=0.05
    )
    
    # Filter entities by confidence
    filtered_entities = [
        entity for entity in pii_entities 
        if entity.get('confidence', 0) >= confidence_threshold
    ]
    
    # Show highlighted text
    if filtered_entities:
        highlighted_html = ui_components.display_pii_highlights(text_content, filtered_entities)
        st.markdown(highlighted_html, unsafe_allow_html=True)
    else:
        st.text_area("Document Text", text_content, height=400)

def show_pii_entities(results: Dict[str, Any]):
    """Show PII entities table"""
    st.markdown("### Detected PII Entities")
    
    pii_entities = results.get('pii_entities', [])
    total_entities = results.get('total_entities', 0)
    processing_method = results.get('processing_method', 'unknown')
    processing_time = results.get('processing_time', 0)
    
    # Show processing summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Entities", total_entities)
    with col2:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    with col3:
        st.metric("Method", processing_method)
    with col4:
        selected_model = results.get('selected_model', 'Unknown')
        st.metric("Model", selected_model.split(' (')[0])
    
    if not pii_entities:
        st.info("No PII entities detected")
        return
    
    # Convert to DataFrame for display
    import pandas as pd
    
    entities_data = []
    for entity in pii_entities:
        entities_data.append({
            'Type': entity.get('type', ''),
            'Text': entity.get('text', ''),
            'Confidence': f"{entity.get('confidence', 0):.2%}",
            'Position': f"{entity.get('start', 0)}-{entity.get('end', 0)}"
        })
    
    df = pd.DataFrame(entities_data)
    st.dataframe(df, use_container_width=True)

def show_statistics(results: Dict[str, Any]):
    """Show processing statistics"""
    st.markdown("### Processing Statistics")
    
    pii_entities = results.get('pii_entities', [])
    
    if not pii_entities:
        st.info("No statistics available")
        return
    
    # Count by category
    category_counts = {}
    confidence_scores = []
    
    for entity in pii_entities:
        entity_type = entity.get('type', 'UNKNOWN')
        category_counts[entity_type] = category_counts.get(entity_type, 0) + 1
        confidence_scores.append(entity.get('confidence', 0))
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total PII Entities", len(pii_entities))
    
    with col2:
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
    
    with col3:
        st.metric("PII Categories", len(category_counts))
    
    with col4:
        selected_model = results.get('selected_model', 'Unknown')
        st.metric("Model Used", selected_model.split(' (')[0])  # Remove the "(All Models)" part
    
    # Show filtering information if applicable
    total_before_filter = results.get('total_entities_before_filter', len(pii_entities))
    confidence_threshold = results.get('confidence_threshold', 0.5)
    
    if total_before_filter > len(pii_entities):
        filtered_count = total_before_filter - len(pii_entities)
        st.info(f"📊 Showing {len(pii_entities)} entities (filtered out {filtered_count} below {confidence_threshold:.0%} confidence)")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution
        if category_counts:
            fig = ui_components.create_pii_category_chart(category_counts)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Confidence distribution
        if confidence_scores:
            fig = ui_components.create_confidence_histogram(confidence_scores)
            st.plotly_chart(fig, use_container_width=True)

def show_export_options(file_id: str, results: Dict[str, Any]):
    """Show export options for results"""
    st.markdown("### Export Results")
    
    if not auth.has_permission('write'):
        st.warning("Export requires write permissions")
        return
    
    doc_info = session_state.get_document_info(file_id)
    filename_base = f"pii_results_{doc_info['name'].split('.')[0]}" if doc_info else "pii_results"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export
        ui_components.export_data(results, filename_base, 'json')
    
    with col2:
        # CSV export (entities only)
        pii_entities = results.get('pii_entities', [])
        if pii_entities:
            import pandas as pd
            df = pd.DataFrame(pii_entities)
            ui_components.export_data(df, f"{filename_base}_entities", 'csv')
    
    with col3:
        # Text export (highlighted)
        text_content = results.get('text_content', '')
        ui_components.export_data(text_content, f"{filename_base}_text", 'txt')

def delete_document(file_id: str):
    """Delete document and its results"""
    if not auth.has_permission('delete'):
        st.error("Delete operation requires delete permissions")
        return
    
    # Remove from session state
    if file_id in st.session_state.uploaded_documents:
        del st.session_state.uploaded_documents[file_id]
    
    if file_id in st.session_state.processing_results:
        del st.session_state.processing_results[file_id]
    
    if hasattr(st.session_state, 'file_data') and file_id in st.session_state.file_data:
        del st.session_state.file_data[file_id]
    
    # Clear current document if it was deleted
    if st.session_state.get('current_document') == file_id:
        st.session_state.current_document = None
    
    st.success("Document deleted successfully")
    st.rerun()