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
    supported_formats = config.get('supported_formats', ['pdf', 'docx', 'jpg', 'png'])
    max_size_mb = config.get('max_file_size_mb', 50)
    
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
    
    # Auto-process if user has write permissions
    if auth.has_permission('write'):
        if st.button(f"Process {uploaded_file.name}", key=f"process_{file_id}"):
            process_document(file_id)

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
        
        # Extract text based on file type
        text_content = extract_text_content(file_data, doc_info['type'])
        
        # Mock PII extraction (integrate with actual pipeline)
        pii_results = mock_pii_extraction(text_content)
        
        # Store results
        results = {
            'text_content': text_content,
            'pii_entities': pii_results,
            'processing_method': 'mock_extraction',
            'confidence_threshold': st.session_state.get('confidence_threshold', 0.5)
        }
        
        session_state.store_processing_results(file_id, results)
        session_state.update_document_status(file_id, 'completed')
        
        st.success(f"✅ Processing completed for {doc_info['name']}")
        st.rerun()
        
    except Exception as e:
        session_state.update_document_status(file_id, 'error', error_message=str(e))
        st.error(f"❌ Processing failed: {str(e)}")

def extract_text_content(file_data: bytes, file_type: str) -> str:
    """Extract text content from file data"""
    try:
        if 'pdf' in file_type.lower():
            # PDF extraction
            pdf_file = io.BytesIO(file_data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif 'docx' in file_type.lower() or 'word' in file_type.lower():
            # DOCX extraction
            docx_file = io.BytesIO(file_data)
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif any(img_type in file_type.lower() for img_type in ['jpg', 'jpeg', 'png']):
            # Image OCR (simplified - would use actual OCR in production)
            return "OCR extraction would be implemented here for image files"
        
        else:
            return "Unsupported file type for text extraction"
    
    except Exception as e:
        raise ValueError(f"Text extraction failed: {str(e)}")

def mock_pii_extraction(text: str) -> List[Dict[str, Any]]:
    """Mock PII extraction for demonstration"""
    import re
    
    pii_entities = []
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        pii_entities.append({
            'type': 'EMAIL',
            'text': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': 0.95
        })
    
    # Phone pattern
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    for match in re.finditer(phone_pattern, text):
        pii_entities.append({
            'type': 'PHONE',
            'text': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': 0.85
        })
    
    # Simple name pattern (capitalized words)
    name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    for match in re.finditer(name_pattern, text):
        # Skip common false positives
        if match.group() not in ['Dear Sir', 'John Doe', 'Jane Doe']:
            pii_entities.append({
                'type': 'PERSON',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.70
            })
    
    return pii_entities

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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total PII Entities", len(pii_entities))
    
    with col2:
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
    
    with col3:
        st.metric("PII Categories", len(category_counts))
    
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