"""
Enhanced Document Processing Page with Pure GPT Approach

This page provides the main interface for uploading documents, processing them
using multimodal LLMs for PII extraction, and viewing results with cost tracking.
"""

import streamlit as st
import uuid
import io
import base64
from PIL import Image
import PyPDF2
from docx import Document
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
import tempfile
import os
import time
import json

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

# Load environment variables
project_root = src_path.parent
env_loader_path = project_root / "load_env.py"
if env_loader_path.exists():
    sys.path.insert(0, str(project_root))
    try:
        from load_env import load_env_file
        load_env_file()
    except ImportError:
        pass

from dashboard.utils import session_state, ui_components, auth, run_history
from llm.multimodal_llm_service import llm_service
from utils.document_processor import DocumentProcessor

def show_page():
    """Main document processing page with LLM selection"""
    st.markdown('<div class="section-header">🤖 AI-Powered Document Processing</div>', 
                unsafe_allow_html=True)
    st.markdown("Upload and process documents using advanced multimodal LLMs for PII extraction.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Two column layout for main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        show_llm_selection_and_upload()
    
    with col2:
        show_document_list()
    
    # Full width results section
    show_processing_results()

def show_llm_selection_and_upload():
    """LLM selection and document upload interface"""
    st.markdown("### 🧠 Select AI Model")
    
    # Get available models
    available_models = llm_service.get_available_models()
    
    if not available_models:
        st.error("❌ No multimodal LLM models available!")
        st.info("Please configure API keys for OpenAI, Anthropic, or Google in your environment variables.")
        return
    
    # Model selection with detailed info
    selected_model = st.selectbox(
        "Choose AI Model",
        available_models,
        format_func=lambda x: format_model_display(x),
        help="Select which AI model to use for document processing"
    )
    
    # Show model information
    if selected_model:
        model_info = llm_service.get_model_info(selected_model)
        show_model_info_card(model_info, selected_model)
    
    st.markdown("---")
    
    # Processing settings
    st.markdown("### ⚙️ Processing Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        document_type = st.selectbox(
            "Document Type",
            ["document", "form", "resume", "check", "medical_record", "legal_document", "invoice"],
            help="Document type helps optimize the AI prompt for better extraction"
        )
        
        max_cost_per_doc = st.slider(
            "Max Cost per Document ($)",
            0.01, 1.00, 0.10, 0.01,
            help="Processing will stop if cost exceeds this limit"
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0, 1.0, 0.85, 0.05,
            help="Minimum confidence for PII entities"
        )
        
        enable_cost_tracking = st.checkbox(
            "Enable Detailed Cost Tracking",
            value=True,
            help="Track costs and usage for analysis"
        )
    
    # Store settings in session state
    st.session_state.selected_llm_model = selected_model
    st.session_state.document_type = document_type
    st.session_state.max_cost_per_doc = max_cost_per_doc
    st.session_state.confidence_threshold = confidence_threshold
    st.session_state.enable_cost_tracking = enable_cost_tracking
    
    st.markdown("---")
    
    # Document upload
    st.markdown("### 📁 Upload Documents")
    
    # Get configuration
    config = st.session_state.get('app_config', {})
    supported_formats = ['pdf', 'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'docx', 'doc']
    max_size_mb = config.get('max_file_size_mb', 50)
    
    # Password input for protected documents
    document_password = st.text_input(
        "Document Password (if required)",
        type="password",
        value="Hubert",
        help="Leave blank for unprotected documents"
    )
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose documents to process",
        type=supported_formats,
        accept_multiple_files=True,
        help=f"Supported formats: {', '.join(supported_formats)}"
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
                    process_uploaded_file(uploaded_file, document_password)

def format_model_display(model_key: str) -> str:
    """Format model display name"""
    provider, model = model_key.split('/', 1)
    provider_names = {
        "openai": "🔵 OpenAI",
        "anthropic": "🟡 Anthropic", 
        "google": "🔴 Google"
    }
    
    return f"{provider_names.get(provider, provider)} - {model}"

def show_model_info_card(model_info: Dict[str, Any], model_key: str):
    """Display model information card"""
    if not model_info.get("available"):
        st.error(f"Model {model_key} is not available")
        return
    
    with st.container():
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <h4>📊 Model Information</h4>
            <p><strong>Provider:</strong> {model_info['provider'].title()}</p>
            <p><strong>Model:</strong> {model_info['model']}</p>
            <p><strong>Cost per 1K tokens:</strong> Input: ${model_info['cost_per_1k_input_tokens']:.4f} | Output: ${model_info['cost_per_1k_output_tokens']:.4f}</p>
            <p><strong>Capabilities:</strong> 🖼️ Images, 📄 OCR, 🔍 PII Extraction</p>
        </div>
        """, unsafe_allow_html=True)

def process_uploaded_file(uploaded_file, password: str = ""):
    """Process and store uploaded file"""
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Extract file info
    file_info = {
        'name': uploaded_file.name,
        'size': uploaded_file.size,
        'type': uploaded_file.type,
        'file_id': file_id,
        'password': password
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
    st.info("📝 Document uploaded successfully. Click 'Process with AI' to analyze for PII.")

def show_document_list():
    """Display uploaded documents with AI processing options"""
    st.markdown("### 📄 Uploaded Documents")
    
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
                    st.spinner("🤖 Processing with AI...")
                elif doc_info.get('status') == 'completed':
                    st.success("✅ AI processing completed")
                    # Show cost if available
                    results = session_state.get_processing_results(file_id)
                    if results and 'cost' in results:
                        st.info(f"💰 Cost: ${results['cost']:.4f}")
                elif doc_info.get('status') == 'error':
                    st.error("❌ AI processing failed")
                    if doc_info.get('error_message'):
                        st.error(f"Error: {doc_info['error_message']}")
            
            with col2:
                if auth.has_permission('write'):
                    # Check if model is selected
                    selected_model = st.session_state.get('selected_llm_model')
                    if selected_model:
                        if st.button("🤖 Process with AI", key=f"btn_process_{file_id}"):
                            process_document_with_llm(file_id)
                    else:
                        st.warning("Select AI model first")
                
                if st.button("👁️ View", key=f"btn_view_{file_id}"):
                    st.session_state.current_document = file_id
                    st.rerun()
                
                if auth.has_permission('delete'):
                    if st.button("🗑️ Delete", key=f"btn_delete_{file_id}"):
                        delete_document(file_id)

def convert_document_to_image(file_data: bytes, file_name: str, password: str = "") -> Optional[str]:
    """Convert document to base64 image for LLM processing"""
    try:
        st.info(f"🔄 Converting {file_name} to image...")
        file_ext = Path(file_name).suffix.lower()
        st.info(f"📎 File extension: {file_ext}")
        
        if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            # Already an image
            image = Image.open(io.BytesIO(file_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            if max(image.size) > 2048:
                image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        
        elif file_ext == '.pdf':
            # Convert PDF to image using pdf2image
            try:
                from pdf2image import convert_from_bytes
                
                # Try with password first
                if password:
                    pages = convert_from_bytes(file_data, dpi=200, userpw=password, first_page=1, last_page=1)
                else:
                    pages = convert_from_bytes(file_data, dpi=200, first_page=1, last_page=1)
                
                if pages:
                    image = pages[0]
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Resize if needed
                    if max(image.size) > 2048:
                        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                    
                    buffer = io.BytesIO()
                    image.save(buffer, format='PNG')
                    return base64.b64encode(buffer.getvalue()).decode()
                
            except Exception as e:
                st.error(f"❌ Failed to convert PDF {file_name}: {e}")
                import traceback
                st.code(traceback.format_exc())
                return None
        
        elif file_ext in ['.doc', '.docx']:
            # For Word documents, we'd need to extract text and create an image
            # For now, we'll skip these as they're text-based
            st.warning(f"Word documents not supported for LLM processing yet: {file_name}")
            return None
        
        else:
            st.warning(f"Unsupported file format for LLM processing: {file_ext}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error converting document {file_name} to image: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def process_document_with_llm(file_id: str):
    """Process document using selected LLM model"""
    try:
        # Debug info
        st.info(f"🔍 Starting processing for file ID: {file_id}")
        
        doc_info = session_state.get_document_info(file_id)
        if not doc_info:
            st.error("❌ Document not found in session state")
            return
        
        st.info(f"📄 Document info: {doc_info['name']} ({doc_info['size']} bytes)")
        
        # Get processing settings
        selected_model = st.session_state.get('selected_llm_model')
        document_type = st.session_state.get('document_type', 'document')
        max_cost = st.session_state.get('max_cost_per_doc', 0.10)
        confidence_threshold = st.session_state.get('confidence_threshold', 0.85)
        enable_cost_tracking = st.session_state.get('enable_cost_tracking', True)
        
        st.info(f"🤖 Selected model: {selected_model}")
        st.info(f"📋 Document type: {document_type}")
        
        if not selected_model:
            st.error("❌ No LLM model selected")
            return
        
        # Check if model is available
        available_models = llm_service.get_available_models()
        if selected_model not in available_models:
            st.error(f"❌ Model {selected_model} is not available. Available models: {available_models}")
            return
        
        # Update status
        session_state.update_document_status(file_id, 'processing')
        # Note: Removed st.rerun() to prevent processing interruption
        
    except Exception as e:
        st.error(f"❌ Error in initial setup: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return
    
    try:
        # Get file data
        st.info("📁 Retrieving file data from session...")
        file_data = st.session_state.file_data.get(file_id)
        if not file_data:
            raise ValueError(f"File data not found for ID: {file_id}. Available IDs: {list(st.session_state.file_data.keys()) if hasattr(st.session_state, 'file_data') else 'No file_data in session'}")
        
        st.info(f"✅ File data retrieved: {len(file_data)} bytes")
        
        # Convert document to image
        st.info("🖼️ Converting document to image...")
        password = doc_info.get('password', '')
        image_data = convert_document_to_image(file_data, doc_info['name'], password)
        
        if not image_data:
            raise ValueError("Could not convert document to image format")
        
        st.info(f"✅ Document converted to image: {len(image_data)} chars base64")
        
        # Process with LLM
        st.info(f"🤖 Starting LLM processing with {selected_model}...")
        start_time = time.time()
        
        with st.spinner(f"🤖 Processing with {selected_model}..."):
            result = llm_service.extract_pii_from_image(
                image_data=image_data,
                model_key=selected_model,
                document_type=document_type,
                max_tokens=4000,
                temperature=0.0
            )
        
        st.info(f"📊 LLM processing result: {result.get('success', False)}")
        if not result.get('success'):
            st.error(f"❌ LLM Error: {result.get('error', 'Unknown error')}")
        
        processing_time = time.time() - start_time
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            session_state.update_document_status(file_id, 'error', error_message=error_msg)
            st.error(f"❌ LLM processing failed: {error_msg}")
            return
        
        # Check cost limit
        estimated_cost = result.get("usage", {}).get("estimated_cost", 0)
        if estimated_cost > max_cost:
            error_msg = f"Cost ${estimated_cost:.4f} exceeds limit ${max_cost:.4f}"
            session_state.update_document_status(file_id, 'error', error_message=error_msg)
            st.error(f"❌ {error_msg}")
            return
        
        # Filter entities by confidence
        pii_entities = result.get("pii_entities", [])
        filtered_entities = [
            entity for entity in pii_entities 
            if entity.get("confidence", 0) >= confidence_threshold
        ]
        
        # Format results for storage
        processing_results = {
            'text_content': result.get("transcribed_text", ""),
            'pii_entities': filtered_entities,
            'processing_method': 'multimodal_llm',
            'processing_time': processing_time,
            'total_entities': len(filtered_entities),
            'total_entities_before_filter': len(pii_entities),
            'confidence_threshold': confidence_threshold,
            'model_used': selected_model,
            'document_type': document_type,
            'cost': estimated_cost,
            'usage': result.get("usage", {}),
            'extraction_method': result.get("extraction_method", "pure_llm"),
            'structured_data': result.get("structured_data"),
            'llm_response': result.get("content", "")
        }
        
        # Store results
        session_state.store_processing_results(file_id, processing_results)
        session_state.update_document_status(file_id, 'completed')
        
        # Track in run history if enabled
        if enable_cost_tracking:
            document_result = {
                "document_name": doc_info['name'],
                "document_size": doc_info['size'],
                "processing_time": processing_time,
                "cost": estimated_cost,
                "entities_found": len(filtered_entities),
                "success": True,
                "entities": filtered_entities,
                "usage": result.get("usage", {}),
                "metadata": {
                    "model_used": selected_model,
                    "document_type": document_type,
                    "extraction_method": result.get("extraction_method")
                }
            }
            
            # Calculate average confidence
            confidences = [e.get("confidence", 0) for e in filtered_entities]
            if confidences:
                document_result["average_confidence"] = sum(confidences) / len(confidences)
            
            # Save to run history
            run_id = run_history.run_history_manager.save_run(
                run_type="single",
                model_used=selected_model,
                document_results=[document_result],
                settings={
                    "document_type": document_type,
                    "confidence_threshold": confidence_threshold,
                    "max_cost_per_doc": max_cost
                },
                user_id=st.session_state.get('username')
            )
            
            st.session_state[f"run_id_{file_id}"] = run_id
        
        st.success(f"✅ AI processing completed! Found {len(filtered_entities)} PII entities. Cost: ${estimated_cost:.4f}")
        # Note: Removed st.rerun() to prevent page refresh during processing
        
    except Exception as e:
        session_state.update_document_status(file_id, 'error', error_message=str(e))
        st.error(f"❌ Processing failed: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        
        # Additional debugging info
        st.markdown("### 🔍 Debug Information")
        st.write(f"**File ID:** {file_id}")
        st.write(f"**Session state keys:** {list(st.session_state.keys())}")
        st.write(f"**File data available:** {hasattr(st.session_state, 'file_data')}")
        if hasattr(st.session_state, 'file_data'):
            st.write(f"**File data keys:** {list(st.session_state.file_data.keys())}")
        st.write(f"**Available models:** {llm_service.get_available_models()}")
        st.write(f"**Selected model:** {st.session_state.get('selected_llm_model')}")

def show_processing_results():
    """Display processing results for current document"""
    current_doc_id = st.session_state.get('current_document')
    if not current_doc_id:
        return
    
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 AI Processing Results</div>', 
                unsafe_allow_html=True)
    
    # Get document info and results
    doc_info = session_state.get_document_info(current_doc_id)
    results = session_state.get_processing_results(current_doc_id)
    
    if not doc_info:
        st.error("Document not found")
        return
    
    if not results:
        st.info(f"No AI processing results for {doc_info['name']}. Please process the document first.")
        return
    
    # Show processing summary
    show_processing_summary(results)
    
    # Results tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Document View", 
        "📋 PII Entities", 
        "📊 Analysis", 
        "🤖 AI Response",
        "💾 Export"
    ])
    
    with tab1:
        show_document_view(results)
    
    with tab2:
        show_pii_entities(results)
    
    with tab3:
        show_analysis(results)
    
    with tab4:
        show_ai_response(results)
    
    with tab5:
        show_export_options(current_doc_id, results)

def show_processing_summary(results: Dict[str, Any]):
    """Show processing summary metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🤖 Model", results.get('model_used', '').split('/')[-1])
    
    with col2:
        entities_count = results.get('total_entities', 0)
        st.metric("🔍 PII Found", entities_count)
    
    with col3:
        cost = results.get('cost', 0)
        st.metric("💰 Cost", f"${cost:.4f}")
    
    with col4:
        processing_time = results.get('processing_time', 0)
        st.metric("⏱️ Time", f"{processing_time:.2f}s")
    
    with col5:
        extraction_method = results.get('extraction_method', 'unknown')
        method_display = {
            'pure_llm_structured': '🎯 Structured',
            'pure_llm_unstructured': '📝 Text Only',
            'multimodal_llm': '🤖 AI Vision'
        }.get(extraction_method, extraction_method)
        st.metric("📊 Method", method_display)

def show_document_view(results: Dict[str, Any]):
    """Show document with PII highlighting"""
    st.markdown("### 📄 Document with AI-Detected PII")
    
    text_content = results.get('text_content', '')
    pii_entities = results.get('pii_entities', [])
    
    if not text_content:
        st.warning("No text content available from AI processing")
        return
    
    # Confidence threshold filter
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=results.get('confidence_threshold', 0.85),
        step=0.05,
        help="Filter PII entities by confidence level"
    )
    
    # Filter entities by confidence
    filtered_entities = [
        entity for entity in pii_entities 
        if entity.get('confidence', 0) >= confidence_threshold
    ]
    
    st.info(f"📊 Showing {len(filtered_entities)} PII entities above {confidence_threshold:.0%} confidence")
    
    # Show highlighted text
    if filtered_entities:
        # Create a simple highlighting by showing entities in a table
        st.markdown("#### 🔍 Extracted Text Content")
        st.text_area(
            "AI Transcribed Text", 
            text_content, 
            height=300,
            help="This is the text that the AI model extracted from your document"
        )
        
        st.markdown("#### 🎯 Detected PII Entities")
        for i, entity in enumerate(filtered_entities[:10]):  # Show first 10
            entity_type = entity.get('type', 'UNKNOWN')
            entity_text = entity.get('text', '')
            confidence = entity.get('confidence', 0)
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid #1f77b4;">
                <strong>{entity_type}:</strong> {entity_text} 
                <span style="color: #666; font-size: 0.9em;">(Confidence: {confidence:.0%})</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.text_area("Document Text", text_content, height=400)

def show_pii_entities(results: Dict[str, Any]):
    """Show PII entities table with AI insights"""
    st.markdown("### 🔍 AI-Detected PII Entities")
    
    pii_entities = results.get('pii_entities', [])
    
    if not pii_entities:
        st.info("No PII entities detected by the AI model")
        return
    
    # Convert to DataFrame for display
    import pandas as pd
    
    entities_data = []
    for entity in pii_entities:
        entities_data.append({
            'Type': entity.get('type', ''),
            'Text': entity.get('text', ''),
            'Confidence': f"{entity.get('confidence', 0):.1%}",
            'Source': entity.get('source', 'llm_extraction')
        })
    
    df = pd.DataFrame(entities_data)
    
    # Group by type
    st.markdown("#### 📊 PII Categories Summary")
    category_counts = df['Type'].value_counts()
    
    cols = st.columns(min(len(category_counts), 4))
    for i, (category, count) in enumerate(category_counts.items()):
        with cols[i % 4]:
            st.metric(category, count)
    
    st.markdown("#### 📋 Detailed Entities")
    st.dataframe(df, use_container_width=True)
    
    # Show filtering stats
    total_before_filter = results.get('total_entities_before_filter', len(pii_entities))
    confidence_threshold = results.get('confidence_threshold', 0.85)
    
    if total_before_filter > len(pii_entities):
        filtered_count = total_before_filter - len(pii_entities)
        st.info(f"📊 Showing {len(pii_entities)} entities (filtered out {filtered_count} below {confidence_threshold:.0%} confidence)")

def show_analysis(results: Dict[str, Any]):
    """Show detailed analysis and statistics"""
    st.markdown("### 📊 Processing Analysis")
    
    # Model performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🤖 AI Model Performance")
        
        processing_time = results.get('processing_time', 0)
        cost = results.get('cost', 0)
        usage = results.get('usage', {})
        
        metrics_data = {
            "Processing Time": f"{processing_time:.2f} seconds",
            "Total Cost": f"${cost:.4f}",
            "Input Tokens": f"{usage.get('prompt_tokens', 0):,}",
            "Output Tokens": f"{usage.get('completion_tokens', 0):,}",
            "Total Tokens": f"{usage.get('total_tokens', 0):,}"
        }
        
        for metric, value in metrics_data.items():
            st.metric(metric, value)
    
    with col2:
        st.markdown("#### 🎯 Extraction Quality")
        
        pii_entities = results.get('pii_entities', [])
        if pii_entities:
            confidences = [e.get('confidence', 0) for e in pii_entities]
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            max_confidence = max(confidences)
            
            quality_metrics = {
                "Entities Found": len(pii_entities),
                "Average Confidence": f"{avg_confidence:.1%}",
                "Min Confidence": f"{min_confidence:.1%}",
                "Max Confidence": f"{max_confidence:.1%}",
                "Extraction Method": results.get('extraction_method', 'unknown')
            }
            
            for metric, value in quality_metrics.items():
                st.metric(metric, value)
    
    # Show structured data if available
    structured_data = results.get('structured_data')
    if structured_data:
        st.markdown("#### 🗂️ Structured AI Response")
        st.json(structured_data)

def show_ai_response(results: Dict[str, Any]):
    """Show raw AI response"""
    st.markdown("### 🤖 Raw AI Response")
    
    llm_response = results.get('llm_response', '')
    model_used = results.get('model_used', 'unknown')
    
    st.info(f"Response from: {model_used}")
    
    if llm_response:
        st.text_area(
            "AI Model Response",
            llm_response,
            height=400,
            help="This is the raw response from the AI model before processing"
        )
        
        # Show response analysis
        st.markdown("#### 📊 Response Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Response Length", f"{len(llm_response)} chars")
        
        with col2:
            word_count = len(llm_response.split())
            st.metric("Word Count", word_count)
        
        with col3:
            extraction_method = results.get('extraction_method', 'unknown')
            st.metric("Parsing", "✅ JSON" if "structured" in extraction_method else "📝 Text")
    else:
        st.warning("No AI response available")

def show_export_options(file_id: str, results: Dict[str, Any]):
    """Show export options for results"""
    st.markdown("### 💾 Export Options")
    
    if not auth.has_permission('write'):
        st.warning("Export requires write permissions")
        return
    
    doc_info = session_state.get_document_info(file_id)
    filename_base = f"ai_pii_results_{doc_info['name'].split('.')[0]}" if doc_info else "ai_pii_results"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export (complete results)
        if st.button("📄 Export JSON"):
            export_data = {
                "document_info": doc_info,
                "processing_results": results,
                "export_timestamp": time.time(),
                "export_format": "json"
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_str,
                file_name=f"{filename_base}.json",
                mime="application/json"
            )
    
    with col2:
        # CSV export (entities only)
        pii_entities = results.get('pii_entities', [])
        if pii_entities and st.button("📊 Export CSV"):
            import pandas as pd
            df = pd.DataFrame(pii_entities)
            csv_str = df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_str,
                file_name=f"{filename_base}_entities.csv",
                mime="text/csv"
            )
    
    with col3:
        # Text export (transcribed content)
        text_content = results.get('text_content', '')
        if text_content and st.button("📝 Export Text"):
            st.download_button(
                label="⬇️ Download Text",
                data=text_content,
                file_name=f"{filename_base}_text.txt",
                mime="text/plain"
            )
    
    # Show run history link if available
    run_id = st.session_state.get(f"run_id_{file_id}")
    if run_id:
        st.markdown("---")
        st.info(f"🗂️ **Run ID:** {run_id}")
        st.markdown("View this processing run in the Model Comparison page for detailed analytics.")

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

if __name__ == "__main__":
    show_page()