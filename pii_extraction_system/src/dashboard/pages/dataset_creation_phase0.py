"""
Phase 0 Dataset Creation - Ground Truth Labeling Interface

This module provides the Phase 0 dataset creation functionality including:
- File upload interface for documents
- GPT-4o integration for ground truth labeling
- Metadata tagging system (document type, difficulty level, domain)
- Export labeled dataset functionality
- Interactive labeling interface with validation tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
import uuid
import logging
import tempfile
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
from PIL import Image
import sys
from pathlib import Path

# For PDF to image conversion
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# For password-protected documents
try:
    import msoffcrypto
    MSOFFCRYPTO_AVAILABLE = True
except ImportError:
    MSOFFCRYPTO_AVAILABLE = False

# For Office documents
try:
    from docx import Document
    import openpyxl
    OFFICE_AVAILABLE = True
except ImportError:
    OFFICE_AVAILABLE = False

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import session_state, ui_components, auth
from llm.multimodal_llm_service import llm_service
from utils.document_processor import DocumentProcessor
from core.ground_truth_validation import ground_truth_validator
from core.logging_config import get_logger
from utils.s3_integration import S3DocumentProcessor, process_s3_bucket_to_phase0, convert_document_to_images

# Import the new contextual PII pipeline
try:
    from extractors.contextual_pii_pipeline import ContextualPIIPipeline, DocumentDomain, ConfidenceLevel
    CONTEXTUAL_PII_AVAILABLE = True
except ImportError as e:
    st.error(f"Contextual PII Pipeline not available: {e}")
    CONTEXTUAL_PII_AVAILABLE = False
from core.config import settings

# Initialize logger with appropriate level for batch processing
logger = get_logger(__name__)

# Context manager for suppressing verbose logs during batch operations
class SuppressVerboseLogs:
    """Context manager to reduce verbose output during batch processing"""
    def __init__(self):
        self.suppressed = False
        
    def __enter__(self):
        # For batch processing, we'll rely on the simplified UI updates
        # rather than trying to modify logger levels directly
        self.suppressed = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.suppressed = False

# Dataset creation state management
def initialize_session_state():
    """Initialize session state variables for phase 0 dataset creation"""
    if 'phase0_dataset' not in st.session_state:
        st.session_state.phase0_dataset = []

    if 'phase0_current_document' not in st.session_state:
        st.session_state.phase0_current_document = None

    if 'phase0_labeling_queue' not in st.session_state:
        st.session_state.phase0_labeling_queue = []

def show_page():
    """Main Phase 0 dataset creation page"""
    # Initialize session state first
    initialize_session_state()
    
    st.markdown('<div class="section-header">🎯 Phase 0 Dataset Creation</div>', 
                unsafe_allow_html=True)
    st.markdown("Create high-quality ground truth datasets with Claude assistance and metadata tagging.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Main tabs  
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📤 File Upload",
        "🗄️ S3 Batch Processing", 
        "🤖 Claude Labeling",
        "🎯 3-Step Contextual PII",
        "🏷️ Metadata Tagging",
        "⚡ Interactive Labeling",
        "📊 Dataset Export"
    ])
    
    with tab1:
        show_file_upload_interface()
    
    with tab2:
        show_s3_batch_processing_interface()
    
    with tab3:
        show_claude_labeling_interface()
    
    with tab4:
        show_contextual_pii_interface()
        
    with tab5:
        show_metadata_tagging_interface()
    
    with tab6:
        show_interactive_labeling_interface()
    
    with tab7:
        show_dataset_export_interface()

def show_file_upload_interface():
    """Show file upload interface with validation"""
    # Ensure session state is initialized
    initialize_session_state()
    
    st.markdown("### 📤 Document Upload & Ingestion")
    
    if not auth.has_permission('write'):
        st.warning("File upload requires write permissions.")
        return
    
    # Upload statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Uploaded", len(st.session_state.phase0_dataset))
    
    with col2:
        labeled_count = sum(1 for doc in st.session_state.phase0_dataset if doc.get('labeled', False))
        st.metric("Labeled Documents", labeled_count)
    
    with col3:
        pending_count = len(st.session_state.phase0_dataset) - labeled_count
        st.metric("Pending Labels", pending_count)
    
    with col4:
        if st.session_state.phase0_dataset:
            completion_rate = labeled_count / len(st.session_state.phase0_dataset)
            st.metric("Completion Rate", f"{completion_rate:.1%}")
        else:
            st.metric("Completion Rate", "0%")
    
    # File upload interface
    st.markdown("#### Upload Documents")
    st.info("💡 **Quick Upload**: Files can be uploaded without any LLM processing. Enable 'Auto-label with Claude' in Upload Options only if you want immediate labeling.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose document files",
            type=['pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'tiff', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, DOC, TXT, PNG, JPG, JPEG, TIFF, XLSX, XLS"
        )
    
    with col2:
        upload_options = st.expander("Upload Options")
        with upload_options:
            auto_label = st.checkbox(
                "Auto-label with Claude (Optional)", 
                value=False,
                help="Enable to automatically run Claude labeling on upload. You can always label documents later in the Claude Labeling tab."
            )
            batch_size = st.slider("Batch Size", 1, 10, 5)
            priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=1)
            
            # Password support for encrypted documents
            password = st.text_input(
                "Document Password (if required)",
                type="password",
                value="Hubert",
                help="Password for encrypted documents (PDF, DOCX, Excel)"
            )
    
    if uploaded_files:
        if st.button("Process Uploaded Files"):
            process_uploaded_files(uploaded_files, auto_label, batch_size, priority, password)
    
    # Document queue display
    st.markdown("#### Document Queue")
    
    if st.session_state.phase0_dataset:
        display_document_queue()
    else:
        st.info("No documents uploaded yet. Upload documents to get started.")
    
    # Bulk actions
    if st.session_state.phase0_dataset:
        st.markdown("#### Bulk Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Label All Unlabeled"):
                st.info("💡 Using default password 'Hubert' for bulk operations. Use Claude Labeling tab for custom passwords.")
                label_all_unlabeled_documents()
        
        with col2:
            if st.button("Clear All Documents"):
                st.warning("⚠️ This will permanently delete all uploaded documents!")
                if st.button("🗑️ Yes, Clear All Documents", type="secondary"):
                    st.session_state.phase0_dataset = []
                    st.success("All documents cleared!")
                    st.rerun()
        
        with col3:
            if st.button("Export Label Dataset", help="Export PII labels and metadata only (no document content)"):
                export_current_dataset()

def process_uploaded_files(uploaded_files, auto_label: bool, batch_size: int, priority: str, password: str = ""):
    """Process uploaded files and add to dataset"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Generate unique document ID
        doc_id = str(uuid.uuid4())
        
        # Read file content
        file_content = uploaded_file.read()
        
        # Create document record
        document = {
            'id': doc_id,
            'name': uploaded_file.name,
            'type': uploaded_file.type,
            'size': len(file_content),
            'content': base64.b64encode(file_content).decode(),
            'uploaded_at': datetime.now().isoformat(),
            'priority': priority,
            'labeled': False,
            'metadata': {
                'document_type': detect_document_type(uploaded_file.name),
                'difficulty_level': 'Unknown',
                'domain': 'Unknown'
            },
            'ground_truth_labels': [],
            'claude_labels': None,
            'validation_status': 'Pending'
        }
        
        # Add to dataset
        st.session_state.phase0_dataset.append(document)
        
        # Auto-label if requested
        if auto_label:
            try:
                claude_labels = auto_label_document(doc_id, password)
                document['claude_labels'] = claude_labels
                document['labeled'] = True
                document['validation_status'] = 'Auto-labeled'
            except Exception as e:
                st.warning(f"Auto-labeling failed for {uploaded_file.name}: {e}")
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text(f"Successfully processed {len(uploaded_files)} files!")
    st.success(f"Added {len(uploaded_files)} documents to the dataset.")

def detect_document_type(filename: str) -> str:
    """Detect document type from filename"""
    extension = filename.lower().split('.')[-1]
    
    type_mapping = {
        'pdf': 'PDF Document',
        'docx': 'Word Document',
        'doc': 'Word Document',
        'txt': 'Text Document',
        'xlsx': 'Excel Spreadsheet',
        'xls': 'Excel Spreadsheet',
        'png': 'Image',
        'jpg': 'Image',
        'jpeg': 'Image',
        'tiff': 'Image'
    }
    
    return type_mapping.get(extension, 'Unknown')

def auto_label_document(doc_id: str, password: str = "", model: str = "anthropic/claude-3-5-sonnet-20241022") -> Dict[str, Any]:
    """Auto-label document using Claude vision - direct image processing approach"""
    document = next((doc for doc in st.session_state.phase0_dataset if doc['id'] == doc_id), None)
    
    if not document:
        raise ValueError("Document not found")
    
    try:
        # Decode the base64 content
        content_str = document['content']
        if isinstance(content_str, str):
            # content_str is already a base64 string, decode it directly
            file_content = base64.b64decode(content_str)
        else:
            # content_str is bytes, decode to string first then base64 decode
            file_content = base64.b64decode(content_str.decode('utf-8'))
        file_extension = os.path.splitext(document['name'])[1]
        
        # Convert document to images for vision processing
        images = convert_document_to_images(file_content, file_extension, password)
        
        if not images:
            st.warning(f"⚠️ Could not convert {document['name']} to images for processing")
            return {
                'method': 'conversion_failed',
                'entities': [],
                'confidence_score': 0.0,
                'processing_time': 0,
                'cost': 0,
                'error': 'Document could not be converted to images'
            }
        
        # Process all images and combine results
        all_entities = []
        total_cost = 0.0
        total_time = 0.0
        all_transcribed_text = []
        document_classification = None
        
        # Create progress placeholder for multi-page documents
        progress_placeholder = st.empty() if len(images) > 1 else None
        
        for i, image_data in enumerate(images):
            # Show progress for multi-page documents only
            if progress_placeholder:
                progress_placeholder.text(f"Processing page {i+1}/{len(images)}...")
            
            # Extract PII using vision
            result = llm_service.extract_pii_from_image(
                image_data,
                model,
                document_type=document['metadata']['document_type']
            )
            
            if result.get('success'):
                # Add page information to entities
                page_entities = result.get('pii_entities', [])
                for entity in page_entities:
                    entity['page'] = i + 1
                    entity['source'] = 'gpt4o_vision'
                
                all_entities.extend(page_entities)
                total_cost += result.get('usage', {}).get('estimated_cost', 0)
                total_time += result.get('processing_time', 0)
                
                # Collect transcribed text
                if result.get('transcribed_text'):
                    page_text = f"Page {i+1}:\n{result['transcribed_text']}"
                    all_transcribed_text.append(page_text)
                
                # Extract document classification from first page
                if i == 0 and result.get('structured_data'):
                    classification = result['structured_data'].get('document_classification')
                    if classification:
                        document_classification = classification
            else:
                # Only log errors and warnings, not individual page status
                logger.warning(f"Failed to process page {i+1} of {document['name']}: {result.get('error', 'Unknown error')}")
                if len(images) == 1:  # Show warning only for single-page documents
                    st.warning(f"⚠️ Processing failed: {result.get('error', 'Unknown error')}")
        
        # Clear progress placeholder
        if progress_placeholder:
            progress_placeholder.empty()
        
        # Combine results
        combined_text = '\n\n'.join(all_transcribed_text)
        
        # Show concise summary only - no verbose details
        if all_entities:
            entity_types = [entity.get('type', 'unknown') for entity in all_entities]
            unique_types = list(set(entity_types))[:3]  # Show first 3 types
            types_display = ', '.join(unique_types)
            if len(set(entity_types)) > 3:
                types_display += '...'
            
            classification_info = ""
            if document_classification:
                domain = document_classification.get('domain', 'Unknown')
                difficulty = document_classification.get('difficulty_level', 'Unknown')
                classification_info = f" | {domain}/{difficulty}"
            
            st.success(f"✅ Found {len(all_entities)} entities ({types_display}) - ${total_cost:.4f}{classification_info}")
        else:
            st.info(f"ℹ️ No PII entities found - ${total_cost:.4f}")
        
        # Update document metadata with classification
        if document_classification:
            document['metadata']['difficulty_level'] = document_classification.get('difficulty_level', 'Unknown')
            document['metadata']['domain'] = document_classification.get('domain', 'Unknown')
            document['metadata']['domain_detail'] = document_classification.get('domain_detail', '')
            
            # Update in session state
            for doc in st.session_state.phase0_dataset:
                if doc['id'] == doc_id:
                    doc['metadata'].update(document['metadata'])
                    break
        
        return {
            'method': 'gpt4o_vision_direct',
            'entities': all_entities,
            'confidence_score': calculate_confidence_score(all_entities),
            'processing_time': total_time,
            'cost': total_cost,
            'pages_processed': len(images),
            'extracted_text': combined_text[:500] + "..." if len(combined_text) > 500 else combined_text,
            'full_transcribed_text': combined_text,
            'approach': 'Direct vision processing - no text extraction step',
            'document_classification': document_classification
        }
    
    except Exception as e:
        error_message = str(e)
        
        # Provide more specific error messages based on the error type
        if "pdf2image" in error_message.lower():
            user_error = "PDF processing failed. Install pdf2image: pip install pdf2image"
            suggestion = "Install required dependency or convert PDF to images manually"
        elif "password" in error_message.lower() or "incorrect" in error_message.lower():
            user_error = f"Password-protected document failed. Verify password 'Hubert' is correct."
            suggestion = "Check password spelling, try different password, or decrypt document manually"
        elif "no such file" in error_message.lower() or "not found" in error_message.lower():
            user_error = "File not found or corrupted during processing."
            suggestion = "Re-upload the document or check file integrity"
        elif "msoffcrypto" in error_message.lower():
            user_error = "Office document decryption failed. Install msoffcrypto: pip install msoffcrypto-tool"
            suggestion = "Install required dependency for Office document processing"
        elif "unsupported" in error_message.lower() or ".doc" in error_message.lower():
            user_error = "Unsupported file format. Convert .doc files to .docx format."
            suggestion = "Use Microsoft Word to save as .docx, or convert to PDF format"
        else:
            user_error = f"Processing error: {error_message}"
            suggestion = "Check document format and try re-uploading"
        
        st.error(f"❌ {user_error}")
        if suggestion:
            st.info(f"💡 {suggestion}")
        
        return {
            'method': 'vision_processing_failed',
            'entities': [],
            'confidence_score': 0.0,
            'processing_time': 0,
            'cost': 0,
            'error': user_error,
            'suggestion': suggestion,
            'processing_details': {
                'file_extension': os.path.splitext(document['name'])[1],
                'password_provided': bool(password),
                'error_type': 'vision_processing_failed'
            }
        }

def calculate_confidence_score(entities: List[Dict]) -> float:
    """Calculate overall confidence score for entities"""
    if not entities:
        return 0.0
    
    return sum(entity.get('confidence', 0) for entity in entities) / len(entities)

def display_document_queue():
    """Display the document queue with actions"""
    # Create dataframe for display
    queue_data = []
    for doc in st.session_state.phase0_dataset:
        # Determine status with more granular information
        if doc['labeled']:
            labels = doc.get('claude_labels', {})
            if labels.get('method') == 'unsupported_format':
                status = '⚠️ Unsupported Format'
            elif labels.get('method') == 'no_text_content':
                status = 'ℹ️ No Text Found'
            elif labels.get('method') == 'no_text_extracted':
                status = 'ℹ️ No Text Extracted'
            elif labels.get('method') == 'failed':
                status = '❌ Processing Failed'
            elif labels.get('entities'):
                status = f'✅ Labeled ({len(labels["entities"])} entities)'
            else:
                status = '✅ Labeled (0 entities)'
        else:
            status = '⏳ Pending'
            
        queue_data.append({
            'ID': doc['id'][:8] + '...',
            'Name': doc['name'],
            'Type': doc['metadata']['document_type'],
            'Size': f"{doc['size'] / 1024:.1f} KB",
            'Priority': doc['priority'],
            'Status': status,
            'Difficulty': doc['metadata']['difficulty_level'],
            'Domain': doc['metadata']['domain'],
            'Uploaded': doc['uploaded_at'][:19].replace('T', ' ')
        })
    
    df = pd.DataFrame(queue_data)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get unique status values from the data
        unique_statuses = df['Status'].unique().tolist()
        status_filter = st.multiselect(
            "Filter by Status",
            unique_statuses,
            default=unique_statuses
        )
    
    with col2:
        priority_filter = st.multiselect(
            "Filter by Priority",
            ['High', 'Medium', 'Low'],
            default=['High', 'Medium', 'Low']
        )
    
    with col3:
        type_filter = st.multiselect(
            "Filter by Type",
            df['Type'].unique().tolist(),
            default=df['Type'].unique().tolist()
        )
    
    # Apply filters
    filtered_df = df[
        (df['Status'].isin(status_filter)) &
        (df['Priority'].isin(priority_filter)) &
        (df['Type'].isin(type_filter))
    ]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Document selection for detailed view
    if not filtered_df.empty:
        selected_doc_name = st.selectbox(
            "Select document for detailed view:",
            ['None'] + filtered_df['Name'].tolist()
        )
        
        if selected_doc_name != 'None':
            show_document_details(selected_doc_name)

def show_s3_batch_processing_interface():
    """Scalable S3 batch processing interface for 10k+ documents"""
    
    st.markdown("### 🗄️ S3 Batch Processing")
    st.markdown("**Enterprise-scale processing**: Handle up to 10,000 documents with real-time progress tracking.")
    
    if not auth.has_permission('write'):
        st.warning("S3 batch processing requires write permissions.")
        return
    
    # Import scalable components
    from database.document_store import get_document_db
    from processing.background_processor import get_background_processor
    
    db = get_document_db()
    processor = get_background_processor()
    
    # Main interface tabs
    tab1, tab2, tab3 = st.tabs([
        "🚀 New Batch",
        "📊 Active Batches", 
        "📜 Batch History"
    ])
    
    with tab1:
        show_new_batch_interface(db, processor)
    
    with tab2:
        show_active_batches_interface(db, processor)
    
    with tab3:
        show_batch_history_interface(db)


def show_new_batch_interface(db, processor):
    """Interface for creating new processing batches"""
    
    st.markdown("#### 🆕 Create New Processing Batch")
    
    # S3 Configuration Section
    with st.expander("📋 S3 Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Connection Settings**")
            bucket_name = st.text_input(
                "S3 Bucket Name",
                value="", 
                help="Name of the S3 bucket containing documents"
            )
            
            aws_region = st.selectbox(
                "AWS Region",
                ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-1"],
                index=0
            )
            
            input_prefix = st.text_input(
                "Input Prefix",
                value="documents/",
                help="S3 prefix where documents are stored"
            )
        
        with col2:
            st.markdown("**Processing Settings**")
            batch_name = st.text_input(
                "Batch Name",
                value=f"S3_Batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                help="Descriptive name for this batch"
            )
            
            password = st.text_input(
                "Document Password",
                value="Hubert",
                type="password",
                help="Password for encrypted documents"
            )
            
            model_options = ["openai/gpt-4o-mini", "openai/gpt-4o", "deepseek/deepseek-chat"]
            selected_model = st.selectbox(
                "Processing Model",
                model_options,
                index=0,
                help="LLM model for PII extraction"
            )
    
    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_documents = st.number_input(
                "Maximum Documents",
                min_value=10,
                max_value=10000,
                value=100,
                step=10,
                help="Maximum documents to process (up to 10,000)"
            )
            
            max_budget = st.number_input(
                "Maximum Budget ($)",
                min_value=1.0,
                max_value=2000.0,
                value=100.0,
                step=10.0,
                help="Maximum spend for this batch"
            )
        
        with col2:
            max_concurrent = st.slider(
                "Concurrent Processing",
                min_value=1,
                max_value=20,
                value=8,
                help="Parallel processing workers"
            )
            
            skip_processed = st.checkbox(
                "Skip Already Processed",
                value=True,
                help="Skip documents with .processed markers"
            )
    
    # Cost Estimation
    if max_documents > 0:
        estimated_cost = max_documents * 0.03  # Conservative estimate
        processing_time_hours = max_documents / (max_concurrent * 20)  # ~20 docs per worker per hour
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Estimated Cost", f"${estimated_cost:.2f}")
        with col2:
            st.metric("Est. Processing Time", f"{processing_time_hours:.1f}h")
        with col3:
            if estimated_cost > max_budget:
                st.error(f"Cost exceeds budget!")
            else:
                st.success("✅ Within budget")
    
    # Document Discovery
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Discover Documents", help="Preview documents in S3 bucket"):
            if bucket_name:
                discover_s3_documents(bucket_name, aws_region, input_prefix, max_documents)
            else:
                st.error("Please provide S3 bucket name")
    
    with col2:
        start_batch = st.button(
            "▶️ Create & Start Batch", 
            help="Create batch and start background processing",
            type="primary",
            disabled=not bucket_name or not batch_name
        )
    
    # Start Processing
    if start_batch:
        try:
            # Pre-flight budget check
            from llm.cost_tracker import cost_tracker
            
            if not cost_tracker.can_afford('openai', estimated_cost):
                remaining = cost_tracker.get_remaining_budget('openai')
                st.error(f"Insufficient budget. Estimated: ${estimated_cost:.2f}, Available: ${remaining:.2f}")
                return
            
            # Discover documents to create batch
            with st.spinner("🔍 Discovering S3 documents..."):
                s3_processor = S3DocumentProcessor(bucket_name, aws_region)
                s3_documents = s3_processor.discover_documents(
                    prefix=input_prefix,
                    max_documents=max_documents,
                    skip_processed=skip_processed
                )
            
            if not s3_documents:
                st.warning("No documents found in S3 bucket")
                return
            
            # Convert S3Documents to database format
            doc_records = []
            for s3_doc in s3_documents:
                doc_records.append({
                    'filename': s3_doc.filename,
                    's3_key': s3_doc.key,
                    'file_size': s3_doc.size,
                    'document_type': s3_doc.extension,
                    'metadata': {
                        'bucket': s3_doc.bucket,
                        'last_modified': s3_doc.last_modified.isoformat()
                    }
                })
            
            # Create batch in database
            with st.spinner("📝 Creating processing batch..."):
                batch_id = db.create_batch(
                    name=batch_name,
                    documents=doc_records,
                    estimated_cost=estimated_cost
                )
            
            # Start background processing
            with st.spinner("🚀 Starting background processing..."):
                job_id = processor.start_batch_processing(
                    batch_id=batch_id,
                    model_key=selected_model,
                    password=password,
                    max_budget=max_budget
                )
            
            st.success(f"✅ Batch created and processing started!")
            st.info(f"**Batch ID:** `{batch_id}`")
            st.info(f"**Job ID:** `{job_id}`")
            st.info(f"**Documents:** {len(doc_records)}")
            
            # Show progress tracking tip
            st.markdown("💡 **Tip:** Switch to the 'Active Batches' tab to monitor real-time progress.")
            
        except Exception as e:
            st.error(f"Failed to start batch processing: {str(e)}")
            logger.error(f"Batch creation error: {e}")


def show_active_batches_interface(db, processor):
    """Interface for monitoring active batch processing"""
    
    st.markdown("#### 📊 Active Processing Batches")
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns(3)
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
    with col2:
        refresh_interval = st.selectbox("Refresh Rate", [5, 10, 30, 60], index=1)
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Get active batches
    all_batches = db.get_all_batches(limit=50)
    active_batches = [b for b in all_batches if b.status in ['pending', 'processing']]
    
    if not active_batches:
        st.info("No active batches. Create a new batch to start processing.")
        return
    
    # Display active batches
    for batch in active_batches:
        with st.expander(f"📦 {batch.name} ({batch.status.title()})", expanded=True):
            
            # Batch overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                progress = batch.processed_documents / max(batch.total_documents, 1)
                st.metric("Progress", f"{progress:.1%}")
                st.progress(progress)
            
            with col2:
                st.metric("Documents", f"{batch.processed_documents + batch.failed_documents} / {batch.total_documents}")
            
            with col3:
                st.metric("Cost", f"${batch.total_cost:.2f}")
                if batch.estimated_cost > 0:
                    cost_progress = batch.total_cost / batch.estimated_cost
                    st.progress(min(cost_progress, 1.0))
            
            with col4:
                if batch.status == 'processing':
                    st.success("🟢 Processing")
                elif batch.status == 'pending':
                    st.info("🟡 Pending")
            
            # Detailed information
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Batch Details**")
                st.text(f"ID: {batch.batch_id}")
                st.text(f"Created: {batch.created_at}")
                if batch.started_at:
                    st.text(f"Started: {batch.started_at}")
                
                # Control buttons
                if batch.status == 'processing':
                    if st.button(f"🛑 Cancel", key=f"cancel_{batch.batch_id}"):
                        if processor.cancel_job(f"job_{batch.batch_id}"):
                            st.success("Batch cancelled")
                            st.rerun()
                        else:
                            st.error("Failed to cancel batch")
            
            with col2:
                st.markdown("**Recent Activity**")
                recent_logs = db.get_recent_logs(batch.batch_id, limit=5)
                
                if recent_logs:
                    for log in recent_logs:
                        level_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(log['level'], "📝")
                        timestamp = datetime.fromisoformat(log['timestamp']).strftime('%H:%M:%S')
                        st.text(f"{level_emoji} {timestamp}: {log['message']}")
                else:
                    st.text("No recent activity")
            
            # Detailed document view
            if st.checkbox(f"Show Document Details", key=f"details_{batch.batch_id}"):
                show_batch_documents_paginated(db, batch.batch_id)
    
    # Auto-refresh
    if auto_refresh and active_batches:
        time.sleep(refresh_interval)
        st.rerun()


def show_batch_documents_paginated(db, batch_id: str):
    """Show paginated document list for a batch"""
    
    # Pagination controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        page_size = st.selectbox("Docs per page", [25, 50, 100], index=1, key=f"page_size_{batch_id}")
    
    with col2:
        status_filter = st.selectbox(
            "Filter by status", 
            ["All", "pending", "processing", "completed", "failed"],
            key=f"status_filter_{batch_id}"
        )
    
    # Get documents with pagination
    status = None if status_filter == "All" else status_filter
    documents, total_count = db.get_documents_paginated(
        batch_id=batch_id,
        offset=0,
        limit=page_size,
        status_filter=status
    )
    
    with col3:
        st.metric("Total Documents", total_count)
    
    if documents:
        # Create dataframe for display
        doc_data = []
        for doc in documents:
            doc_data.append({
                "Filename": doc.filename,
                "Status": doc.status,
                "Type": doc.document_type,
                "Size (KB)": f"{doc.file_size / 1024:.1f}",
                "Cost": f"${doc.cost:.3f}",
                "Processing Time": f"{doc.processing_time:.1f}s" if doc.processing_time > 0 else "-"
            })
        
        df = pd.DataFrame(doc_data)
        
        # Color-code status
        def color_status(val):
            colors = {
                'completed': 'background-color: #d4edda',
                'failed': 'background-color: #f8d7da',
                'processing': 'background-color: #fff3cd',
                'pending': 'background-color: #e2e3e5'
            }
            return colors.get(val, '')
        
        styled_df = df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No documents found for selected criteria")


def show_batch_history_interface(db):
    """Interface for viewing batch processing history"""
    
    st.markdown("#### 📜 Processing History")
    
    # Get all batches
    all_batches = db.get_all_batches(limit=100)
    completed_batches = [b for b in all_batches if b.status in ['completed', 'completed_with_errors', 'failed', 'cancelled']]
    
    if not completed_batches:
        st.info("No completed batches yet.")
        return
    
    # Summary statistics
    total_processed = sum(b.processed_documents for b in completed_batches)
    total_cost = sum(b.total_cost for b in completed_batches)
    success_rate = sum(b.processed_documents for b in completed_batches) / max(sum(b.total_documents for b in completed_batches), 1)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents Processed", f"{total_processed:,}")
    with col2:
        st.metric("Total Cost", f"${total_cost:.2f}")
    with col3:
        st.metric("Overall Success Rate", f"{success_rate:.1%}")
    
    # Batch history table
    if completed_batches:
        history_data = []
        for batch in completed_batches:
            success_rate = batch.processed_documents / max(batch.total_documents, 1)
            
            history_data.append({
                "Batch Name": batch.name,
                "Status": batch.status,
                "Documents": f"{batch.processed_documents} / {batch.total_documents}",
                "Success Rate": f"{success_rate:.1%}",
                "Cost": f"${batch.total_cost:.2f}",
                "Created": batch.created_at,
                "Completed": batch.completed_at or "-"
            })
        
        df = pd.DataFrame(history_data)
        
        # Color-code status
        def color_status(val):
            colors = {
                'completed': 'background-color: #d4edda',
                'completed_with_errors': 'background-color: #fff3cd',
                'failed': 'background-color: #f8d7da',
                'cancelled': 'background-color: #e2e3e5'
            }
            return colors.get(val, '')
        
        styled_df = df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
    
    # Cleanup options
    with st.expander("🧹 Cleanup Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            days_old = st.number_input("Delete batches older than (days)", min_value=1, value=30)
        
        with col2:
            if st.button("🗑️ Cleanup Old Batches"):
                cleaned = db.cleanup_old_batches(days_old)
                if cleaned > 0:
                    st.success(f"Cleaned up {cleaned} old batches")
                    st.rerun()
                else:
                    st.info("No old batches to clean up")


def discover_s3_documents(bucket_name: str, aws_region: str, input_prefix: str, max_documents: int = 100):
    """Discover and preview S3 documents"""
    try:
        # Initialize S3 processor
        processor = S3DocumentProcessor(
            bucket_name=bucket_name,
            aws_region=aws_region
        )
        
        with st.spinner("🔍 Discovering S3 documents..."):
            documents = processor.discover_documents(
                prefix=input_prefix,
                max_documents=50,  # Limit for preview
                skip_processed=True
            )
        
        if documents:
            st.success(f"✅ Found {len(documents)} documents in S3")
            
            # Show preview
            with st.expander("📋 Document Preview", expanded=True):
                preview_data = []
                for doc in documents[:20]:  # Show first 20
                    preview_data.append({
                        "Filename": doc.filename,
                        "Size (KB)": f"{doc.size / 1024:.1f}",
                        "Type": doc.extension,
                        "Last Modified": doc.last_modified.strftime('%Y-%m-%d %H:%M')
                    })
                
                st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
                
                if len(documents) > 20:
                    st.info(f"Showing first 20 documents. Total found: {len(documents)}")
        else:
            st.warning("No unprocessed documents found in the specified S3 location")
            
    except Exception as e:
        st.error(f"Failed to discover S3 documents: {str(e)}")
        logger.error(f"S3 discovery error: {e}")

def show_s3_dataset_import_interface():
    """Interface for importing existing S3 datasets"""
    with st.expander("📥 Import S3 Dataset", expanded=True):
        dataset_s3_key = st.text_input(
            "Dataset S3 Key",
            placeholder="labeled-datasets/phase0_dataset_20241220_143022.json",
            help="S3 key of the previously exported labeled dataset"
        )
        
        if st.button("Import Dataset") and dataset_s3_key:
            try:
                import_s3_dataset_by_key(dataset_s3_key)
                st.success("Dataset imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Import failed: {str(e)}")

def import_s3_results_to_session(result: Dict[str, Any]):
    """Import S3 processing results into current session"""
    # This would need to fetch the actual dataset from S3
    # For now, we'll create placeholder entries
    
    if result.get('dataset_s3_key'):
        # Add processing history
        if 's3_processing_history' not in st.session_state:
            st.session_state.s3_processing_history = []
        
        history_entry = {
            'timestamp': datetime.now(),
            'documents_processed': result.get('documents_processed', 0),
            'success_rate': result.get('success_rate', 0),
            'total_cost': result.get('total_cost', 0),
            'dataset_s3_key': result.get('dataset_s3_key')
        }
        
        st.session_state.s3_processing_history.append(history_entry)
        
        # Keep only last 10 entries
        if len(st.session_state.s3_processing_history) > 10:
            st.session_state.s3_processing_history = st.session_state.s3_processing_history[-10:]

def import_s3_dataset_by_key(dataset_s3_key: str):
    """Import a specific S3 dataset by its key"""
    # Implementation would fetch and parse the S3 dataset
    # For now, this is a placeholder
    st.info("S3 dataset import functionality will be implemented based on the S3 integration setup")

def show_document_details(doc_name: str):
    """Show detailed view of selected document"""
    document = next((doc for doc in st.session_state.phase0_dataset if doc['name'] == doc_name), None)
    
    if not document:
        st.error("Document not found")
        return
    
    with st.expander(f"Document Details: {doc_name}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Document Information**")
            st.write(f"ID: {document['id']}")
            st.write(f"Type: {document['metadata']['document_type']}")
            st.write(f"Size: {document['size'] / 1024:.1f} KB")
            st.write(f"Priority: {document['priority']}")
            
            # Enhanced status display
            if document['labeled']:
                labels = document.get('claude_labels', {})
                if labels.get('method') == 'unsupported_format':
                    st.error(f"⚠️ Unsupported Format: {labels.get('error', '')}")
                    if labels.get('suggestion'):
                        st.info(f"💡 {labels['suggestion']}")
                elif labels.get('method') == 'no_text_content':
                    st.warning(f"ℹ️ No Text Found: {labels.get('error', '')}")
                    if labels.get('suggestion'):
                        st.info(f"💡 {labels['suggestion']}")
                elif labels.get('method') == 'failed':
                    st.error(f"❌ Processing Failed: {labels.get('error', '')}")
                elif labels.get('entities'):
                    st.success(f"✅ Successfully Labeled ({len(labels['entities'])} entities found)")
                else:
                    st.success("✅ Processed (no PII entities found)")
            else:
                st.info("⏳ Pending Processing")
        
        with col2:
            st.markdown("**Metadata**")
            st.write(f"Difficulty: {document['metadata']['difficulty_level']}")
            st.write(f"Domain: {document['metadata']['domain']}")
            if document['metadata'].get('domain_detail'):
                st.write(f"Domain Detail: {document['metadata']['domain_detail']}")
            st.write(f"Validation: {document['validation_status']}")
            st.write(f"Uploaded: {document['uploaded_at'][:19].replace('T', ' ')}")
            
            # Processing details
            if document['labeled']:
                labels = document.get('claude_labels', {})
                if labels.get('method'):
                    st.write(f"Processing Method: {labels['method']}")
                if labels.get('processing_time'):
                    st.write(f"Processing Time: {labels['processing_time']:.2f}s")
                if labels.get('cost'):
                    st.write(f"Processing Cost: ${labels['cost']:.4f}")
        
        # Show labels if available
        if document.get('claude_labels'):
            st.markdown("**GPT-4o Labels**")
            labels = document['claude_labels']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Entities Found", len(labels.get('entities', [])))
            with col2:
                st.metric("Confidence Score", f"{labels.get('confidence_score', 0):.1%}")
            with col3:
                st.metric("Processing Cost", f"${labels.get('cost', 0):.4f}")
            
            if labels.get('entities'):
                entities_df = pd.DataFrame(labels['entities'])
                st.dataframe(entities_df, use_container_width=True)

def show_claude_labeling_interface():
    """Show Claude labeling interface"""
    st.markdown("### 🤖 Claude Ground Truth Labeling")
    
    if not auth.has_permission('write'):
        st.warning("Claude labeling requires write permissions.")
        return
    
    # Add troubleshooting tips for password-protected documents
    with st.expander("🔐 Troubleshooting Password-Protected Documents"):
        st.markdown("""
        **If password-protected documents show 0 entities:**
        
        1. **Verify Password**: Ensure the password is exactly "Hubert" (case-sensitive)
        2. **Document Type**: Some PDFs are scanned images requiring OCR
        3. **OCR Settings**: Enable LLM OCR in Configuration for better results
        4. **Processing Time**: OCR can take 30-60 seconds per page
        5. **Check Logs**: Look for debug messages during processing
        
        **Common Issues:**
        - Scanned PDFs need OCR processing which may take longer
        - Some PDFs have text layers but are actually images
        - Ensure sufficient processing time for large documents
        """)
    
    # Claude availability check
    available_models = llm_service.get_available_models()
    claude_models = [model for model in available_models if 'claude' in model.lower()]
    
    if not claude_models:
        st.error("Claude models are not available. Please check your Anthropic API configuration.")
        return
    
    # Model selection and configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_model = st.selectbox(
            "Select Claude Model",
            claude_models,
            help="Choose the Claude model for labeling"
        )
    
    with col2:
        # Model info
        model_info = llm_service.get_model_info(selected_model)
        st.info(f"Cost: $**{model_info.get('cost_per_1k_input_tokens', 0):.3f}**/1K input tokens")
    
    with col3:
        # Password for encrypted documents
        batch_password = st.text_input(
            "Document Password",
            type="password",
            value="Hubert",
            help="Password for encrypted documents (PDF, DOCX, Excel)",
            key="batch_labeling_password"
        )
    
    # Labeling queue management
    st.markdown("#### Labeling Queue")
    
    unlabeled_docs = [doc for doc in st.session_state.phase0_dataset if not doc['labeled']]
    
    if not unlabeled_docs:
        st.info("All documents are already labeled. Upload more documents or clear existing labels.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Unlabeled Documents", len(unlabeled_docs))
    
    with col2:
        total_size = sum(doc['size'] for doc in unlabeled_docs) / (1024 * 1024)
        st.metric("Total Size", f"{total_size:.1f} MB")
    
    with col3:
        estimated_cost = estimate_labeling_cost(unlabeled_docs, selected_model)
        st.metric("Estimated Cost", f"${estimated_cost:.3f}")
    
    # Batch labeling options
    st.markdown("#### Batch Labeling Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.slider("Batch Size", 1, min(10, len(unlabeled_docs)), 3)
        difficulty_filter = st.multiselect(
            "Filter by Difficulty",
            ['High', 'Medium', 'Low', 'Unknown'],
            default=['High', 'Medium', 'Low', 'Unknown']
        )
    
    with col2:
        domain_filter = st.multiselect(
            "Filter by Domain",
            ['Finance', 'Healthcare', 'Legal', 'HR', 'General', 'Unknown'],
            default=['Finance', 'Healthcare', 'Legal', 'HR', 'General', 'Unknown']
        )
        priority_order = st.selectbox(
            "Priority Order",
            ['High to Low', 'Low to High', 'Random']
        )
    
    # Filter documents
    filtered_docs = filter_documents_for_labeling(
        unlabeled_docs, difficulty_filter, domain_filter, priority_order
    )
    
    st.write(f"**{len(filtered_docs)} documents** match your criteria")
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Label Next Batch", disabled=len(filtered_docs) == 0):
            label_document_batch(filtered_docs[:batch_size], selected_model, batch_password)
    
    with col2:
        if st.button("Label All Filtered", disabled=len(filtered_docs) == 0):
            estimated_cost = estimate_labeling_cost(filtered_docs, selected_model)
            st.warning(f"⚠️ This will label {len(filtered_docs)} documents with estimated cost: ${estimated_cost:.3f}")
            if st.button("🚀 Confirm Label All", type="primary"):
                label_document_batch(filtered_docs, selected_model, batch_password)
    
    with col3:
        if st.button("Preview Labeling"):
            show_labeling_preview(filtered_docs[:3], selected_model)
    
    # Labeling history
    st.markdown("#### Recent Labeling Activity")
    show_labeling_history()

def estimate_labeling_cost(documents: List[Dict], model_key: str) -> float:
    """Estimate cost for labeling documents"""
    model_info = llm_service.get_model_info(model_key)
    cost_per_1k_tokens = model_info.get('cost_per_1k_input_tokens', 0.0025)
    
    # Rough estimation: 1000 tokens per image, 500 tokens per text document
    total_cost = 0
    for doc in documents:
        if doc['type'].startswith('image/') or doc['name'].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
            # Image processing
            total_cost += cost_per_1k_tokens * 1.5  # Input + output tokens
        else:
            # Text processing
            total_cost += cost_per_1k_tokens * 0.8
    
    return total_cost

def filter_documents_for_labeling(
    documents: List[Dict], 
    difficulty_filter: List[str],
    domain_filter: List[str],
    priority_order: str
) -> List[Dict]:
    """Filter and sort documents for labeling"""
    filtered = [
        doc for doc in documents
        if doc['metadata']['difficulty_level'] in difficulty_filter
        and doc['metadata']['domain'] in domain_filter
    ]
    
    if priority_order == 'High to Low':
        priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
        filtered.sort(key=lambda x: priority_map.get(x['priority'], 0), reverse=True)
    elif priority_order == 'Low to High':
        priority_map = {'High': 3, 'Medium': 2, 'Low': 1}
        filtered.sort(key=lambda x: priority_map.get(x['priority'], 0))
    else:  # Random
        np.random.shuffle(filtered)
    
    return filtered

def label_document_batch(documents: List[Dict], model_key: str, password: str = ""):
    """Label a batch of documents with optimized logging for large volumes"""
    if not documents:
        return
    
    # Initialize tracking
    progress_bar = st.progress(0)
    status_container = st.empty()
    error_container = st.empty()
    
    total_cost = 0
    successful_labels = 0
    error_count = 0
    start_time = time.time()
    
    # Batch processing to prevent UI blocking
    batch_size = min(20, len(documents))  # Process in chunks
    
    for batch_start in range(0, len(documents), batch_size):
        batch_end = min(batch_start + batch_size, len(documents))
        batch_docs = documents[batch_start:batch_end]
        
        for i, document in enumerate(batch_docs):
            overall_index = batch_start + i
            
            try:
                # Suppress individual document logs during batch processing
                labels = auto_label_document(document['id'], password, model_key)
                document['claude_labels'] = labels
                document['labeled'] = True
                document['validation_status'] = 'Auto-labeled'
                
                total_cost += labels.get('cost', 0)
                successful_labels += 1
                
            except Exception as e:
                error_count += 1
                # Log only critical errors to prevent log spam
                logger.error(f"Failed to label {document['name']}: {str(e)}")
            
            # Update progress
            progress = (overall_index + 1) / len(documents)
            progress_bar.progress(progress)
            
            # Update status every 10 documents or at completion
            if (overall_index + 1) % 10 == 0 or overall_index == len(documents) - 1:
                elapsed = time.time() - start_time
                rate = (overall_index + 1) / elapsed if elapsed > 0 else 0
                eta = (len(documents) - overall_index - 1) / rate if rate > 0 else 0
                
                status_container.text(
                    f"Processing: {overall_index + 1}/{len(documents)} "
                    f"({rate:.1f} docs/sec, ETA: {eta:.0f}s, Cost: ${total_cost:.4f})"
                )
    
    # Final summary
    total_time = time.time() - start_time
    status_container.text(
        f"Complete: {successful_labels} labeled, {error_count} errors, "
        f"${total_cost:.4f} cost, {total_time:.1f}s ({len(documents)/total_time:.1f} docs/sec)"
    )
    
    # Show errors summary if any
    if error_count > 0:
        error_container.warning(
            f"⚠️ {error_count}/{len(documents)} documents failed. "
            f"Check application logs for details."
        )
    
    if successful_labels > 0:
        st.success(f"✅ Batch complete: {successful_labels}/{len(documents)} documents labeled successfully.")

def show_labeling_preview(documents: List[Dict], model_key: str):
    """Show preview of what would be labeled"""
    with st.expander("Labeling Preview", expanded=True):
        st.markdown("**Documents to be labeled:**")
        
        for doc in documents:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**{doc['name']}**")
            
            with col2:
                st.write(f"Type: {doc['metadata']['document_type']}")
            
            with col3:
                st.write(f"Priority: {doc['priority']}")

def show_labeling_history():
    """Show recent labeling activity"""
    labeled_docs = [doc for doc in st.session_state.phase0_dataset if doc['labeled']]
    
    if not labeled_docs:
        st.info("No labeling activity yet.")
        return
    
    # Sort by upload time (most recent first)
    labeled_docs.sort(key=lambda x: x['uploaded_at'], reverse=True)
    
    for doc in labeled_docs[:5]:  # Show last 5
        labels = doc.get('claude_labels', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write(f"**{doc['name']}**")
        
        with col2:
            st.write(f"Entities: {len(labels.get('entities', []))}")
        
        with col3:
            st.write(f"Confidence: {labels.get('confidence_score', 0):.1%}")
        
        with col4:
            st.write(f"Cost: ${labels.get('cost', 0):.4f}")

def show_metadata_tagging_interface():
    """Show metadata tagging interface"""
    st.markdown("### 🏷️ Metadata Tagging System")
    
    if not auth.has_permission('write'):
        st.warning("Metadata tagging requires write permissions.")
        return
    
    if not st.session_state.phase0_dataset:
        st.info("No documents uploaded yet. Upload documents first.")
        return
    
    # Metadata overview
    st.markdown("#### Metadata Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        difficulty_dist = {}
        for doc in st.session_state.phase0_dataset:
            diff = doc['metadata']['difficulty_level']
            difficulty_dist[diff] = difficulty_dist.get(diff, 0) + 1
        
        st.markdown("**Difficulty Distribution**")
        for diff, count in difficulty_dist.items():
            st.write(f"{diff}: {count}")
    
    with col2:
        domain_dist = {}
        for doc in st.session_state.phase0_dataset:
            domain = doc['metadata']['domain']
            domain_dist[domain] = domain_dist.get(domain, 0) + 1
        
        st.markdown("**Domain Distribution**")
        for domain, count in domain_dist.items():
            st.write(f"{domain}: {count}")
    
    with col3:
        type_dist = {}
        for doc in st.session_state.phase0_dataset:
            doc_type = doc['metadata']['document_type']
            type_dist[doc_type] = type_dist.get(doc_type, 0) + 1
        
        st.markdown("**Document Type Distribution**")
        for doc_type, count in type_dist.items():
            st.write(f"{doc_type}: {count}")
    
    # Document selection for tagging
    st.markdown("#### Tag Documents")
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        tag_filter = st.selectbox(
            "Filter documents to tag",
            ['All documents', 'Untagged only', 'By document type', 'By priority']
        )
    
    with col2:
        if tag_filter == 'By document type':
            selected_type = st.selectbox(
                "Select document type",
                list(type_dist.keys())
            )
        elif tag_filter == 'By priority':
            selected_priority = st.selectbox(
                "Select priority",
                ['High', 'Medium', 'Low']
            )
        else:
            selected_type = None
            selected_priority = None
    
    # Get filtered documents
    filtered_docs = filter_documents_for_tagging(
        st.session_state.phase0_dataset,
        tag_filter,
        selected_type if tag_filter == 'By document type' else None,
        selected_priority if tag_filter == 'By priority' else None
    )
    
    st.write(f"**{len(filtered_docs)} documents** match your filter criteria")
    
    # Tagging interface
    if filtered_docs:
        selected_doc = st.selectbox(
            "Select document to tag:",
            ['None'] + [doc['name'] for doc in filtered_docs]
        )
        
        if selected_doc != 'None':
            show_document_tagging_interface(selected_doc)
    
    # Bulk tagging
    st.markdown("#### Bulk Tagging")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bulk_difficulty = st.selectbox(
            "Apply difficulty to filtered:",
            ['No change', 'Easy', 'Medium', 'Hard']
        )
    
    with col2:
        bulk_domain = st.selectbox(
            "Apply domain to filtered:",
            ['No change', 'HR', 'Finance', 'Legal', 'Medical', 'Government', 'Education', 'Other']
        )
    
    with col3:
        if st.button("Apply Bulk Tags"):
            apply_bulk_tags(filtered_docs, bulk_difficulty, bulk_domain)

def filter_documents_for_tagging(
    documents: List[Dict],
    filter_type: str,
    selected_type: Optional[str] = None,
    selected_priority: Optional[str] = None
) -> List[Dict]:
    """Filter documents for tagging"""
    if filter_type == 'All documents':
        return documents
    elif filter_type == 'Untagged only':
        return [
            doc for doc in documents
            if doc['metadata']['difficulty_level'] == 'Unknown' 
            or doc['metadata']['domain'] == 'Unknown'
        ]
    elif filter_type == 'By document type' and selected_type:
        return [
            doc for doc in documents
            if doc['metadata']['document_type'] == selected_type
        ]
    elif filter_type == 'By priority' and selected_priority:
        return [
            doc for doc in documents
            if doc['priority'] == selected_priority
        ]
    else:
        return documents

def show_document_tagging_interface(doc_name: str):
    """Show tagging interface for specific document"""
    document = next((doc for doc in st.session_state.phase0_dataset if doc['name'] == doc_name), None)
    
    if not document:
        st.error("Document not found")
        return
    
    with st.expander(f"Tag Document: {doc_name}", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            new_difficulty = st.selectbox(
                "Difficulty Level",
                ['Easy', 'Medium', 'Hard'],
                index=['Easy', 'Medium', 'Hard'].index(document['metadata']['difficulty_level'])
                if document['metadata']['difficulty_level'] in ['Easy', 'Medium', 'Hard'] else 1
            )
            
            domain_options = ['HR', 'Finance', 'Legal', 'Medical', 'Government', 'Education', 'Other']
            new_domain = st.selectbox(
                "Domain",
                domain_options,
                index=domain_options.index(document['metadata']['domain'])
                if document['metadata']['domain'] in domain_options else 6  # Default to 'Other'
            )
            
            # Domain detail for additional context
            domain_detail = st.text_input(
                "Domain Detail (optional)",
                value=document['metadata'].get('domain_detail', ''),
                help="Specific subdomain or document type (e.g., 'Absence Request Form', 'Pay Stub', 'Tax Document')"
            )
        
        with col2:
            # Additional metadata fields
            complexity_score = st.slider(
                "Complexity Score",
                1, 10,
                document['metadata'].get('complexity_score', 5)
            )
            
            contains_sensitive = st.checkbox(
                "Contains Sensitive Data",
                document['metadata'].get('contains_sensitive', False)
            )
            
            requires_review = st.checkbox(
                "Requires Manual Review",
                document['metadata'].get('requires_review', False)
            )
        
        # Notes field
        notes = st.text_area(
            "Notes",
            document['metadata'].get('notes', ''),
            help="Add any additional notes about this document"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(f"Update Tags for {doc_name}"):
                update_document_metadata(
                    document['id'],
                    new_difficulty,
                    new_domain,
                    domain_detail,
                    complexity_score,
                    contains_sensitive,
                    requires_review,
                    notes
                )
                st.success("Metadata updated successfully!")
                st.rerun()
        
        with col2:
            if st.button("Auto-suggest Tags"):
                auto_suggest_tags(document)

def update_document_metadata(
    doc_id: str,
    difficulty: str,
    domain: str,
    domain_detail: str,
    complexity_score: int,
    contains_sensitive: bool,
    requires_review: bool,
    notes: str
):
    """Update document metadata"""
    document = next((doc for doc in st.session_state.phase0_dataset if doc['id'] == doc_id), None)
    
    if document:
        document['metadata'].update({
            'difficulty_level': difficulty,
            'domain': domain,
            'domain_detail': domain_detail,
            'complexity_score': complexity_score,
            'contains_sensitive': contains_sensitive,
            'requires_review': requires_review,
            'notes': notes,
            'last_updated': datetime.now().isoformat()
        })

def auto_suggest_tags(document: Dict):
    """Auto-suggest tags based on document content and labels"""
    suggestions = {}
    
    # Suggest difficulty based on entity count and types
    if document.get('claude_labels'):
        entities = document['claude_labels'].get('entities', [])
        entity_count = len(entities)
        
        if entity_count > 10:
            suggestions['difficulty'] = 'High'
        elif entity_count > 5:
            suggestions['difficulty'] = 'Medium'
        else:
            suggestions['difficulty'] = 'Low'
        
        # Suggest domain based on entity types
        entity_types = [entity['type'] for entity in entities]
        
        if 'SSN' in entity_types or 'ID_NUMBER' in entity_types:
            suggestions['domain'] = 'HR'
        elif 'ORGANIZATION' in entity_types and 'ADDRESS' in entity_types:
            suggestions['domain'] = 'Finance'
        else:
            suggestions['domain'] = 'General'
    
    # Suggest based on filename
    filename = document['name'].lower()
    if any(word in filename for word in ['medical', 'health', 'patient']):
        suggestions['domain'] = 'Healthcare'
    elif any(word in filename for word in ['legal', 'contract', 'agreement']):
        suggestions['domain'] = 'Legal'
    elif any(word in filename for word in ['financial', 'bank', 'invoice']):
        suggestions['domain'] = 'Finance'
    
    if suggestions:
        st.info(f"**Suggestions:** Difficulty: {suggestions.get('difficulty', 'N/A')}, Domain: {suggestions.get('domain', 'N/A')}")

def apply_bulk_tags(documents: List[Dict], difficulty: str, domain: str):
    """Apply bulk tags to filtered documents"""
    updated_count = 0
    
    for document in documents:
        if difficulty != 'No change':
            document['metadata']['difficulty_level'] = difficulty
            updated_count += 1
        
        if domain != 'No change':
            document['metadata']['domain'] = domain
            updated_count += 1
        
        document['metadata']['last_updated'] = datetime.now().isoformat()
    
    st.success(f"Updated metadata for {len(documents)} documents.")

def label_all_unlabeled_documents():
    """Label all unlabeled documents with GPT-4o"""
    unlabeled_docs = [doc for doc in st.session_state.phase0_dataset if not doc['labeled']]
    
    if not unlabeled_docs:
        st.info("All documents are already labeled.")
        return
    
    # Use first available GPT-4o model
    available_models = llm_service.get_available_models()
    gpt4o_models = [model for model in available_models if 'gpt-4o' in model]
    
    if not gpt4o_models:
        st.error("No GPT-4o models available.")
        return
    
    selected_model = gpt4o_models[0]
    estimated_cost = estimate_labeling_cost(unlabeled_docs, selected_model)
    
    st.warning(f"⚠️ This will label {len(unlabeled_docs)} documents with {selected_model}. Estimated cost: ${estimated_cost:.3f}")
    if st.button("🚀 Confirm Label All Unlabeled", type="primary"):
        # Use default password "Hubert" for bulk operations
        label_document_batch(unlabeled_docs, selected_model, "Hubert")

def export_current_dataset():
    """Export current dataset (labels only, without document content)"""
    if not st.session_state.phase0_dataset:
        st.warning("No dataset to export.")
        return
    
    # Create export data (labels only - no document content)
    export_documents = []
    for doc in st.session_state.phase0_dataset:
        export_doc = {
            'id': doc['id'],
            'name': doc['name'],
            'type': doc['type'],
            'uploaded_at': doc['uploaded_at'],
            'labeled': doc['labeled'],
            'metadata': doc['metadata']
        }
        
        # Only include labels if they exist
        if doc.get('claude_labels'):
            export_doc['labels'] = doc['claude_labels']
        
        export_documents.append(export_doc)
    
    # Create export data
    export_data = {
        'metadata': {
            'export_date': datetime.now().isoformat(),
            'total_documents': len(st.session_state.phase0_dataset),
            'labeled_documents': sum(1 for doc in st.session_state.phase0_dataset if doc['labeled']),
            'export_version': '1.0',
            'export_type': 'labels_only'
        },
        'documents': export_documents
    }
    
    # Convert to JSON
    json_data = json.dumps(export_data, indent=2)
    
    # Provide download
    st.download_button(
        label="Download Label Dataset (JSON)",
        data=json_data,
        file_name=f"phase0_labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("Label dataset export ready for download! (Contains PII labels and metadata only - no document content)")

def show_interactive_labeling_interface():
    """Show interactive labeling interface with validation"""
    st.markdown("### ⚡ Interactive Labeling & Validation")
    
    if not auth.has_permission('write'):
        st.warning("Interactive labeling requires write permissions.")
        return
    
    if not st.session_state.phase0_dataset:
        st.info("No documents uploaded yet. Upload documents first.")
        return
    
    # Validation overview
    st.markdown("#### Validation Overview")
    
    # Generate validation report
    validation_report = ground_truth_validator.generate_validation_report(
        st.session_state.phase0_dataset,
        include_detailed_scores=False
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_score = validation_report['summary_statistics'].get('mean_validation_score', 0)
        st.metric("Overall Quality Score", f"{overall_score:.1%}")
    
    with col2:
        needs_review = validation_report['summary_statistics'].get('documents_needing_review', 0)
        st.metric("Needs Review", needs_review)
    
    with col3:
        total_issues = validation_report['issues_summary'].get('total_issues', 0)
        st.metric("Total Issues", total_issues)
    
    with col4:
        quality_level = validation_report['quality_metrics'].get('quality_level', 'Unknown')
        st.metric("Quality Level", quality_level)
    
    # Human review queue
    st.markdown("#### Human Review Queue")
    
    review_queue = ground_truth_validator.create_human_review_queue(
        st.session_state.phase0_dataset,
        prioritization_method="priority_score"
    )
    
    if review_queue:
        st.write(f"**{len(review_queue)} documents** require human review")
        
        # Display review queue
        queue_df = pd.DataFrame([
            {
                'Document': item['document_name'],
                'Validation Score': f"{item['validation_score']:.1%}",
                'Issues': item['issues_count'],
                'High Severity': item['high_severity_issues'],
                'Est. Time': f"{item['estimated_review_time']} min",
                'Priority': f"{item['priority_score']:.1f}"
            }
            for item in review_queue[:10]  # Show top 10
        ])
        
        st.dataframe(queue_df, use_container_width=True)
        
        # Document selection for review
        selected_doc = st.selectbox(
            "Select document for detailed review:",
            ['None'] + [item['document_name'] for item in review_queue]
        )
        
        if selected_doc != 'None':
            show_document_validation_interface(selected_doc, review_queue)
    
    else:
        st.success("All documents have passed validation! No human review needed.")
    
    # Validation actions
    st.markdown("#### Validation Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Validate All Documents"):
            run_full_validation()
    
    with col2:
        if st.button("Generate Quality Report"):
            generate_and_download_quality_report()
    
    with col3:
        if st.button("Export Validation Results"):
            export_validation_results()

def show_document_validation_interface(doc_name: str, review_queue: List[Dict]):
    """Show detailed validation interface for specific document"""
    document = next((doc for doc in st.session_state.phase0_dataset if doc['name'] == doc_name), None)
    review_item = next((item for item in review_queue if item['document_name'] == doc_name), None)
    
    if not document or not review_item:
        st.error("Document not found in review queue")
        return
    
    with st.expander(f"Review Document: {doc_name}", expanded=True):
        st.markdown("#### Validation Results")
        
        validation_details = review_item['validation_details']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Validation Score", f"{validation_details['overall_score']:.1%}")
        
        with col2:
            st.metric("Issues Detected", len(validation_details['issues_detected']))
        
        with col3:
            st.metric("Estimated Review Time", f"{review_item['estimated_review_time']} min")
        
        # Show detected issues
        if validation_details['issues_detected']:
            st.markdown("**Issues Detected:**")
            for issue in validation_details['issues_detected']:
                severity_color = {
                    'low': 'blue',
                    'medium': 'orange', 
                    'high': 'red',
                    'critical': 'darkred'
                }.get(issue.get('severity', 'low'), 'gray')
                
                st.markdown(f"""
                <div style="border-left: 4px solid {severity_color}; padding: 10px; margin: 5px 0; background-color: #f8f9fa;">
                    <strong>{issue.get('severity', 'Unknown').title()} - {issue.get('type', 'Unknown')}</strong><br>
                    {issue.get('description', 'No description')}
                </div>
                """, unsafe_allow_html=True)
        
        # Show entity validation scores
        if validation_details.get('entity_scores'):
            st.markdown("**Entity Validation Scores:**")
            
            entity_data = []
            for entity_score in validation_details['entity_scores']:
                entity_data.append({
                    'Type': entity_score.get('entity_type', 'Unknown'),
                    'Text': entity_score.get('entity_text', ''),
                    'Confidence': f"{entity_score.get('confidence', 0):.1%}",
                    'Validation Score': f"{entity_score.get('validation_score', 0):.1%}",
                    'Issues': ', '.join(entity_score.get('issues', []))
                })
            
            if entity_data:
                entity_df = pd.DataFrame(entity_data)
                st.dataframe(entity_df, use_container_width=True)
        
        # Manual validation interface
        st.markdown("#### Manual Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            validation_decision = st.selectbox(
                "Validation Decision",
                ['Pending', 'Approved', 'Rejected', 'Needs Revision']
            )
            
            reviewer_notes = st.text_area(
                "Reviewer Notes",
                document.get('reviewer_notes', ''),
                help="Add notes about the validation decision"
            )
        
        with col2:
            confidence_override = st.slider(
                "Override Confidence Score",
                0.0, 1.0,
                document.get('manual_confidence', validation_details['overall_score']),
                help="Override the automatic confidence score"
            )
            
            requires_reprocessing = st.checkbox(
                "Requires Reprocessing",
                document.get('requires_reprocessing', False)
            )
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"Save Validation for {doc_name}"):
                save_manual_validation(
                    document['id'],
                    validation_decision,
                    reviewer_notes,
                    confidence_override,
                    requires_reprocessing
                )
                st.success("Validation saved!")
                st.rerun()
        
        with col2:
            if st.button("Reprocess with Claude"):
                reprocess_document_with_claude(document['id'])
        
        with col3:
            if st.button("Mark as Reviewed"):
                mark_document_as_reviewed(document['id'])
                st.success("Document marked as reviewed!")
                st.rerun()

def save_manual_validation(
    doc_id: str,
    decision: str,
    notes: str,
    confidence_override: float,
    requires_reprocessing: bool
):
    """Save manual validation results"""
    document = next((doc for doc in st.session_state.phase0_dataset if doc['id'] == doc_id), None)
    
    if document:
        document['manual_validation'] = {
            'decision': decision,
            'reviewer_notes': notes,
            'confidence_override': confidence_override,
            'requires_reprocessing': requires_reprocessing,
            'review_date': datetime.now().isoformat(),
            'reviewer': st.session_state.get('username', 'Unknown')
        }
        
        # Update validation status
        if decision == 'Approved':
            document['validation_status'] = 'Approved'
        elif decision == 'Rejected':
            document['validation_status'] = 'Rejected'
        elif decision == 'Needs Revision':
            document['validation_status'] = 'Needs Revision'
        else:
            document['validation_status'] = 'Under Review'

def reprocess_document_with_claude(doc_id: str):
    """Reprocess document with Claude"""
    try:
        # Use default password "Hubert" for reprocessing
        labels = auto_label_document(doc_id, "Hubert")
        document = next((doc for doc in st.session_state.phase0_dataset if doc['id'] == doc_id), None)
        
        if document:
            document['claude_labels'] = labels
            document['labeled'] = True
            document['validation_status'] = 'Reprocessed'
            st.success("Document reprocessed successfully!")
            
    except Exception as e:
        st.error(f"Reprocessing failed: {e}")

def mark_document_as_reviewed(doc_id: str):
    """Mark document as manually reviewed"""
    document = next((doc for doc in st.session_state.phase0_dataset if doc['id'] == doc_id), None)
    
    if document:
        document['validation_status'] = 'Reviewed'
        document['reviewed_at'] = datetime.now().isoformat()
        document['reviewed_by'] = st.session_state.get('username', 'Unknown')

def run_full_validation():
    """Run validation on all documents"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, document in enumerate(st.session_state.phase0_dataset):
        status_text.text(f"Validating {document['name']}...")
        
        validation_result = ground_truth_validator.validate_document_labels(document)
        document['validation_result'] = validation_result
        
        progress_bar.progress((i + 1) / len(st.session_state.phase0_dataset))
    
    status_text.text("Validation complete!")
    st.success("All documents have been validated!")

def generate_and_download_quality_report():
    """Generate and provide download for quality report"""
    report = ground_truth_validator.generate_validation_report(
        st.session_state.phase0_dataset,
        include_detailed_scores=True
    )
    
    json_data = json.dumps(report, indent=2)
    
    st.download_button(
        label="Download Quality Report",
        data=json_data,
        file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("Quality report generated!")

def export_validation_results():
    """Export validation results"""
    validation_data = []
    
    for document in st.session_state.phase0_dataset:
        validation_result = ground_truth_validator.validate_document_labels(document)
        
        validation_data.append({
            'document_id': document['id'],
            'document_name': document['name'],
            'validation_score': validation_result['overall_score'],
            'issues_count': len(validation_result['issues_detected']),
            'needs_review': validation_result['needs_human_review'],
            'validation_status': document.get('validation_status', 'Pending'),
            'validation_details': validation_result
        })
    
    json_data = json.dumps(validation_data, indent=2)
    
    st.download_button(
        label="Download Validation Results",
        data=json_data,
        file_name=f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )
    
    st.success("Validation results exported!")

def show_dataset_export_interface():
    """Show dataset export interface"""
    st.markdown("### 📊 Dataset Export & Management")
    
    if not st.session_state.phase0_dataset:
        st.info("No dataset to export. Upload and label documents first.")
        return
    
    # Export statistics
    labeled_count = sum(1 for doc in st.session_state.phase0_dataset if doc['labeled'])
    total_entities = 0
    
    for doc in st.session_state.phase0_dataset:
        if doc.get('claude_labels'):
            total_entities += len(doc['claude_labels'].get('entities', []))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(st.session_state.phase0_dataset))
    
    with col2:
        st.metric("Labeled Documents", labeled_count)
    
    with col3:
        st.metric("Total Entities", total_entities)
    
    with col4:
        completion_rate = labeled_count / len(st.session_state.phase0_dataset) if st.session_state.phase0_dataset else 0
        st.metric("Completion Rate", f"{completion_rate:.1%}")
    
    # Dataset distribution analysis
    st.markdown("#### 📊 Dataset Distribution")
    
    # Calculate distributions
    difficulty_dist = {}
    domain_dist = {}
    
    for doc in st.session_state.phase0_dataset:
        if doc['labeled']:
            difficulty = doc['metadata']['difficulty_level']
            domain = doc['metadata']['domain']
            
            difficulty_dist[difficulty] = difficulty_dist.get(difficulty, 0) + 1
            domain_dist[domain] = domain_dist.get(domain, 0) + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Distribution by Difficulty:**")
        if difficulty_dist:
            for level in ['Easy', 'Medium', 'Hard', 'Unknown']:
                if level in difficulty_dist:
                    percentage = (difficulty_dist[level] / labeled_count) * 100
                    st.write(f"- {level}: {difficulty_dist[level]} documents ({percentage:.1f}%)")
        else:
            st.info("No labeled documents to analyze")
    
    with col2:
        st.markdown("**Distribution by Domain:**")
        if domain_dist:
            # Sort by count descending
            sorted_domains = sorted(domain_dist.items(), key=lambda x: x[1], reverse=True)
            for domain, count in sorted_domains[:5]:  # Show top 5
                percentage = (count / labeled_count) * 100
                st.write(f"- {domain}: {count} documents ({percentage:.1f}%)")
            if len(sorted_domains) > 5:
                others_count = sum(count for domain, count in sorted_domains[5:])
                others_percentage = (others_count / labeled_count) * 100
                st.write(f"- Others: {others_count} documents ({others_percentage:.1f}%)")
        else:
            st.info("No labeled documents to analyze")
    
    # Average entities by difficulty
    if labeled_count > 0:
        st.markdown("**Average PII Entities by Difficulty:**")
        
        difficulty_entities = {}
        for doc in st.session_state.phase0_dataset:
            if doc['labeled'] and doc.get('claude_labels'):
                difficulty = doc['metadata']['difficulty_level']
                entities_count = len(doc['claude_labels'].get('entities', []))
                
                if difficulty not in difficulty_entities:
                    difficulty_entities[difficulty] = []
                difficulty_entities[difficulty].append(entities_count)
        
        col1, col2, col3 = st.columns(3)
        
        for i, (level, col) in enumerate(zip(['Easy', 'Medium', 'Hard'], [col1, col2, col3])):
            with col:
                if level in difficulty_entities and difficulty_entities[level]:
                    avg_entities = sum(difficulty_entities[level]) / len(difficulty_entities[level])
                    st.metric(f"{level} Avg. Entities", f"{avg_entities:.1f}")
                else:
                    st.metric(f"{level} Avg. Entities", "N/A")
    
    # Export options
    st.markdown("#### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export Format",
            ['JSON', 'CSV', 'XML', 'COCO JSON']
        )
        
        include_content = st.checkbox("Include document content", value=False)
        include_metadata = st.checkbox("Include metadata", value=True)
    
    with col2:
        export_filter = st.selectbox(
            "Export Filter",
            ['All documents', 'Labeled only', 'High confidence only', 'By domain']
        )
        
        if export_filter == 'By domain':
            selected_domain = st.selectbox(
                "Select domain",
                ['HR', 'Finance', 'Legal', 'Medical', 'Government', 'Education', 'Other']
            )
    
    # Export preview
    if st.button("Preview Export"):
        show_export_preview(export_format, export_filter)
    
    # Export buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export Dataset"):
            export_dataset(export_format, export_filter, include_content, include_metadata)
    
    with col2:
        if st.button("Export Statistics"):
            export_statistics()
    
    with col3:
        if st.button("Export Quality Report"):
            export_quality_report()

def show_export_preview(export_format: str, export_filter: str):
    """Show preview of export data"""
    with st.expander("Export Preview", expanded=True):
        # Filter documents based on selection
        filtered_docs = filter_documents_for_export(st.session_state.phase0_dataset, export_filter)
        
        st.write(f"**{len(filtered_docs)} documents** will be exported in **{export_format}** format")
        
        if filtered_docs:
            # Show first document as example
            example_doc = filtered_docs[0]
            
            if export_format == 'JSON':
                preview_data = {
                    'id': example_doc['id'],
                    'name': example_doc['name'],
                    'metadata': example_doc['metadata'],
                    'labels': example_doc.get('claude_labels', {}).get('entities', [])
                }
                st.json(preview_data)
            
            elif export_format == 'CSV':
                # Show CSV structure
                csv_preview = pd.DataFrame([{
                    'document_id': example_doc['id'],
                    'document_name': example_doc['name'],
                    'document_type': example_doc['metadata']['document_type'],
                    'difficulty_level': example_doc['metadata']['difficulty_level'],
                    'domain': example_doc['metadata']['domain'],
                    'entity_count': len(example_doc.get('claude_labels', {}).get('entities', [])),
                    'labeled': example_doc['labeled']
                }])
                st.dataframe(csv_preview)

def filter_documents_for_export(documents: List[Dict], export_filter: str) -> List[Dict]:
    """Filter documents for export"""
    if export_filter == 'All documents':
        return documents
    elif export_filter == 'Labeled only':
        return [doc for doc in documents if doc['labeled']]
    elif export_filter == 'High confidence only':
        return [
            doc for doc in documents 
            if doc.get('claude_labels', {}).get('confidence_score', 0) > 0.8
        ]
    else:
        return documents

def export_dataset(export_format: str, export_filter: str, include_content: bool, include_metadata: bool):
    """Export dataset in specified format"""
    filtered_docs = filter_documents_for_export(st.session_state.phase0_dataset, export_filter)
    
    if export_format == 'JSON':
        export_data = prepare_json_export(filtered_docs, include_content, include_metadata)
        json_data = json.dumps(export_data, indent=2)
        
        st.download_button(
            label="Download JSON Export",
            data=json_data,
            file_name=f"phase0_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    elif export_format == 'CSV':
        csv_data = prepare_csv_export(filtered_docs, include_metadata)
        
        st.download_button(
            label="Download CSV Export", 
            data=csv_data,
            file_name=f"phase0_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    st.success(f"Export ready! {len(filtered_docs)} documents exported in {export_format} format.")

def prepare_json_export(documents: List[Dict], include_content: bool, include_metadata: bool) -> Dict:
    """Prepare data for JSON export"""
    export_docs = []
    
    for doc in documents:
        export_doc = {
            'id': doc['id'],
            'name': doc['name'],
            'type': doc['type'],
            'uploaded_at': doc['uploaded_at'],
            'labeled': doc['labeled']
        }
        
        if include_metadata:
            export_doc['metadata'] = doc['metadata']
        
        if include_content:
            export_doc['content'] = doc['content']
        
        if doc.get('claude_labels'):
            export_doc['labels'] = doc['claude_labels']
        
        export_docs.append(export_doc)
    
    return {
        'export_metadata': {
            'export_date': datetime.now().isoformat(),
            'total_documents': len(export_docs),
            'format': 'JSON',
            'version': '1.0'
        },
        'documents': export_docs
    }

def prepare_csv_export(documents: List[Dict], include_metadata: bool) -> str:
    """Prepare data for CSV export"""
    rows = []
    
    for doc in documents:
        labels = doc.get('claude_labels', {})
        entities = labels.get('entities', [])
        
        base_row = {
            'document_id': doc['id'],
            'document_name': doc['name'],
            'document_type': doc['type'],
            'uploaded_at': doc['uploaded_at'],
            'labeled': doc['labeled'],
            'entity_count': len(entities),
            'confidence_score': labels.get('confidence_score', 0)
        }
        
        if include_metadata:
            base_row.update({
                'metadata_document_type': doc['metadata']['document_type'],
                'metadata_difficulty': doc['metadata']['difficulty_level'],
                'metadata_domain': doc['metadata']['domain']
            })
        
        # Add entity details
        if entities:
            for entity in entities:
                row = base_row.copy()
                row.update({
                    'entity_type': entity.get('type', ''),
                    'entity_text': entity.get('text', ''),
                    'entity_confidence': entity.get('confidence', 0)
                })
                rows.append(row)
        else:
            rows.append(base_row)
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

def export_statistics():
    """Export dataset statistics"""
    stats = calculate_dataset_statistics()
    
    json_data = json.dumps(stats, indent=2)
    
    st.download_button(
        label="Download Statistics",
        data=json_data,
        file_name=f"phase0_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def export_quality_report():
    """Export quality assessment report"""
    report = generate_quality_report()
    
    json_data = json.dumps(report, indent=2)
    
    st.download_button(
        label="Download Quality Report",
        data=json_data,
        file_name=f"phase0_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def calculate_dataset_statistics() -> Dict[str, Any]:
    """Calculate comprehensive dataset statistics"""
    stats = {
        'total_documents': len(st.session_state.phase0_dataset),
        'labeled_documents': sum(1 for doc in st.session_state.phase0_dataset if doc['labeled']),
        'document_types': {},
        'difficulty_distribution': {},
        'domain_distribution': {},
        'entity_statistics': {},
        'total_entities': 0,
        'average_entities_per_document': 0,
        'total_labeling_cost': 0
    }
    
    for doc in st.session_state.phase0_dataset:
        # Document type distribution
        doc_type = doc['metadata']['document_type']
        stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
        
        # Difficulty distribution
        difficulty = doc['metadata']['difficulty_level']
        stats['difficulty_distribution'][difficulty] = stats['difficulty_distribution'].get(difficulty, 0) + 1
        
        # Domain distribution
        domain = doc['metadata']['domain']
        stats['domain_distribution'][domain] = stats['domain_distribution'].get(domain, 0) + 1
        
        # Entity statistics
        if doc.get('claude_labels'):
            entities = doc['claude_labels'].get('entities', [])
            stats['total_entities'] += len(entities)
            stats['total_labeling_cost'] += doc['claude_labels'].get('cost', 0)
            
            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                if entity_type not in stats['entity_statistics']:
                    stats['entity_statistics'][entity_type] = {
                        'count': 0, 
                        'avg_confidence': 0, 
                        'confidences': []
                    }
                stats['entity_statistics'][entity_type]['count'] += 1
                stats['entity_statistics'][entity_type]['confidences'].append(entity.get('confidence', 0))
    
    # Calculate averages
    if stats['labeled_documents'] > 0:
        stats['average_entities_per_document'] = stats['total_entities'] / stats['labeled_documents']
    
    # Calculate average confidence by entity type
    for entity_type in stats['entity_statistics']:
        confidences = stats['entity_statistics'][entity_type]['confidences']
        if confidences:
            stats['entity_statistics'][entity_type]['avg_confidence'] = sum(confidences) / len(confidences)
    
    return stats

def generate_quality_report() -> Dict[str, Any]:
    """Generate quality assessment report"""
    report = {
        'report_date': datetime.now().isoformat(),
        'overall_quality_score': 0,
        'completeness_score': 0,
        'consistency_score': 0,
        'confidence_distribution': {},
        'issues_detected': [],
        'recommendations': []
    }
    
    if not st.session_state.phase0_dataset:
        return report
    
    # Calculate completeness
    labeled_count = sum(1 for doc in st.session_state.phase0_dataset if doc['labeled'])
    report['completeness_score'] = labeled_count / len(st.session_state.phase0_dataset)
    
    # Analyze confidence distribution
    all_confidences = []
    for doc in st.session_state.phase0_dataset:
        if doc.get('claude_labels'):
            entities = doc['claude_labels'].get('entities', [])
            for entity in entities:
                all_confidences.append(entity.get('confidence', 0))
    
    if all_confidences:
        report['confidence_distribution'] = {
            'mean': np.mean(all_confidences),
            'median': np.median(all_confidences),
            'std': np.std(all_confidences),
            'min': np.min(all_confidences),
            'max': np.max(all_confidences)
        }
        
        # Quality issues detection
        low_confidence_count = sum(1 for conf in all_confidences if conf < 0.7)
        if low_confidence_count > len(all_confidences) * 0.1:
            report['issues_detected'].append({
                'type': 'Low Confidence Entities',
                'severity': 'Medium',
                'description': f'{low_confidence_count} entities have confidence < 70%',
                'recommendation': 'Review and manually validate low-confidence entities'
            })
    
    # Overall quality score (weighted average)
    report['overall_quality_score'] = (
        report['completeness_score'] * 0.4 +
        report['confidence_distribution'].get('mean', 0) * 0.6
    ) if report['confidence_distribution'] else report['completeness_score'] * 0.4
    
    # General recommendations
    if report['completeness_score'] < 0.8:
        report['recommendations'].append("Complete labeling for all uploaded documents")
    
    if report['confidence_distribution'].get('mean', 1) < 0.8:
        report['recommendations'].append("Review and improve labels with low confidence scores")
    
    return report


def show_contextual_pii_interface():
    """
    3-Step Contextual PII Processing Interface
    Solves the core problem: Extract data subject PII while ignoring professional PII
    """
    # Ensure session state is initialized
    initialize_session_state()
    
    st.markdown("### 🎯 3-Step Contextual PII Pipeline")
    
    # Problem explanation
    st.info("""
    **🧩 The Real Problem:** Extract PII of the data subject (person whose consent is needed) while ignoring PII of professionals (doctors, lawyers, staff).
    
    **🎯 3-Step Solution:**
    1. **Document Classification** - Identify domain (health, legal, HR, etc.)
    2. **Prompt Routing** - Use domain-specific prompts for better accuracy  
    3. **Role-Based Extraction** - Distinguish data subject vs professional PII
    """)
    
    if not CONTEXTUAL_PII_AVAILABLE:
        st.error("❌ Contextual PII Pipeline not available. Please check the installation.")
        return
        
    if not auth.has_permission('write'):
        st.warning("Contextual PII processing requires write permissions.")
        return
    
    # Initialize pipeline
    if 'contextual_pipeline' not in st.session_state:
        st.session_state.contextual_pipeline = ContextualPIIPipeline()
    
    # Document selection
    st.markdown("#### 📄 Select Document to Process")
    
    if not st.session_state.phase0_dataset:
        st.warning("No documents uploaded. Please upload documents in the 'File Upload' tab first.")
        return
    
    # Create document selection dropdown
    doc_options = {}
    for doc in st.session_state.phase0_dataset:
        doc_name = doc.get('name', 'Unknown')
        doc_id = doc.get('id', 'unknown')
        doc_options[f"{doc_name} (ID: {doc_id[:8]})"] = doc
    
    selected_doc_key = st.selectbox(
        "Choose a document to process:",
        options=list(doc_options.keys()),
        help="Select a document for contextual PII extraction"
    )
    
    if not selected_doc_key:
        return
        
    selected_doc = doc_options[selected_doc_key]
    
    # Display document info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Document", selected_doc.get('name', 'Unknown'))
    with col2:
        st.metric("Size", f"{selected_doc.get('size', 0)/1024:.1f} KB")
    with col3:
        domain = selected_doc.get('metadata', {}).get('domain', 'Unknown')
        st.metric("Current Domain", domain)
    
    # Processing options
    st.markdown("#### ⚙️ Processing Options")
    
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.75,
            help="Minimum confidence for auto-classification"
        )
    with col2:
        model_choice = st.selectbox(
            "LLM Model",
            ["anthropic/claude-3-5-sonnet-20241022", "anthropic/claude-3-5-haiku-20241022", "openai/gpt-4o", "openai/gpt-4o-mini", "🧪 TEST MODE (No API)"],
            help="Choose the model for processing. Test mode works without API calls."
        )
    
    # API and Processing Info
    st.info("""
    💡 **Processing Options:**
    - **Claude/OpenAI Models**: Use real API calls for actual document processing
    - **🧪 TEST MODE**: Try the pipeline without API calls (demonstration only)
    - **Troubleshooting**: If you get errors, TEST MODE can help verify the interface works
    """)
    
    st.warning("""
    ⚠️ **API Quota Notice**: If you see "quota exceeded" errors:
    - **Anthropic**: Check your usage at https://console.anthropic.com/
    - **OpenAI**: Check your billing at https://platform.openai.com/account/billing
    - **Alternative**: Try different model providers or use TEST MODE for demonstration
    """)
    
    # Process button
    if st.button("🚀 Run 3-Step Contextual PII Extraction", type="primary"):
        
        # Convert document to image data
        try:
            # Convert document content to images for processing
            content_str = selected_doc['content']
            
            if isinstance(content_str, str):
                # content_str is already a base64 string, decode it directly
                file_content = base64.b64decode(content_str)
            else:
                # content_str is bytes, decode to string first then base64 decode
                file_content = base64.b64decode(content_str.decode('utf-8'))
            
            file_extension = os.path.splitext(selected_doc['name'])[1]
            
            # Convert document to images
            images = convert_document_to_images(file_content, file_extension, "")
            
            if not images:
                st.error("❌ Could not convert document to images for processing. Please check if the document format is supported.")
                return
                
            # Use the first image for processing
            image_base64 = images[0]  # This is a base64 string from convert_document_to_images
            
            # Convert base64 string back to bytes for the contextual PII pipeline
            # (The pipeline expects bytes, but convert_document_to_images returns base64 strings)
            image_data = base64.b64decode(image_base64)
            
            with st.spinner("Processing through 3-step pipeline..."):
                
                # Create progress indicators for the 3 steps
                progress_col1, progress_col2, progress_col3 = st.columns(3)
                
                with progress_col1:
                    st.markdown("**Step 1: Classification**")
                    step1_placeholder = st.empty()
                    step1_placeholder.info("🔄 Processing...")
                
                with progress_col2:
                    st.markdown("**Step 2: Prompt Routing**")
                    step2_placeholder = st.empty()
                    step2_placeholder.info("⏳ Waiting...")
                    
                with progress_col3:
                    st.markdown("**Step 3: PII Extraction**")
                    step3_placeholder = st.empty()
                    step3_placeholder.info("⏳ Waiting...")
                
                # Run the pipeline - use test mode if selected
                if model_choice == "🧪 TEST MODE (No API)":
                    result = st.session_state.contextual_pipeline.process_document_test_mode(
                        document_id=selected_doc.get('id')
                    )
                else:
                    result = st.session_state.contextual_pipeline.process_document(
                        image_data=image_data,
                        document_id=selected_doc.get('id')
                    )
                
                # Update step indicators
                step1_placeholder.success(f"✅ {result.classification.domain.value.title()} ({result.classification.confidence:.2f})")
                
                strategy = "Domain-specific" if result.classification.confidence >= 0.85 else "General"
                step2_placeholder.success(f"✅ {strategy} prompt")
                
                step3_placeholder.success(f"✅ {len(result.data_subject_entities)} subject, {len(result.professional_entities)} professional")
                
                # Show test mode notice
                if model_choice == "🧪 TEST MODE (No API)":
                    st.info("🧪 **Test Mode Results**: These are demonstration results. No actual document processing occurred.")
                
                # Display detailed results
                st.markdown("---")
                st.markdown("### 📊 Processing Results")
                
                # Classification results
                st.markdown("#### Step 1: Document Classification")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Domain", result.classification.domain.value.title())
                with col2:
                    st.metric("Confidence", f"{result.classification.confidence:.3f}")
                with col3:
                    confidence_level = "High" if result.classification.confidence >= 0.85 else "Medium" if result.classification.confidence >= 0.65 else "Low"
                    color = "🟢" if confidence_level == "High" else "🟡" if confidence_level == "Medium" else "🔴"
                    st.metric("Level", f"{color} {confidence_level}")
                
                st.text_area("Classification Reasoning", result.classification.reasoning, height=100)
                
                # PII Extraction Results
                st.markdown("#### Step 3: Contextual PII Extraction Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📋 Total Entities", len(result.all_entities))
                with col2:
                    st.metric("👤 Data Subject PII", len(result.data_subject_entities), 
                            help="PII requiring consent")
                with col3:
                    st.metric("🏥 Professional PII", len(result.professional_entities),
                            help="PII to ignore (doctors, lawyers, etc.)")
                with col4:
                    unknown_entities = len([e for e in result.all_entities if e.role == "unknown"])
                    st.metric("❓ Unknown Role", unknown_entities,
                            help="Entities needing manual review")
                    
                # Data Subject PII (Most Important)
                if result.data_subject_entities:
                    st.markdown("##### 🎯 Data Subject PII (Requires Consent)")
                    ds_df = pd.DataFrame([
                        {
                            "Type": entity.type,
                            "Value": entity.value,
                            "Confidence": f"{entity.confidence:.3f}",
                            "Context": entity.context[:50] + "..." if len(entity.context) > 50 else entity.context,
                            "Flags": ", ".join(entity.flags) if entity.flags else "None"
                        }
                        for entity in result.data_subject_entities
                    ])
                    st.dataframe(ds_df, use_container_width=True)
                else:
                    st.warning("⚠️ No data subject PII identified. This may indicate a processing issue.")
                    
                # Professional PII (Excluded)
                if result.professional_entities:
                    st.markdown("##### 🏥 Professional PII (Excluded from Consent)")
                    prof_df = pd.DataFrame([
                        {
                            "Type": entity.type,
                            "Value": entity.value,
                            "Role Context": entity.context[:50] + "..." if len(entity.context) > 50 else entity.context
                        }
                        for entity in result.professional_entities
                    ])
                    st.dataframe(prof_df, use_container_width=True)
                    
                # Quality Assessment
                quality = st.session_state.contextual_pipeline.validate_extraction_quality(result)
                
                st.markdown("#### 🎯 Quality Assessment")
                col1, col2, col3 = st.columns(3)
                with col1:
                    color = "🟢" if quality['grade'] in ['A', 'B'] else "🟡" if quality['grade'] == 'C' else "🔴"
                    st.metric("Quality Grade", f"{color} {quality['grade']}")
                with col2:
                    st.metric("Quality Score", f"{quality['quality_score']:.2f}")
                with col3:
                    review_needed = "Yes" if quality['requires_review'] else "No"
                    color = "🔴" if quality['requires_review'] else "🟢"
                    st.metric("Review Needed", f"{color} {review_needed}")
                
                if quality['recommendations']:
                    st.markdown("**Recommendations:**")
                    for rec in quality['recommendations']:
                        st.write(f"• {rec}")
                    
                # Confidence Flags
                if result.confidence_flags:
                    st.markdown("#### ⚠️ Processing Flags")
                    for flag in result.confidence_flags:
                        flag_color = "🔴" if "failed" in flag else "🟡"
                        st.write(f"{flag_color} {flag.replace('_', ' ').title()}")
                
                # Save results option
                st.markdown("#### 💾 Save Results")
                if st.button("Save Contextual PII Results"):
                    # Add contextual results to document metadata
                    selected_doc['contextual_pii_results'] = {
                        'classification': {
                            'domain': result.classification.domain.value,
                            'confidence': result.classification.confidence,
                            'reasoning': result.classification.reasoning
                        },
                        'data_subject_entities': [
                            {
                                'type': e.type,
                                'value': e.value,
                                'confidence': e.confidence,
                                'context': e.context
                            }
                            for e in result.data_subject_entities
                        ],
                        'professional_entities': [
                            {
                                'type': e.type,
                                'value': e.value,
                                'context': e.context
                            }
                            for e in result.professional_entities
                        ],
                        'quality_assessment': quality,
                        'processing_metadata': result.processing_metadata
                    }
                    st.success("✅ Contextual PII results saved to document!")
                        
        except Exception as e:
            error_message = str(e)
            logger.error(f"Contextual PII processing failed: {e}", exc_info=True)
            
            if "quota" in error_message.lower() or "429" in error_message:
                st.error("🚨 **API Quota Exceeded**")
                st.info("""
                **How to fix this:**
                1. **Anthropic**: Check your usage at https://console.anthropic.com/
                2. **OpenAI**: Add credits at https://platform.openai.com/account/billing
                3. **Try different model**: Switch between Claude and OpenAI models
                4. **Wait**: Quotas may reset hourly/daily depending on your plan
                5. **Use TEST MODE**: Try the test mode for demonstration
                """)
                st.warning(f"Technical details: {error_message}")
            elif "bytes-like object" in error_message:
                st.error("🔧 **Document Processing Error**")
                st.info("""
                **This appears to be a data format issue. Try:**
                1. **Re-upload the document**: Sometimes re-uploading fixes format issues
                2. **Different file format**: Try converting to PDF if using other formats
                3. **Use TEST MODE**: Test the pipeline functionality without processing your document
                4. **Check file integrity**: Ensure the document isn't corrupted
                """)
                st.warning(f"Technical details: {error_message}")
            else:
                st.error(f"❌ Processing failed: {error_message}")
                st.info("💡 **Troubleshooting suggestions:**")
                st.write("1. Try **TEST MODE** to verify the pipeline works")
                st.write("2. Check if the document is supported (PDF, DOCX, images)")
                st.write("3. Try a different model from the dropdown")
                st.write("4. Re-upload the document if needed")
    
    # Show existing contextual results if available
    if selected_doc.get('contextual_pii_results'):
        st.markdown("#### 📋 Previous Contextual PII Results")
        prev_results = selected_doc['contextual_pii_results']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Previous Domain", prev_results['classification']['domain'].title())
        with col2:
            st.metric("Data Subject Entities", len(prev_results['data_subject_entities']))
        with col3:
            st.metric("Professional Entities", len(prev_results['professional_entities']))
        
        if st.button("🔄 Show Previous Results Details"):
            st.json(prev_results)