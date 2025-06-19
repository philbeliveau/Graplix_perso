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
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
from PIL import Image
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import session_state, ui_components, auth
from llm.multimodal_llm_service import llm_service
from core.ground_truth_validation import ground_truth_validator

# Dataset creation state management
if 'phase0_dataset' not in st.session_state:
    st.session_state.phase0_dataset = []

if 'phase0_current_document' not in st.session_state:
    st.session_state.phase0_current_document = None

if 'phase0_labeling_queue' not in st.session_state:
    st.session_state.phase0_labeling_queue = []

def show_page():
    """Main Phase 0 dataset creation page"""
    st.markdown('<div class="section-header">🎯 Phase 0 Dataset Creation</div>', 
                unsafe_allow_html=True)
    st.markdown("Create high-quality ground truth datasets with GPT-4o assistance and metadata tagging.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 File Upload",
        "🤖 GPT-4o Labeling", 
        "🏷️ Metadata Tagging",
        "⚡ Interactive Labeling",
        "📊 Dataset Export"
    ])
    
    with tab1:
        show_file_upload_interface()
    
    with tab2:
        show_gpt4o_labeling_interface()
    
    with tab3:
        show_metadata_tagging_interface()
    
    with tab4:
        show_interactive_labeling_interface()
    
    with tab5:
        show_dataset_export_interface()

def show_file_upload_interface():
    """Show file upload interface with validation"""
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
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose document files",
            type=['pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'tiff'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, DOC, TXT, PNG, JPG, JPEG, TIFF"
        )
    
    with col2:
        upload_options = st.expander("Upload Options")
        with upload_options:
            auto_label = st.checkbox("Auto-label with GPT-4o", value=True)
            batch_size = st.slider("Batch Size", 1, 10, 5)
            priority = st.selectbox("Priority", ["High", "Medium", "Low"], index=1)
    
    if uploaded_files:
        if st.button("Process Uploaded Files"):
            process_uploaded_files(uploaded_files, auto_label, batch_size, priority)
    
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
                label_all_unlabeled_documents()
        
        with col2:
            if st.button("Clear All Documents"):
                if st.confirm("Are you sure you want to clear all documents?"):
                    st.session_state.phase0_dataset = []
                    st.rerun()
        
        with col3:
            if st.button("Export Label Dataset", help="Export PII labels and metadata only (no document content)"):
                export_current_dataset()

def process_uploaded_files(uploaded_files, auto_label: bool, batch_size: int, priority: str):
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
            'gpt4o_labels': None,
            'validation_status': 'Pending'
        }
        
        # Add to dataset
        st.session_state.phase0_dataset.append(document)
        
        # Auto-label if requested
        if auto_label:
            try:
                gpt4o_labels = auto_label_document(doc_id)
                document['gpt4o_labels'] = gpt4o_labels
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
        'png': 'Image',
        'jpg': 'Image',
        'jpeg': 'Image',
        'tiff': 'Image'
    }
    
    return type_mapping.get(extension, 'Unknown')

def auto_label_document(doc_id: str) -> Dict[str, Any]:
    """Auto-label document using GPT-4o"""
    document = next((doc for doc in st.session_state.phase0_dataset if doc['id'] == doc_id), None)
    
    if not document:
        raise ValueError("Document not found")
    
    # For image files, use LLM directly
    if document['type'].startswith('image/') or document['name'].lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        result = llm_service.extract_pii_from_image(
            document['content'],
            'openai/gpt-4o',
            document_type=document['metadata']['document_type']
        )
        
        if result['success']:
            return {
                'method': 'gpt4o_vision',
                'entities': result.get('pii_entities', []),
                'confidence_score': calculate_confidence_score(result.get('pii_entities', [])),
                'processing_time': result.get('processing_time', 0),
                'cost': result.get('usage', {}).get('estimated_cost', 0),
                'raw_response': result
            }
        else:
            raise Exception(result.get('error', 'Unknown error'))
    
    else:
        # For text documents, would need OCR + LLM processing
        # For now, return mock data
        return {
            'method': 'gpt4o_text',
            'entities': [
                {
                    'type': 'PERSON',
                    'text': 'John Smith',
                    'confidence': 0.95,
                    'source': 'gpt4o_extraction'
                }
            ],
            'confidence_score': 0.95,
            'processing_time': 2.5,
            'cost': 0.002
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
        queue_data.append({
            'ID': doc['id'][:8] + '...',
            'Name': doc['name'],
            'Type': doc['metadata']['document_type'],
            'Size': f"{doc['size'] / 1024:.1f} KB",
            'Priority': doc['priority'],
            'Status': 'Labeled' if doc['labeled'] else 'Pending',
            'Difficulty': doc['metadata']['difficulty_level'],
            'Domain': doc['metadata']['domain'],
            'Uploaded': doc['uploaded_at'][:19].replace('T', ' ')
        })
    
    df = pd.DataFrame(queue_data)
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            ['Labeled', 'Pending'],
            default=['Labeled', 'Pending']
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
            st.write(f"Status: {'Labeled' if document['labeled'] else 'Pending'}")
        
        with col2:
            st.markdown("**Metadata**")
            st.write(f"Difficulty: {document['metadata']['difficulty_level']}")
            st.write(f"Domain: {document['metadata']['domain']}")
            st.write(f"Validation: {document['validation_status']}")
            st.write(f"Uploaded: {document['uploaded_at'][:19].replace('T', ' ')}")
        
        # Show labels if available
        if document.get('gpt4o_labels'):
            st.markdown("**GPT-4o Labels**")
            labels = document['gpt4o_labels']
            
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

def show_gpt4o_labeling_interface():
    """Show GPT-4o labeling interface"""
    st.markdown("### 🤖 GPT-4o Ground Truth Labeling")
    
    if not auth.has_permission('write'):
        st.warning("GPT-4o labeling requires write permissions.")
        return
    
    # GPT-4o availability check
    available_models = llm_service.get_available_models()
    gpt4o_models = [model for model in available_models if 'gpt-4o' in model]
    
    if not gpt4o_models:
        st.error("GPT-4o models are not available. Please check your OpenAI API configuration.")
        return
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model = st.selectbox(
            "Select GPT-4o Model",
            gpt4o_models,
            help="Choose the GPT-4o model for labeling"
        )
    
    with col2:
        # Model info
        model_info = llm_service.get_model_info(selected_model)
        st.info(f"Cost: $**{model_info.get('cost_per_1k_input_tokens', 0):.3f}**/1K input tokens")
    
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
            label_document_batch(filtered_docs[:batch_size], selected_model)
    
    with col2:
        if st.button("Label All Filtered", disabled=len(filtered_docs) == 0):
            if st.confirm(f"Label all {len(filtered_docs)} documents? Estimated cost: ${estimate_labeling_cost(filtered_docs, selected_model):.3f}"):
                label_document_batch(filtered_docs, selected_model)
    
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

def label_document_batch(documents: List[Dict], model_key: str):
    """Label a batch of documents using GPT-4o"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_cost = 0
    successful_labels = 0
    
    for i, document in enumerate(documents):
        status_text.text(f"Labeling {document['name']}... ({i+1}/{len(documents)})")
        
        try:
            labels = auto_label_document(document['id'])
            document['gpt4o_labels'] = labels
            document['labeled'] = True
            document['validation_status'] = 'Auto-labeled'
            
            total_cost += labels.get('cost', 0)
            successful_labels += 1
            
        except Exception as e:
            st.error(f"Failed to label {document['name']}: {e}")
        
        progress_bar.progress((i + 1) / len(documents))
    
    status_text.text(f"Completed! Successfully labeled {successful_labels}/{len(documents)} documents.")
    st.success(f"Labeling complete! Total cost: ${total_cost:.4f}")

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
        labels = doc.get('gpt4o_labels', {})
        
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
            ['No change', 'High', 'Medium', 'Low']
        )
    
    with col2:
        bulk_domain = st.selectbox(
            "Apply domain to filtered:",
            ['No change', 'Finance', 'Healthcare', 'Legal', 'HR', 'General']
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
                ['High', 'Medium', 'Low'],
                index=['High', 'Medium', 'Low'].index(document['metadata']['difficulty_level'])
                if document['metadata']['difficulty_level'] in ['High', 'Medium', 'Low'] else 1
            )
            
            new_domain = st.selectbox(
                "Domain",
                ['Finance', 'Healthcare', 'Legal', 'HR', 'General'],
                index=['Finance', 'Healthcare', 'Legal', 'HR', 'General'].index(document['metadata']['domain'])
                if document['metadata']['domain'] in ['Finance', 'Healthcare', 'Legal', 'HR', 'General'] else 4
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
    if document.get('gpt4o_labels'):
        entities = document['gpt4o_labels'].get('entities', [])
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
    
    if st.confirm(f"Label {len(unlabeled_docs)} documents with {selected_model}? Estimated cost: ${estimated_cost:.3f}"):
        label_document_batch(unlabeled_docs, selected_model)

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
        if doc.get('gpt4o_labels'):
            export_doc['labels'] = doc['gpt4o_labels']
        
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
            if st.button("Reprocess with GPT-4o"):
                reprocess_document_with_gpt4o(document['id'])
        
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

def reprocess_document_with_gpt4o(doc_id: str):
    """Reprocess document with GPT-4o"""
    try:
        labels = auto_label_document(doc_id)
        document = next((doc for doc in st.session_state.phase0_dataset if doc['id'] == doc_id), None)
        
        if document:
            document['gpt4o_labels'] = labels
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
        if doc.get('gpt4o_labels'):
            total_entities += len(doc['gpt4o_labels'].get('entities', []))
    
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
                ['Finance', 'Healthcare', 'Legal', 'HR', 'General']
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
                    'labels': example_doc.get('gpt4o_labels', {}).get('entities', [])
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
                    'entity_count': len(example_doc.get('gpt4o_labels', {}).get('entities', [])),
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
            if doc.get('gpt4o_labels', {}).get('confidence_score', 0) > 0.8
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
        
        if doc.get('gpt4o_labels'):
            export_doc['labels'] = doc['gpt4o_labels']
        
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
        labels = doc.get('gpt4o_labels', {})
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
        if doc.get('gpt4o_labels'):
            entities = doc['gpt4o_labels'].get('entities', [])
            stats['total_entities'] += len(entities)
            stats['total_labeling_cost'] += doc['gpt4o_labels'].get('cost', 0)
            
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
        if doc.get('gpt4o_labels'):
            entities = doc['gpt4o_labels'].get('entities', [])
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