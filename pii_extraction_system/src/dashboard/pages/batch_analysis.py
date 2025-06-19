"""
Batch Analysis Page - Bulk document processing and analysis

This page provides capabilities for bulk document processing and results overview,
essential for handling large-scale document processing scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

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

from dashboard.utils import session_state, ui_components, auth
from core.pipeline import PIIExtractionPipeline
from utils.document_processor import DocumentProcessor

def show_page():
    """Main batch analysis page"""
    st.markdown('<div class="section-header">📊 Batch Analysis</div>', 
                unsafe_allow_html=True)
    st.markdown("Process and analyze multiple documents in bulk operations.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Main layout
    tab1, tab2, tab3 = st.tabs([
        "📁 Batch Upload",
        "📊 Batch Results", 
        "📈 Analytics Dashboard"
    ])
    
    with tab1:
        show_batch_upload()
    
    with tab2:
        show_batch_results()
    
    with tab3:
        show_analytics_dashboard()

def show_batch_upload():
    """Batch document upload and processing interface"""
    st.markdown("### Batch Document Upload")
    
    if not auth.has_permission('write'):
        st.warning("Batch processing requires write permissions.")
        return
    
    # Password input for protected documents
    st.markdown("#### Document Password (if required)")
    st.text_input(
        "Enter password for protected documents",
        type="password",
        value="Hubert",
        help="Leave blank for unprotected documents. Default password 'Hubert' is pre-filled.",
        key="batch_password"
    )
    
    # Batch upload interface
    uploaded_files = st.file_uploader(
        "Upload multiple documents for batch processing",
        type=['pdf', 'docx', 'xlsx', 'xls', 'jpg', 'png'],
        accept_multiple_files=True,
        help="Select multiple files to process in batch"
    )
    
    if uploaded_files:
        st.success(f"Selected {len(uploaded_files)} files for batch processing")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### Processing Options")
            
            processing_model = st.selectbox(
                "Select Processing Model",
                ['Ensemble Method', 'Rule-Based Extractor', 'spaCy NER', 'Transformers NER', 'Layout-Aware NER'],
                help="Choose which PII extraction model(s) to use"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
            
            # OCR Engine Selection
            st.markdown("**OCR Settings (for Images & PDFs):**")
            
            # Check LLM availability
            import os
            llm_available = bool(os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API") or os.getenv("ANTHROPIC_API_KEY"))
            
            ocr_engine_options = ['tesseract', 'easyocr', 'both']
            if llm_available:
                ocr_engine_options.extend(['llm', 'llm+traditional'])
            
            ocr_engine = st.selectbox(
                "OCR Engine",
                ocr_engine_options,
                index=0,
                help="Choose OCR method: Traditional or AI-powered LLM OCR"
            )
            
            if ocr_engine in ['easyocr', 'both']:
                use_gpu = st.checkbox("Use GPU for EasyOCR", value=False)
            else:
                use_gpu = False
            
            # LLM OCR options for batch processing
            if ocr_engine in ['llm', 'llm+traditional'] and llm_available:
                st.markdown("**🤖 LLM OCR Batch Settings:**")
                
                # Model selection
                available_models = []
                if os.getenv("OPENAI_API_KEY"):
                    available_models.extend(["gpt-4o-mini", "gpt-3.5-turbo"])
                if os.getenv("DEEPSEEK_API"):
                    available_models.append("deepseek-chat")
                if os.getenv("ANTHROPIC_API_KEY"):
                    available_models.extend(["claude-3-haiku"])
                
                llm_model = st.selectbox(
                    "LLM Model",
                    available_models,
                    index=0 if available_models else None,
                    help="AI model for batch OCR processing"
                )
                
                # Batch-specific cost controls
                max_total_cost = st.slider(
                    "Max Total Batch Cost ($)",
                    0.10, 10.0, 1.0, 0.10,
                    help="Maximum total cost for entire batch"
                )
                
                max_cost_per_doc = st.slider(
                    "Max Cost per Document ($)",
                    0.01, 0.50, 0.10, 0.01,
                    help="Maximum cost per single document"
                )
                
                # Show cost estimation for batch
                if llm_model and uploaded_files:
                    cost_per_page = {
                        "gpt-4o-mini": 0.00015,
                        "gpt-3.5-turbo": 0.0005,
                        "deepseek-chat": 0.0001,
                        "claude-3-haiku": 0.00025
                    }.get(llm_model, 0.0002)
                    
                    estimated_batch_cost = len(uploaded_files) * cost_per_page
                    st.info(f"💰 Est. batch cost: ~${estimated_batch_cost:.4f} ({len(uploaded_files)} files)")
                    
                    if estimated_batch_cost > max_total_cost:
                        st.warning(f"⚠️ Estimated cost exceeds batch limit!")
            
            parallel_processing = st.checkbox("Enable Parallel Processing", value=True)
            
        with col2:
            st.markdown("#### File Preview")
            for i, file in enumerate(uploaded_files[:5]):  # Show first 5
                st.write(f"• {file.name} ({file.size:,} bytes)")
            
            if len(uploaded_files) > 5:
                st.write(f"... and {len(uploaded_files) - 5} more files")
        
        # Batch processing controls
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Start Batch Processing", type="primary"):
                start_batch_processing(uploaded_files, processing_model, confidence_threshold, ocr_engine, use_gpu)
        
        with col2:
            if st.button("Clear Selection"):
                st.rerun()
        
        with col3:
            if 'batch_processing_status' in st.session_state:
                status = st.session_state.batch_processing_status
                if status.get('is_processing'):
                    if st.button("Cancel Processing"):
                        cancel_batch_processing()

def start_batch_processing(files: List, model: str, threshold: float, ocr_engine: str, use_gpu: bool):
    """Start batch processing of uploaded files"""
    batch_id = f"batch_{len(st.session_state.get('batch_jobs', []))}"
    
    # Initialize batch job
    batch_job = {
        'batch_id': batch_id,
        'total_files': len(files),
        'processed_files': 0,
        'status': 'processing',
        'model': model,
        'threshold': threshold,
        'ocr_engine': ocr_engine,
        'ocr_use_gpu': use_gpu,
        'start_time': pd.Timestamp.now(),
        'results': []
    }
    
    # Store batch job
    if 'batch_jobs' not in st.session_state:
        st.session_state.batch_jobs = []
    
    st.session_state.batch_jobs.append(batch_job)
    st.session_state.batch_processing_status = {'is_processing': True, 'current_batch': batch_id}
    
    # Real PII processing pipeline
    with st.spinner(f"Processing {len(files)} files..."):
        process_batch_real(batch_job, files, model, threshold, ocr_engine, use_gpu)
    
    st.success(f"Batch processing completed! Processed {len(files)} files.")
    st.session_state.batch_processing_status = {'is_processing': False}

def process_batch_real(batch_job: Dict, files: List, model: str, threshold: float, ocr_engine: str, use_gpu: bool):
    """Real batch processing using PII extraction pipeline"""
    import tempfile
    import os
    from core.config import settings
    
    # Map UI model names to internal model names
    model_mapping = {
        'Ensemble Method': ["rule_based", "ner", "layout_aware"],
        'Rule-Based Extractor': ["rule_based"],
        'spaCy NER': ["ner"],
        'Transformers NER': ["ner"],
        'Layout-Aware NER': ["layout_aware"]
    }
    
    enabled_models = model_mapping.get(model, ["rule_based", "ner", "layout_aware"])
    
    # Get password for batch processing
    batch_password = st.session_state.get('batch_password', '')
    
    for i, file in enumerate(files):
        try:
            # Save file temporarily for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp_file:
                tmp_file.write(file.read())
                tmp_file_path = tmp_file.name
            
            try:
                # Process with PII pipeline
                pipeline = PIIExtractionPipeline(models=enabled_models)
                extraction_result = pipeline.extract_from_file(tmp_file_path)
                
                # Extract text content for analysis using selected OCR engine
                # Temporarily override settings with user selection
                original_ocr_engine = settings.processing.ocr_engine
                original_ocr_gpu = settings.processing.easyocr_use_gpu
                
                settings.processing.ocr_engine = ocr_engine
                settings.processing.easyocr_use_gpu = use_gpu
                
                try:
                    doc_processor = DocumentProcessor()
                    processed_doc = doc_processor.process_document(tmp_file_path, batch_password if batch_password else None)
                    text_content = processed_doc.get('raw_text', '')
                    
                    # Store OCR metadata
                    ocr_metadata = {
                        'ocr_engine': processed_doc.get('ocr_engine', ocr_engine),
                        'alternative_text': processed_doc.get('alternative_text', {}),
                        'bounding_boxes': processed_doc.get('bounding_boxes', [])
                    }
                    
                finally:
                    # Restore original settings
                    settings.processing.ocr_engine = original_ocr_engine
                    settings.processing.easyocr_use_gpu = original_ocr_gpu
                
                # Filter entities by confidence threshold
                filtered_entities = []
                category_counts = {}
                confidence_scores = []
                
                for entity in extraction_result.pii_entities:
                    if entity.confidence >= threshold:
                        filtered_entities.append({
                            'type': entity.pii_type.upper(),
                            'text': entity.text,
                            'start': entity.start_pos,
                            'end': entity.end_pos,
                            'confidence': entity.confidence,
                            'context': entity.context,
                            'extractor': entity.extractor
                        })
                        
                        # Count by category
                        entity_type = entity.pii_type.upper()
                        category_counts[entity_type] = category_counts.get(entity_type, 0) + 1
                        confidence_scores.append(entity.confidence)
                
                # Create detailed result
                file_result = {
                    'file_name': file.name,
                    'file_size': file.size,
                    'entities_found': len(filtered_entities),
                    'processing_time': extraction_result.processing_time,
                    'confidence_avg': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                    'categories': category_counts,
                    'pii_entities': filtered_entities,  # Store actual extracted entities
                    'text_content': text_content,
                    'total_entities_before_filter': len(extraction_result.pii_entities),
                    'confidence_threshold': threshold,
                    'selected_model': model,
                    'enabled_models': enabled_models,
                    'ocr_engine': ocr_engine,
                    'ocr_metadata': ocr_metadata,
                    'status': 'completed'
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
        except Exception as e:
            # Handle processing errors
            file_result = {
                'file_name': file.name,
                'file_size': file.size,
                'entities_found': 0,
                'processing_time': 0,
                'confidence_avg': 0,
                'categories': {},
                'pii_entities': [],
                'text_content': '',
                'total_entities_before_filter': 0,
                'confidence_threshold': threshold,
                'selected_model': model,
                'enabled_models': enabled_models,
                'status': 'error',
                'error_message': str(e)
            }
        
        batch_job['results'].append(file_result)
        batch_job['processed_files'] = i + 1
        
        # Update progress
        if i % 2 == 0 or i == len(files) - 1:  # Update every 2 files or on last file
            progress = (i + 1) / len(files)
            st.progress(progress)
            st.text(f"Processing: {file.name} ({i + 1}/{len(files)})")

def cancel_batch_processing():
    """Cancel ongoing batch processing"""
    st.session_state.batch_processing_status = {'is_processing': False}
    st.warning("Batch processing cancelled.")

def show_detailed_file_analysis(file_result: Dict):
    """Show detailed analysis for a single file (similar to document processing)"""
    st.markdown(f"### 📄 Analysis: {file_result['file_name']}")
    
    # File overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Entities", file_result.get('entities_found', 0))
    
    with col2:
        st.metric("Processing Time", f"{file_result.get('processing_time', 0):.2f}s")
    
    with col3:
        avg_confidence = file_result.get('confidence_avg', 0)
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col4:
        selected_model = file_result.get('selected_model', 'Unknown')
        st.metric("Model", selected_model.split(' (')[0] if selected_model else 'Unknown')
    
    # Show filtering information if applicable
    total_before_filter = file_result.get('total_entities_before_filter', file_result.get('entities_found', 0))
    confidence_threshold = file_result.get('confidence_threshold', 0.5)
    
    if total_before_filter > file_result.get('entities_found', 0):
        filtered_count = total_before_filter - file_result.get('entities_found', 0)
        st.info(f"📊 Showing {file_result.get('entities_found', 0)} entities (filtered out {filtered_count} below {confidence_threshold:.0%} confidence)")
    
    # Show error if processing failed
    if file_result.get('status') == 'error':
        st.error(f"❌ Processing failed: {file_result.get('error_message', 'Unknown error')}")
        return
    
    pii_entities = file_result.get('pii_entities', [])
    
    if not pii_entities:
        st.info("No PII entities detected in this file")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document View", "📋 PII Entities", "📊 Statistics", "⚙️ Export"])
    
    with tab1:
        show_file_document_view(file_result)
    
    with tab2:
        show_file_pii_entities(file_result)
    
    with tab3:
        show_file_statistics(file_result)
    
    with tab4:
        show_file_export_options(file_result)

def show_file_document_view(file_result: Dict):
    """Show document with PII highlighting for batch file"""
    st.markdown("### Document with PII Highlighting")
    
    text_content = file_result.get('text_content', '')
    pii_entities = file_result.get('pii_entities', [])
    
    if not text_content:
        st.warning("No text content available")
        return
    
    # Show highlighted text or plain text if no entities
    if pii_entities:
        highlighted_html = ui_components.display_pii_highlights(text_content, pii_entities)
        st.markdown(highlighted_html, unsafe_allow_html=True)
    else:
        st.text_area("Document Text", text_content, height=400)

def show_file_pii_entities(file_result: Dict):
    """Show PII entities table for batch file"""
    st.markdown("### Detected PII Entities")
    
    pii_entities = file_result.get('pii_entities', [])
    
    if not pii_entities:
        st.info("No PII entities detected")
        return
    
    # Convert to DataFrame for display
    entities_data = []
    for entity in pii_entities:
        entities_data.append({
            'Type': entity.get('type', ''),
            'Text': entity.get('text', ''),
            'Confidence': f"{entity.get('confidence', 0):.2%}",
            'Extractor': entity.get('extractor', ''),
            'Position': f"{entity.get('start', 0)}-{entity.get('end', 0)}"
        })
    
    df = pd.DataFrame(entities_data)
    st.dataframe(df, use_container_width=True)
    
    # Show entities by type
    st.markdown("#### Entities by Type")
    entity_types = {}
    for entity in pii_entities:
        entity_type = entity.get('type', 'UNKNOWN')
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(entity.get('text', ''))
    
    for entity_type, texts in entity_types.items():
        with st.expander(f"{entity_type} ({len(texts)} found)"):
            for i, text in enumerate(texts, 1):
                st.write(f"{i}. {text}")

def show_file_statistics(file_result: Dict):
    """Show statistics for batch file"""
    st.markdown("### File Statistics")
    
    pii_entities = file_result.get('pii_entities', [])
    
    if not pii_entities:
        st.info("No statistics available")
        return
    
    # Count by category
    category_counts = {}
    confidence_scores = []
    extractor_counts = {}
    
    for entity in pii_entities:
        entity_type = entity.get('type', 'UNKNOWN')
        category_counts[entity_type] = category_counts.get(entity_type, 0) + 1
        confidence_scores.append(entity.get('confidence', 0))
        
        extractor = entity.get('extractor', 'unknown')
        extractor_counts[extractor] = extractor_counts.get(extractor, 0) + 1
    
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
    
    # Extractor performance
    if len(extractor_counts) > 1:
        st.markdown("#### Extractor Performance")
        for extractor, count in extractor_counts.items():
            st.write(f"• {extractor}: {count} entities")

def show_file_export_options(file_result: Dict):
    """Show export options for single file"""
    st.markdown("### Export File Results")
    
    filename_base = f"pii_results_{file_result['file_name'].split('.')[0]}"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # JSON export
        ui_components.export_data(file_result, filename_base, 'json')
    
    with col2:
        # CSV export (entities only)
        pii_entities = file_result.get('pii_entities', [])
        if pii_entities:
            df = pd.DataFrame(pii_entities)
            ui_components.export_data(df, f"{filename_base}_entities", 'csv')
    
    with col3:
        # Text export
        text_content = file_result.get('text_content', '')
        if text_content:
            ui_components.export_data(text_content, f"{filename_base}_text", 'txt')

def show_batch_results():
    """Display batch processing results"""
    st.markdown("### Batch Processing Results")
    
    batch_jobs = st.session_state.get('batch_jobs', [])
    
    if not batch_jobs:
        st.info("No batch processing jobs found. Upload documents in the Batch Upload tab.")
        return
    
    # Job selection
    job_options = {job['batch_id']: f"{job['batch_id']} ({job['total_files']} files)" 
                   for job in batch_jobs}
    
    selected_job_id = st.selectbox(
        "Select batch job to view:",
        options=list(job_options.keys()),
        format_func=lambda x: job_options[x]
    )
    
    if not selected_job_id:
        return
    
    # Find selected job
    selected_job = None
    for job in batch_jobs:
        if job['batch_id'] == selected_job_id:
            selected_job = job
            break
    
    if not selected_job:
        st.error("Selected job not found.")
        return
    
    # Job overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files", selected_job['total_files'])
    
    with col2:
        st.metric("Processed", selected_job['processed_files'])
    
    with col3:
        completion_rate = selected_job['processed_files'] / selected_job['total_files']
        st.metric("Completion", f"{completion_rate:.1%}")
    
    with col4:
        st.metric("Status", selected_job['status'].title())
    
    # Results analysis
    if selected_job['results']:
        st.markdown("#### Detailed File Analysis")
        
        # File selection for detailed view
        file_results = selected_job['results']
        file_options = {i: f"{result['file_name']} ({result['entities_found']} entities)" 
                       for i, result in enumerate(file_results)}
        
        selected_file_idx = st.selectbox(
            "Select file for detailed analysis:",
            options=list(file_options.keys()),
            format_func=lambda x: file_options[x]
        )
        
        if selected_file_idx is not None:
            selected_file_result = file_results[selected_file_idx]
            show_detailed_file_analysis(selected_file_result)
        
        st.markdown("---")
        st.markdown("#### Batch Summary Table")
        
        df_results = pd.DataFrame(selected_job['results'])
        
        # Add summary columns
        if 'categories' in df_results.columns:
            df_results['total_entities'] = df_results['categories'].apply(
                lambda x: sum(x.values()) if isinstance(x, dict) else 0
            )
        
        # Display table
        display_columns = ['file_name', 'entities_found', 'processing_time', 'confidence_avg', 'status']
        if all(col in df_results.columns for col in display_columns):
            df_display = df_results[display_columns].copy()
            df_display['processing_time'] = df_display['processing_time'].round(2)
            df_display['confidence_avg'] = df_display['confidence_avg'].round(3)
            
            st.dataframe(df_display, use_container_width=True)
        
        # Export options
        st.markdown("#### Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ui_components.export_data(
                selected_job, 
                f"batch_results_{selected_job_id}", 
                'json'
            )
        
        with col2:
            if selected_job['results']:
                ui_components.export_data(
                    df_results, 
                    f"batch_summary_{selected_job_id}", 
                    'csv'
                )
        
        with col3:
            # Generate detailed report
            if st.button("Generate Report"):
                generate_batch_report(selected_job)

def generate_batch_report(job: Dict):
    """Generate comprehensive batch report"""
    st.markdown("#### Batch Processing Report")
    
    results = job.get('results', [])
    if not results:
        st.warning("No results to report.")
        return
    
    df = pd.DataFrame(results)
    
    # Summary statistics
    st.markdown("**Summary Statistics:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"• Total files processed: {len(results)}")
        st.write(f"• Total entities found: {df['entities_found'].sum()}")
        st.write(f"• Average entities per file: {df['entities_found'].mean():.1f}")
        st.write(f"• Average processing time: {df['processing_time'].mean():.2f}s")
    
    with col2:
        st.write(f"• Average confidence: {df['confidence_avg'].mean():.1%}")
        st.write(f"• Fastest processing: {df['processing_time'].min():.2f}s")
        st.write(f"• Slowest processing: {df['processing_time'].max():.2f}s")
        st.write(f"• Total processing time: {df['processing_time'].sum():.1f}s")
    
    # Performance insights
    st.markdown("**Performance Insights:**")
    
    # Files with high entity counts
    high_entity_files = df[df['entities_found'] > df['entities_found'].quantile(0.8)]
    if not high_entity_files.empty:
        st.write("Files with high PII content:")
        for _, row in high_entity_files.iterrows():
            st.write(f"  • {row['file_name']}: {row['entities_found']} entities")
    
    # Files with low confidence
    low_confidence_files = df[df['confidence_avg'] < 0.7]
    if not low_confidence_files.empty:
        st.write("Files with low confidence scores (may need review):")
        for _, row in low_confidence_files.iterrows():
            st.write(f"  • {row['file_name']}: {row['confidence_avg']:.1%} avg confidence")

def show_analytics_dashboard():
    """Show analytics dashboard for batch processing"""
    st.markdown("### Batch Analytics Dashboard")
    
    batch_jobs = st.session_state.get('batch_jobs', [])
    
    if not batch_jobs:
        st.info("No batch processing data available for analytics.")
        return
    
    # Aggregate all results
    all_results = []
    for job in batch_jobs:
        for result in job.get('results', []):
            result['batch_id'] = job['batch_id']
            result['model'] = job.get('model', 'Unknown')
            all_results.append(result)
    
    if not all_results:
        st.info("No processing results available for analytics.")
        return
    
    df_all = pd.DataFrame(all_results)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Files Processed", len(df_all))
    
    with col2:
        st.metric("Total Entities Found", df_all['entities_found'].sum())
    
    with col3:
        st.metric("Average Processing Time", f"{df_all['processing_time'].mean():.2f}s")
    
    with col4:
        st.metric("Average Confidence", f"{df_all['confidence_avg'].mean():.1%}")
    
    # Analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Entities Found Distribution")
        fig_entities = px.histogram(
            df_all, x='entities_found',
            title='Distribution of Entities Found per Document',
            nbins=20
        )
        st.plotly_chart(fig_entities, use_container_width=True)
    
    with col2:
        st.markdown("#### Processing Time Distribution")
        fig_time = px.histogram(
            df_all, x='processing_time',
            title='Distribution of Processing Times',
            nbins=20
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    # Model comparison
    if 'model' in df_all.columns and df_all['model'].nunique() > 1:
        st.markdown("#### Model Performance Comparison")
        
        model_stats = df_all.groupby('model').agg({
            'entities_found': ['mean', 'std'],
            'processing_time': ['mean', 'std'],
            'confidence_avg': ['mean', 'std']
        }).round(3)
        
        st.dataframe(model_stats, use_container_width=True)
        
        # Model performance chart
        fig_models = px.box(
            df_all, x='model', y='entities_found',
            title='Entities Found by Model'
        )
        st.plotly_chart(fig_models, use_container_width=True)
    
    # PII category analysis
    st.markdown("#### PII Category Analysis")
    
    # Aggregate category counts
    category_totals = {}
    for _, row in df_all.iterrows():
        categories = row.get('categories', {})
        if isinstance(categories, dict):
            for cat, count in categories.items():
                category_totals[cat] = category_totals.get(cat, 0) + count
    
    if category_totals:
        fig_categories = px.bar(
            x=list(category_totals.keys()),
            y=list(category_totals.values()),
            title='Total PII Entities by Category (All Batches)',
            labels={'x': 'PII Category', 'y': 'Total Count'}
        )
        st.plotly_chart(fig_categories, use_container_width=True)
    
    # Performance trends
    if len(batch_jobs) > 1:
        st.markdown("#### Performance Trends")
        
        # Create trend data
        trend_data = []
        for job in batch_jobs:
            if job.get('results'):
                job_df = pd.DataFrame(job['results'])
                trend_data.append({
                    'batch_id': job['batch_id'],
                    'avg_entities': job_df['entities_found'].mean(),
                    'avg_time': job_df['processing_time'].mean(),
                    'avg_confidence': job_df['confidence_avg'].mean(),
                    'total_files': len(job_df)
                })
        
        if trend_data:
            df_trends = pd.DataFrame(trend_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_trend1 = px.line(
                    df_trends, x='batch_id', y='avg_entities',
                    title='Average Entities Found per Batch',
                    markers=True
                )
                st.plotly_chart(fig_trend1, use_container_width=True)
            
            with col2:
                fig_trend2 = px.line(
                    df_trends, x='batch_id', y='avg_time',
                    title='Average Processing Time per Batch',
                    markers=True
                )
                st.plotly_chart(fig_trend2, use_container_width=True)