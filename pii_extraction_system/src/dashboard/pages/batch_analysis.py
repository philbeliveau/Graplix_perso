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

from dashboard.utils import session_state, ui_components, auth

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
                ['Rule-Based Extractor', 'spaCy NER', 'Transformers NER', 'Ensemble Method']
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
            
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
                start_batch_processing(uploaded_files, processing_model, confidence_threshold)
        
        with col2:
            if st.button("Clear Selection"):
                st.rerun()
        
        with col3:
            if 'batch_processing_status' in st.session_state:
                status = st.session_state.batch_processing_status
                if status.get('is_processing'):
                    if st.button("Cancel Processing"):
                        cancel_batch_processing()

def start_batch_processing(files: List, model: str, threshold: float):
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
        'start_time': pd.Timestamp.now(),
        'results': []
    }
    
    # Store batch job
    if 'batch_jobs' not in st.session_state:
        st.session_state.batch_jobs = []
    
    st.session_state.batch_jobs.append(batch_job)
    st.session_state.batch_processing_status = {'is_processing': True, 'current_batch': batch_id}
    
    # Mock processing (would integrate with actual processing pipeline)
    with st.spinner(f"Processing {len(files)} files..."):
        process_batch_mock(batch_job, files)
    
    st.success(f"Batch processing completed! Processed {len(files)} files.")
    st.session_state.batch_processing_status = {'is_processing': False}

def process_batch_mock(batch_job: Dict, files: List):
    """Mock batch processing for demonstration"""
    import time
    
    for i, file in enumerate(files):
        # Mock processing delay
        time.sleep(0.1)  # Simulate processing time
        
        # Mock results
        mock_result = {
            'file_name': file.name,
            'file_size': file.size,
            'entities_found': np.random.randint(2, 15),
            'processing_time': np.random.uniform(1.0, 5.0),
            'confidence_avg': np.random.uniform(0.6, 0.95),
            'categories': {
                'PERSON': np.random.randint(0, 5),
                'EMAIL': np.random.randint(0, 3),
                'PHONE': np.random.randint(0, 2),
                'ADDRESS': np.random.randint(0, 2)
            }
        }
        
        batch_job['results'].append(mock_result)
        batch_job['processed_files'] = i + 1
        
        # Update progress
        if i % 5 == 0:  # Update every 5 files
            st.progress((i + 1) / len(files))

def cancel_batch_processing():
    """Cancel ongoing batch processing"""
    st.session_state.batch_processing_status = {'is_processing': False}
    st.warning("Batch processing cancelled.")

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
    
    # Results table
    if selected_job['results']:
        st.markdown("#### Processing Results")
        
        df_results = pd.DataFrame(selected_job['results'])
        
        # Add summary columns
        if 'categories' in df_results.columns:
            df_results['total_entities'] = df_results['categories'].apply(
                lambda x: sum(x.values()) if isinstance(x, dict) else 0
            )
        
        # Display table
        display_columns = ['file_name', 'entities_found', 'processing_time', 'confidence_avg']
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