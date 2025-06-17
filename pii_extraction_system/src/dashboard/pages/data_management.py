"""
Data Management Page - Dataset and annotation management tools

This page provides comprehensive tools for managing datasets, annotations,
and model training data throughout the system lifecycle.
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
    """Main data management page"""
    st.markdown('<div class="section-header">🗄️ Data Management</div>', 
                unsafe_allow_html=True)
    st.markdown("Manage datasets, annotations, and training data for model improvement.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 Dataset Overview",
        "✏️ Annotation Management", 
        "🎯 Active Learning",
        "📊 Data Quality"
    ])
    
    with tab1:
        show_dataset_overview()
    
    with tab2:
        show_annotation_management()
    
    with tab3:
        show_active_learning()
    
    with tab4:
        show_data_quality()

def show_dataset_overview():
    """Show dataset overview and statistics"""
    st.markdown("### Dataset Overview")
    
    # Generate mock dataset statistics
    dataset_stats = get_dataset_statistics()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", dataset_stats['total_documents'])
    
    with col2:
        st.metric("Annotated Documents", dataset_stats['annotated_documents'])
    
    with col3:
        st.metric("Total PII Entities", dataset_stats['total_entities'])
    
    with col4:
        completion_rate = dataset_stats['annotated_documents'] / dataset_stats['total_documents']
        st.metric("Annotation Progress", f"{completion_rate:.1%}")
    
    # Dataset composition
    st.markdown("#### Dataset Composition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Document types
        doc_types = dataset_stats['document_types']
        if doc_types:
            fig_types = px.pie(
                values=list(doc_types.values()),
                names=list(doc_types.keys()),
                title="Document Types Distribution"
            )
            st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:
        # PII categories
        pii_categories = dataset_stats['pii_categories']
        if pii_categories:
            fig_categories = px.bar(
                x=list(pii_categories.keys()),
                y=list(pii_categories.values()),
                title="PII Categories Distribution"
            )
            st.plotly_chart(fig_categories, use_container_width=True)
    
    # Dataset management actions
    st.markdown("#### Dataset Management")
    
    if auth.has_permission('write'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Dataset"):
                export_dataset()
        
        with col2:
            if st.button("Import Annotations"):
                show_import_interface()
        
        with col3:
            if st.button("Create Training Split"):
                create_training_split()
    
    # Recent activity
    st.markdown("#### Recent Activity")
    show_recent_activity()

def get_dataset_statistics() -> Dict[str, Any]:
    """Get comprehensive dataset statistics"""
    # Mock statistics (would integrate with actual data storage)
    return {
        'total_documents': np.random.randint(150, 500),
        'annotated_documents': np.random.randint(80, 200),
        'total_entities': np.random.randint(800, 2500),
        'document_types': {
            'PDF': np.random.randint(50, 150),
            'DOCX': np.random.randint(30, 100),
            'Images': np.random.randint(20, 80)
        },
        'pii_categories': {
            'PERSON': np.random.randint(150, 400),
            'EMAIL': np.random.randint(80, 200),
            'PHONE': np.random.randint(60, 150),
            'ADDRESS': np.random.randint(40, 120),
            'SSN': np.random.randint(20, 60),
            'DATE': np.random.randint(100, 250)
        },
        'annotation_quality': {
            'high_quality': np.random.randint(60, 90),
            'medium_quality': np.random.randint(20, 40),
            'low_quality': np.random.randint(5, 20)
        }
    }

def export_dataset():
    """Export dataset for training"""
    st.success("Dataset export initiated. You will receive a download link when ready.")
    
    # Mock export process
    export_options = {
        'format': 'COCO JSON',
        'include_images': True,
        'include_annotations': True,
        'train_test_split': '80/20'
    }
    
    with st.expander("Export Details"):
        st.json(export_options)

def show_import_interface():
    """Show annotation import interface"""
    with st.expander("Import Annotations", expanded=True):
        st.markdown("#### Import External Annotations")
        
        uploaded_file = st.file_uploader(
            "Choose annotation file",
            type=['json', 'csv', 'xml'],
            help="Supported formats: COCO JSON, CSV, XML"
        )
        
        if uploaded_file:
            import_format = st.selectbox(
                "Annotation Format",
                ['COCO JSON', 'CSV (Custom)', 'XML (Custom)', 'Auto-detect']
            )
            
            merge_strategy = st.selectbox(
                "Merge Strategy",
                ['Replace existing', 'Merge with existing', 'Create new version']
            )
            
            if st.button("Import Annotations"):
                st.success(f"Annotations imported successfully using {import_format} format!")

def create_training_split():
    """Create training/validation split"""
    st.success("Training split created: 70% train, 15% validation, 15% test")
    
    split_stats = {
        'train': {'documents': 105, 'entities': 1200},
        'validation': {'documents': 22, 'entities': 280},
        'test': {'documents': 23, 'entities': 290}
    }
    
    with st.expander("Split Details"):
        for split_name, stats in split_stats.items():
            st.write(f"**{split_name.title()}:** {stats['documents']} documents, {stats['entities']} entities")

def show_recent_activity():
    """Show recent dataset activity"""
    # Mock recent activity
    activities = [
        {
            'timestamp': '2024-01-15 14:30',
            'user': 'analyst_user',
            'action': 'Added annotations',
            'details': '15 new PII entities annotated in document_batch_03'
        },
        {
            'timestamp': '2024-01-15 13:45',
            'user': 'admin_user',
            'action': 'Created dataset split',
            'details': 'Training split created: 150 documents'
        },
        {
            'timestamp': '2024-01-15 11:20',
            'user': 'data_curator',
            'action': 'Quality review',
            'details': 'Reviewed and approved 25 annotations'
        }
    ]
    
    for activity in activities:
        col1, col2, col3 = st.columns([2, 1, 3])
        
        with col1:
            st.write(activity['timestamp'])
        
        with col2:
            st.write(activity['user'])
        
        with col3:
            st.write(f"**{activity['action']}:** {activity['details']}")

def show_annotation_management():
    """Show annotation management interface"""
    st.markdown("### Annotation Management")
    
    if not auth.has_permission('write'):
        st.warning("Annotation management requires write permissions.")
        return
    
    # Annotation statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Pending Annotations", np.random.randint(15, 45))
    
    with col2:
        st.metric("Completed Today", np.random.randint(8, 25))
    
    with col3:
        st.metric("Quality Score", f"{np.random.uniform(0.85, 0.95):.1%}")
    
    # Annotation queue
    st.markdown("#### Annotation Queue")
    
    # Mock annotation queue
    queue_data = []
    for i in range(10):
        queue_data.append({
            'Document ID': f'doc_{i+1:03d}',
            'Type': np.random.choice(['PDF', 'DOCX', 'Image']),
            'Priority': np.random.choice(['High', 'Medium', 'Low']),
            'Estimated Time': f"{np.random.randint(5, 30)} min",
            'Assigned To': np.random.choice(['unassigned', 'annotator_1', 'annotator_2']),
            'Status': np.random.choice(['pending', 'in_progress', 'review'])
        })
    
    df_queue = pd.DataFrame(queue_data)
    
    # Filter options
    col1, col2 = st.columns(2)
    
    with col1:
        status_filter = st.multiselect(
            "Filter by Status",
            ['pending', 'in_progress', 'review'],
            default=['pending', 'in_progress']
        )
    
    with col2:
        priority_filter = st.multiselect(
            "Filter by Priority", 
            ['High', 'Medium', 'Low'],
            default=['High', 'Medium', 'Low']
        )
    
    # Apply filters
    filtered_df = df_queue[
        (df_queue['Status'].isin(status_filter)) &
        (df_queue['Priority'].isin(priority_filter))
    ]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Annotation actions
    st.markdown("#### Annotation Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Annotation Session"):
            start_annotation_session()
    
    with col2:
        if st.button("Review Completed"):
            show_review_interface()
    
    with col3:
        if st.button("Export Annotations"):
            export_annotations()

def start_annotation_session():
    """Start new annotation session"""
    st.success("Annotation session started!")
    
    # Mock annotation interface
    with st.container():
        st.markdown("#### Annotation Interface")
        
        # Document selection
        selected_doc = st.selectbox(
            "Select document to annotate:",
            ['doc_001.pdf', 'doc_002.docx', 'doc_003.jpg']
        )
        
        # Mock document text
        mock_text = """
        John Smith works at Acme Corporation. 
        His email is john.smith@acme.com and phone number is 555-123-4567.
        The company address is 123 Main Street, New York, NY 10001.
        """
        
        st.text_area("Document Text", mock_text, height=150)
        
        # Annotation interface
        ui_components.show_annotation_interface(mock_text)

def show_review_interface():
    """Show annotation review interface"""
    with st.expander("Annotation Review", expanded=True):
        st.markdown("#### Review Completed Annotations")
        
        # Mock annotations to review
        review_items = [
            {
                'Document': 'doc_045.pdf',
                'Entity': 'Jane Doe',
                'Type': 'PERSON',
                'Confidence': 0.92,
                'Annotator': 'annotator_1',
                'Status': 'pending_review'
            },
            {
                'Document': 'doc_046.docx', 
                'Entity': 'jane.doe@company.com',
                'Type': 'EMAIL',
                'Confidence': 0.98,
                'Annotator': 'annotator_2',
                'Status': 'pending_review'
            }
        ]
        
        for item in review_items:
            col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
            
            with col1:
                st.write(f"**{item['Document']}**")
                st.write(f"{item['Type']}: {item['Entity']}")
            
            with col2:
                st.write(f"Annotator: {item['Annotator']}")
                st.write(f"Confidence: {item['Confidence']:.1%}")
            
            with col3:
                if st.button("Approve", key=f"approve_{item['Entity']}"):
                    st.success("Approved!")
            
            with col4:
                if st.button("Reject", key=f"reject_{item['Entity']}"):
                    st.error("Rejected!")

def export_annotations():
    """Export annotations"""
    st.success("Annotations exported successfully!")
    
    export_stats = {
        'total_annotations': 1247,
        'entities_by_type': {
            'PERSON': 245,
            'EMAIL': 189,
            'PHONE': 156,
            'ADDRESS': 98,
            'SSN': 67,
            'DATE': 234,
            'OTHER': 258
        },
        'export_format': 'COCO JSON',
        'file_size': '2.4 MB'
    }
    
    with st.expander("Export Summary"):
        st.json(export_stats)

def show_active_learning():
    """Show active learning interface"""
    st.markdown("### Active Learning")
    
    if not auth.has_permission('write'):
        st.warning("Active learning requires write permissions.")
        return
    
    # Active learning strategies
    st.markdown("#### Active Learning Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy = st.selectbox(
            "Select Active Learning Strategy",
            [
                'Uncertainty Sampling',
                'Diversity Sampling', 
                'Query by Committee',
                'Expected Model Change'
            ]
        )
        
        batch_size = st.slider("Batch Size", 5, 50, 20)
        
    with col2:
        st.markdown("**Strategy Description:**")
        
        descriptions = {
            'Uncertainty Sampling': 'Select documents where model confidence is lowest',
            'Diversity Sampling': 'Select diverse documents to improve generalization',
            'Query by Committee': 'Select documents where multiple models disagree',
            'Expected Model Change': 'Select documents that would most change the model'
        }
        
        st.info(descriptions.get(strategy, 'No description available'))
    
    if st.button("Generate Active Learning Batch"):
        generate_active_learning_batch(strategy, batch_size)
    
    # Active learning results
    st.markdown("#### Active Learning Queue")
    
    # Mock active learning suggestions
    al_suggestions = []
    for i in range(batch_size if 'batch_size' in locals() else 10):
        al_suggestions.append({
            'Document ID': f'al_doc_{i+1:03d}',
            'Uncertainty Score': np.random.uniform(0.3, 0.9),
            'Predicted Difficulty': np.random.choice(['High', 'Medium', 'Low']),
            'Estimated Value': np.random.uniform(0.6, 0.95),
            'Priority': np.random.randint(1, 10)
        })
    
    df_al = pd.DataFrame(al_suggestions)
    df_al = df_al.sort_values('Uncertainty Score', ascending=False)
    
    st.dataframe(df_al, use_container_width=True)
    
    # Annotation efficiency metrics
    st.markdown("#### Annotation Efficiency Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg. Annotation Time", "12.5 min")
    
    with col2:
        st.metric("Model Improvement", "+3.2% F1")
    
    with col3:
        st.metric("Annotation Budget Used", "65%")

def generate_active_learning_batch(strategy: str, batch_size: int):
    """Generate active learning batch"""
    st.success(f"Generated {batch_size} documents using {strategy} strategy!")
    
    # Mock efficiency metrics
    efficiency_gain = np.random.uniform(0.15, 0.35)
    st.info(f"Expected efficiency gain: {efficiency_gain:.1%} compared to random sampling")

def show_data_quality():
    """Show data quality assessment"""
    st.markdown("### Data Quality Assessment")
    
    # Quality overview
    quality_stats = {
        'overall_score': np.random.uniform(0.82, 0.94),
        'completeness': np.random.uniform(0.88, 0.96),
        'consistency': np.random.uniform(0.79, 0.89),
        'accuracy': np.random.uniform(0.85, 0.93)
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = quality_stats['overall_score']
        color = "green" if score > 0.9 else "orange" if score > 0.8 else "red"
        st.metric("Overall Quality", f"{score:.1%}")
    
    with col2:
        st.metric("Completeness", f"{quality_stats['completeness']:.1%}")
    
    with col3:
        st.metric("Consistency", f"{quality_stats['consistency']:.1%}")
    
    with col4:
        st.metric("Accuracy", f"{quality_stats['accuracy']:.1%}")
    
    # Quality analysis
    st.markdown("#### Quality Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Quality Metrics", "🔍 Issue Detection", "📋 Recommendations"])
    
    with tab1:
        show_quality_metrics()
    
    with tab2:
        show_issue_detection()
    
    with tab3:
        show_quality_recommendations()

def show_quality_metrics():
    """Show detailed quality metrics"""
    # Mock quality metrics by category
    quality_by_category = {
        'PERSON': {'precision': 0.92, 'recall': 0.88, 'consistency': 0.85},
        'EMAIL': {'precision': 0.98, 'recall': 0.95, 'consistency': 0.92},
        'PHONE': {'precision': 0.89, 'recall': 0.83, 'consistency': 0.79},
        'ADDRESS': {'precision': 0.84, 'recall': 0.76, 'consistency': 0.74}
    }
    
    df_quality = pd.DataFrame(quality_by_category).T
    df_quality = df_quality.round(3)
    
    st.dataframe(df_quality, use_container_width=True)
    
    # Quality trends
    fig_quality = px.line(
        x=list(quality_by_category.keys()),
        y=[metrics['precision'] for metrics in quality_by_category.values()],
        title='Precision by PII Category',
        markers=True
    )
    st.plotly_chart(fig_quality, use_container_width=True)

def show_issue_detection():
    """Show detected quality issues"""
    # Mock quality issues
    issues = [
        {
            'severity': 'High',
            'category': 'PERSON',
            'issue': 'Low inter-annotator agreement (0.72)',
            'affected_docs': 23,
            'recommendation': 'Review annotation guidelines for person names'
        },
        {
            'severity': 'Medium',
            'category': 'ADDRESS',
            'issue': 'Inconsistent address format annotation',
            'affected_docs': 15,
            'recommendation': 'Standardize address annotation format'
        },
        {
            'severity': 'Low',
            'category': 'EMAIL',
            'issue': 'Missing domain validation in some cases',
            'affected_docs': 8,
            'recommendation': 'Add email format validation rules'
        }
    ]
    
    for issue in issues:
        severity_color = {
            'High': 'red',
            'Medium': 'orange',
            'Low': 'blue'
        }.get(issue['severity'], 'gray')
        
        st.markdown(f"""
        <div style="border-left: 4px solid {severity_color}; padding: 10px; margin: 10px 0; background-color: #f8f9fa;">
            <strong>{issue['severity']} - {issue['category']}</strong><br>
            {issue['issue']}<br>
            <em>Affected documents:</em> {issue['affected_docs']}<br>
            <em>Recommendation:</em> {issue['recommendation']}
        </div>
        """, unsafe_allow_html=True)

def show_quality_recommendations():
    """Show quality improvement recommendations"""
    recommendations = [
        "Increase inter-annotator agreement by providing more specific guidelines",
        "Implement regular quality review sessions with annotators", 
        "Add automated validation rules for common PII formats",
        "Create annotation templates for consistent formatting",
        "Implement confidence scoring for annotations",
        "Regular retraining sessions for annotation team"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"{i}. {rec}")
    
    if st.button("Generate Quality Report"):
        st.success("Quality report generated and sent to administrators.")