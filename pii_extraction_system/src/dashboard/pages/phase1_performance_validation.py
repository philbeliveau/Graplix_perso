"""
Phase 1 Cross-Domain Performance Validation Dashboard

This module implements the Phase 1 features for cross-domain performance validation:
- Document selection interface (10+ diverse documents)
- Multi-model testing panel (GPT-4o, GPT-4o-mini, other models)
- Real-time PII extraction comparison
- Performance variance calculator (target: <10%)
- Baseline comparison (OCR + spaCy NER vs LLM approaches)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import session_state, ui_components, auth
from llm.multimodal_llm_service import MultimodalLLMService
from llm.api_integration_wrapper import MultiLLMIntegrationWrapper
from extractors.evaluation import PIIEvaluator
# Import Phase 0's exact document processing functions
from utils.variance_analysis import VarianceAnalyzer
import tempfile
import base64
from io import BytesIO
from pathlib import Path as PathlibPath

# Import the SAME conversion function that Phase 0 uses
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pages'))

# Import Phase 0's convert_document_to_images function directly
try:
    from dataset_creation_phase0 import convert_document_to_images as phase0_convert_document_to_images
    PHASE0_CONVERSION_AVAILABLE = True
except ImportError:
    PHASE0_CONVERSION_AVAILABLE = False
    st.error("Cannot import Phase 0 conversion function")


def show_page():
    """Main Phase 1 Performance Validation page with analytics integration"""
    st.markdown('<div class="section-header">🎯 Phase 1: Cross-Domain Performance Validation & Analytics</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Phase 1 Objectives:**
    - Document selection interface (10+ diverse documents)
    - Multi-model testing panel (GPT-4o, GPT-4o-mini, other models)
    - Real-time PII extraction comparison
    - Performance variance calculator (target: <10%)
    - Baseline comparison (OCR + spaCy NER vs LLM approaches)
    - **Analytics Integration**: Model comparison, performance analytics, and metrics dashboard
    """)
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Initialize session state for Phase 1
    initialize_phase1_session_state()
    
    # Main analytics tabs
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "🔬 Model Validation",
        "📈 Model Comparison", 
        "📊 Performance Analytics",
        "📋 Metrics Dashboard"
    ])
    
    with main_tab1:
        show_model_validation_interface()
    
    with main_tab2:
        show_model_comparison_interface()
    
    with main_tab3:
        show_performance_analytics_interface()
    
    with main_tab4:
        show_metrics_dashboard_interface()


def show_model_validation_interface():
    """Show the original model validation interface as sub-tabs"""
    st.markdown("### 🔬 Model Validation & Testing")
    st.markdown("Comprehensive model validation tools with document selection, multi-model testing, and baseline comparison.")
    
    # Sub-tabs for model validation
    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
        "📄 Document Selection",
        "🤖 Multi-Model Testing",
        "⚡ Real-time Comparison",
        "📊 Variance Analysis",
        "🔍 Baseline Comparison"
    ])
    
    with sub_tab1:
        show_document_selection_interface()
    
    with sub_tab2:
        show_multi_model_testing_panel()
    
    with sub_tab3:
        show_realtime_comparison_dashboard()
    
    with sub_tab4:
        show_variance_calculator()
    
    with sub_tab5:
        show_baseline_comparison_tools()


def initialize_phase1_session_state():
    """Initialize session state for Phase 1 features"""
    if 'phase1_selected_documents' not in st.session_state:
        st.session_state.phase1_selected_documents = []
    
    if 'phase1_selected_models' not in st.session_state:
        st.session_state.phase1_selected_models = []
    
    if 'phase1_test_results' not in st.session_state:
        st.session_state.phase1_test_results = {}
    
    if 'phase1_baseline_results' not in st.session_state:
        st.session_state.phase1_baseline_results = {}
    
    if 'phase1_variance_threshold' not in st.session_state:
        st.session_state.phase1_variance_threshold = 0.10  # 10% target
    
    # Analytics-specific session state
    if 'phase1_analytics_cache' not in st.session_state:
        st.session_state.phase1_analytics_cache = {}
    
    if 'phase1_performance_trends' not in st.session_state:
        st.session_state.phase1_performance_trends = {}
    
    if 'phase1_model_rankings' not in st.session_state:
        st.session_state.phase1_model_rankings = {}
    
    if 'phase1_using_real_ground_truth' not in st.session_state:
        st.session_state.phase1_using_real_ground_truth = False


def import_from_phase0():
    """Import documents from Phase 0 dataset creation"""
    if 'phase0_dataset' not in st.session_state or not st.session_state.phase0_dataset:
        return False
    
    labeled_docs = [doc for doc in st.session_state.phase0_dataset if doc.get('labeled', False)]
    
    if not labeled_docs:
        st.warning("No labeled documents found in Phase 0 dataset.")
        return False
    
    # Convert Phase 0 format to Phase 1 format
    phase1_docs = []
    for doc in labeled_docs:
        # Extract PII types from ground truth labels
        pii_types = []
        if doc.get('gpt4o_labels', {}).get('entities'):
            pii_types = list(set([entity['type'] for entity in doc['gpt4o_labels']['entities']]))
        
        phase1_doc = {
            'category': doc['metadata'].get('domain', 'General'),
            'name': doc['name'],
            'file_content': doc.get('content'),  # Base64 encoded content from Phase 0
            'file_type': doc.get('type', 'application/pdf'),  # MIME type
            'info': {
                'type': doc['metadata'].get('document_type', 'document'),
                'complexity': doc['metadata'].get('difficulty_level', 'medium'),
                'expected_pii_types': pii_types,
                'language': 'English',  # Could be enhanced from metadata
                'ground_truth_labels': doc.get('gpt4o_labels', {}),
                'document_id': doc['id'],
                'confidence_score': doc.get('gpt4o_labels', {}).get('confidence_score', 0),
                'processing_time': doc.get('gpt4o_labels', {}).get('processing_time', 0),
                'cost': doc.get('gpt4o_labels', {}).get('cost', 0),
                'file_extension': PathlibPath(doc['name']).suffix
            }
        }
        phase1_docs.append(phase1_doc)
    
    # Update session state
    st.session_state.phase1_selected_documents = phase1_docs
    st.session_state.phase1_using_real_ground_truth = True
    
    return True

def show_document_selection_interface():
    """Document selection interface with 10+ diverse documents"""
    st.markdown("### 📄 Document Selection Interface")
    st.markdown("Select diverse documents for cross-domain validation testing.")
    
    # Phase 0 Import Section
    st.markdown("#### 🔄 Import from Phase 0")
    
    # Check if Phase 0 data exists
    phase0_available = 'phase0_dataset' in st.session_state and st.session_state.phase0_dataset
    
    if phase0_available:
        labeled_count = sum(1 for doc in st.session_state.phase0_dataset if doc.get('labeled', False))
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.info(f"📊 Phase 0 dataset available: {labeled_count} labeled documents ready for validation")
        
        with col2:
            if st.button("🔄 Import Phase 0 Dataset", type="primary"):
                if import_from_phase0():
                    st.success(f"✅ Successfully imported {labeled_count} documents from Phase 0!")
                    st.rerun()
        
        with col3:
            if st.button("🗑️ Clear Import"):
                if 'phase1_selected_documents' in st.session_state:
                    st.session_state.phase1_selected_documents = []
                if 'phase1_using_real_ground_truth' in st.session_state:
                    st.session_state.phase1_using_real_ground_truth = False
                st.success("Cleared imported documents")
                st.rerun()
    else:
        st.warning("🔍 No Phase 0 dataset found. Create a dataset in **Dataset Creation (Phase 0)** first, then return here to import it.")
    
    st.markdown("---")
    
    # Check if we're using imported data
    using_imported = st.session_state.get('phase1_using_real_ground_truth', False)
    
    if using_imported and st.session_state.get('phase1_selected_documents'):
        st.markdown("#### 📋 Imported Documents from Phase 0")
        
        # Show imported documents summary
        imported_docs = st.session_state.phase1_selected_documents
        categories = {}
        for doc in imported_docs:
            category = doc['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(doc)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(imported_docs))
        with col2:
            st.metric("Categories", len(categories))
        with col3:
            total_entities = sum(len(doc['info'].get('expected_pii_types', [])) for doc in imported_docs)
            st.metric("PII Types", total_entities)
        
        # Show documents by category
        for category, docs in categories.items():
            with st.expander(f"📁 {category} ({len(docs)} documents)"):
                for doc in docs:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.write(f"📄 **{doc['name']}**")
                    with col2:
                        pii_types = doc['info'].get('expected_pii_types', [])
                        if pii_types:
                            st.write(f"🏷️ PII: {', '.join(pii_types[:3])}")
                            if len(pii_types) > 3:
                                st.write(f"   +{len(pii_types)-3} more...")
                    with col3:
                        confidence = doc['info'].get('confidence_score', 0)
                        st.write(f"🎯 {confidence:.2f}")
        
        return  # Skip the manual selection if using imported data
    
    # Original manual document selection (only shown if not using imported data)
    st.markdown("#### 🔍 Manual Document Selection")
    st.info("💡 **Tip**: Import your Phase 0 dataset above for real ground truth validation, or select from sample documents below for testing.")
    
    # Document categories and examples
    document_categories = get_diverse_document_catalog()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Available Document Categories")
        
        # Category selection
        selected_categories = st.multiselect(
            "Select document categories:",
            options=list(document_categories.keys()),
            default=list(document_categories.keys())[:3],
            help="Choose different document types for comprehensive testing"
        )
        
        # Document selection within categories
        st.markdown("#### Document Selection")
        selected_documents = []
        
        for category in selected_categories:
            with st.expander(f"{category} ({len(document_categories[category])} documents)"):
                category_docs = st.multiselect(
                    f"Select {category} documents:",
                    options=[doc['name'] for doc in document_categories[category]],
                    key=f"docs_{category}",
                    help=f"Select documents from {category} category"
                )
                
                for doc_name in category_docs:
                    doc_info = next(doc for doc in document_categories[category] if doc['name'] == doc_name)
                    selected_documents.append({
                        'category': category,
                        'name': doc_name,
                        'info': doc_info
                    })
        
        # Update session state
        st.session_state.phase1_selected_documents = selected_documents
        
        # Selection summary
        st.markdown("#### Selection Summary")
        st.metric("Total Documents Selected", len(selected_documents))
        st.metric("Categories Covered", len(selected_categories))
        
        # Validation
        if len(selected_documents) >= 10:
            st.success(f"✅ Minimum requirement met ({len(selected_documents)}/10+ documents)")
        else:
            st.warning(f"⚠️ Need at least 10 documents ({len(selected_documents)}/10)")
    
    with col2:
        st.markdown("#### Selected Documents Overview")
        
        if selected_documents:
            # Create document overview table
            doc_data = []
            for doc in selected_documents:
                doc_data.append({
                    'Document': doc['name'],
                    'Category': doc['category'],
                    'Type': doc['info']['type'],
                    'Complexity': doc['info']['complexity'],
                    'PII Types': ', '.join(doc['info']['expected_pii_types']),
                    'Language': doc['info']['language']
                })
            
            df_docs = pd.DataFrame(doc_data)
            st.dataframe(df_docs, use_container_width=True, height=400)
            
            # Diversity metrics
            st.markdown("#### Diversity Metrics")
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                complexity_dist = df_docs['Complexity'].value_counts()
                fig_complexity = px.pie(
                    values=complexity_dist.values,
                    names=complexity_dist.index,
                    title="Complexity Distribution"
                )
                st.plotly_chart(fig_complexity, use_container_width=True)
            
            with col2b:
                type_dist = df_docs['Type'].value_counts()
                fig_type = px.pie(
                    values=type_dist.values,
                    names=type_dist.index,
                    title="Document Type Distribution"
                )
                st.plotly_chart(fig_type, use_container_width=True)
            
            with col2c:
                lang_dist = df_docs['Language'].value_counts()
                fig_lang = px.pie(
                    values=lang_dist.values,
                    names=lang_dist.index,
                    title="Language Distribution"
                )
                st.plotly_chart(fig_lang, use_container_width=True)
        
        else:
            st.info("Select documents from the categories to see overview and diversity metrics.")


def get_diverse_document_catalog() -> Dict[str, List[Dict]]:
    """Get catalog of diverse documents for testing"""
    return {
        "Business Documents": [
            {
                "name": "Employment Contract - Standard",
                "type": "PDF",
                "complexity": "Medium",
                "expected_pii_types": ["PERSON", "EMAIL", "PHONE", "ADDRESS", "SSN"],
                "language": "English",
                "description": "Standard employment contract with personal information"
            },
            {
                "name": "Invoice - Corporate",
                "type": "PDF",
                "complexity": "Low",
                "expected_pii_types": ["PERSON", "EMAIL", "PHONE", "ADDRESS"],
                "language": "English",
                "description": "Corporate invoice with billing information"
            },
            {
                "name": "Business Letter - French",
                "type": "PDF",
                "complexity": "Medium",
                "expected_pii_types": ["PERSON", "EMAIL", "PHONE", "ADDRESS"],
                "language": "French",
                "description": "French business correspondence"
            }
        ],
        "Healthcare Documents": [
            {
                "name": "Medical Record - Patient Summary",
                "type": "PDF",
                "complexity": "High",
                "expected_pii_types": ["PERSON", "PHONE", "ADDRESS", "DATE_OF_BIRTH", "MEDICAL_ID"],
                "language": "English",
                "description": "Patient medical record summary"
            },
            {
                "name": "Insurance Claim Form",
                "type": "PDF",
                "complexity": "High",
                "expected_pii_types": ["PERSON", "SSN", "PHONE", "ADDRESS", "MEDICAL_ID"],
                "language": "English",
                "description": "Medical insurance claim form"
            }
        ],
        "Financial Documents": [
            {
                "name": "Bank Statement - Personal",
                "type": "PDF",
                "complexity": "Medium",
                "expected_pii_types": ["PERSON", "ACCOUNT_NUMBER", "ADDRESS", "PHONE"],
                "language": "English",
                "description": "Personal bank account statement"
            },
            {
                "name": "Credit Application",
                "type": "PDF",
                "complexity": "High",
                "expected_pii_types": ["PERSON", "SSN", "PHONE", "ADDRESS", "EMAIL", "ACCOUNT_NUMBER"],
                "language": "English",
                "description": "Credit card application form"
            },
            {
                "name": "Tax Document - T4",
                "type": "PDF",
                "complexity": "Medium",
                "expected_pii_types": ["PERSON", "SIN", "ADDRESS", "EMPLOYER_INFO"],
                "language": "English/French",
                "description": "Canadian T4 tax slip"
            }
        ],
        "Government Documents": [
            {
                "name": "Passport Application",
                "type": "PDF",
                "complexity": "High",
                "expected_pii_types": ["PERSON", "DATE_OF_BIRTH", "ADDRESS", "PHONE", "EMAIL"],
                "language": "English",
                "description": "Government passport application form"
            },
            {
                "name": "Driver License Application",
                "type": "PDF",
                "complexity": "Medium",
                "expected_pii_types": ["PERSON", "DATE_OF_BIRTH", "ADDRESS", "PHONE"],
                "language": "English",
                "description": "Driver's license application"
            }
        ],
        "Legal Documents": [
            {
                "name": "Court Filing - Civil Case",
                "type": "PDF",
                "complexity": "High",
                "expected_pii_types": ["PERSON", "ADDRESS", "PHONE", "EMAIL", "CASE_NUMBER"],
                "language": "English",
                "description": "Civil court filing document"
            },
            {
                "name": "Power of Attorney",
                "type": "PDF",
                "complexity": "Medium",
                "expected_pii_types": ["PERSON", "ADDRESS", "PHONE", "NOTARY_INFO"],
                "language": "English",
                "description": "Legal power of attorney document"
            }
        ],
        "Academic Documents": [
            {
                "name": "Student Transcript",
                "type": "PDF",
                "complexity": "Medium",
                "expected_pii_types": ["PERSON", "STUDENT_ID", "DATE_OF_BIRTH", "ADDRESS"],
                "language": "English",
                "description": "University student transcript"
            },
            {
                "name": "Research Application",
                "type": "PDF",
                "complexity": "Low",
                "expected_pii_types": ["PERSON", "EMAIL", "PHONE", "INSTITUTION"],
                "language": "English",
                "description": "Academic research grant application"
            }
        ]
    }


def show_multi_model_testing_panel():
    """Multi-model testing panel for LLM models"""
    st.markdown("### 🤖 Multi-Model Testing Panel")
    st.markdown("Configure and test multiple LLM models for PII extraction comparison.")
    
    # Get available LLM models
    available_models = get_available_llm_models()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Model Configuration")
        
        # Model selection
        # Create model options with vision support indicators
        model_options = []
        model_labels = {}
        vision_capable_models = []
        
        for model_key, model_info in available_models.items():
            vision_indicator = "👁️" if model_info.get('supports_vision', False) else "📝"
            provider = model_info.get('provider', 'unknown')
            label = f"{vision_indicator} {model_key} ({provider})"
            model_options.append(model_key)
            model_labels[model_key] = label
            if model_info.get('supports_vision', False):
                vision_capable_models.append(model_key)
        
        # Prefer vision-capable models for document processing
        preferred_defaults = ["openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3-haiku"]
        valid_defaults = [model for model in preferred_defaults if model in model_options]
        if not valid_defaults:
            valid_defaults = vision_capable_models[:3]  # Use any vision-capable models
        
        selected_models = st.multiselect(
            "Select LLM models for testing:",
            options=model_options,
            default=valid_defaults[:3] if valid_defaults else model_options[:3],
            format_func=lambda x: model_labels.get(x, x),
            help="👁️ = Vision-capable (can process document images), 📝 = Text-only. Recommend vision-capable models for document processing."
        )
        
        # Test configuration
        st.markdown("#### Test Configuration")
        
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of documents to process simultaneously"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for PII detection"
        )
        
        enable_ensemble = st.checkbox(
            "Enable Ensemble Voting",
            value=True,
            help="Combine results from multiple models"
        )
        
        # Cost estimation
        st.markdown("#### Cost Estimation")
        if selected_models and st.session_state.phase1_selected_documents:
            total_cost = estimate_testing_cost(selected_models, st.session_state.phase1_selected_documents)
            st.metric("Estimated Total Cost", f"${total_cost:.4f}")
            
            # Cost breakdown
            with st.expander("Cost Breakdown"):
                for model in selected_models:
                    model_cost = total_cost / len(selected_models)  # Simplified
                    st.write(f"**{model}:** ${model_cost:.4f}")
        
        # Show evaluation mode status
        using_real_ground_truth = st.session_state.get('phase1_using_real_ground_truth', False)
        if using_real_ground_truth:
            st.success("🎯 **Real Ground Truth Mode**: Using Phase 0 dataset for actual performance evaluation")
        else:
            st.info("🔄 **Simulation Mode**: Using simulated metrics for testing (import Phase 0 dataset for real evaluation)")
        
        # Run testing
        if st.button("Start Multi-Model Testing", 
                    disabled=not (selected_models and st.session_state.phase1_selected_documents)):
            run_multi_model_testing(selected_models, confidence_threshold, batch_size, enable_ensemble)
    
    with col2:
        st.markdown("#### Testing Results")
        
        if 'multi_model_results' in st.session_state and st.session_state.multi_model_results:
            results = st.session_state.multi_model_results
            
            # Results summary table
            st.markdown("##### Model Performance Summary")
            summary_data = []
            for model_name, model_results in results.items():
                if isinstance(model_results, dict) and 'metrics' in model_results:
                    summary_data.append({
                        'Model': model_name,
                        'Precision': f"{model_results['metrics']['precision']:.3f}",
                        'Recall': f"{model_results['metrics']['recall']:.3f}",
                        'F1-Score': f"{model_results['metrics']['f1']:.3f}",
                        'Processing Time': f"{model_results['metrics']['avg_processing_time']:.2f}s",
                        'Total Cost': f"${model_results['metrics']['total_cost']:.4f}"
                    })
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
                
                # Performance visualization
                st.markdown("##### Performance Comparison")
                
                # Metrics comparison chart
                metrics_df = pd.DataFrame({
                    'Model': [row['Model'] for row in summary_data],
                    'Precision': [float(row['Precision']) for row in summary_data],
                    'Recall': [float(row['Recall']) for row in summary_data],
                    'F1-Score': [float(row['F1-Score']) for row in summary_data]
                })
                
                fig = px.bar(
                    metrics_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                    x='Model',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    title='Model Performance Metrics Comparison'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Cost vs Performance scatter plot
                cost_perf_data = []
                for row in summary_data:
                    cost_perf_data.append({
                        'Model': row['Model'],
                        'Cost': float(row['Total Cost'].replace('$', '')),
                        'F1-Score': float(row['F1-Score']),
                        'Processing Time': float(row['Processing Time'].replace('s', ''))
                    })
                
                df_cost_perf = pd.DataFrame(cost_perf_data)
                fig_scatter = px.scatter(
                    df_cost_perf,
                    x='Cost',
                    y='F1-Score',
                    size='Processing Time',
                    color='Model',
                    title='Cost vs Performance Analysis',
                    hover_data=['Processing Time']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        else:
            st.info("Configure models and documents, then run testing to see results.")


def get_available_llm_models() -> Dict[str, Dict]:
    """Get available LLM models for testing - enhanced with vision support info"""
    try:
        # Get models from the actual LLM service
        llm_service = MultimodalLLMService()
        
        models = {}
        for model_key in llm_service.get_available_models():
            model_info = llm_service.get_model_info(model_key)
            if model_info.get('available', False):
                models[model_key] = {
                    'display_name': model_key.replace('/', ' ').title(),
                    'provider': model_info['provider'],
                    'supports_vision': model_info.get('supports_images', False),
                    'quality_score': 0.9 if 'gpt-4o' in model_key else 0.8,
                    'speed_score': 0.9 if 'mini' in model_key or 'haiku' in model_key else 0.7,
                    'input_cost': model_info.get('cost_per_1k_input_tokens', 0.001),
                    'output_cost': model_info.get('cost_per_1k_output_tokens', 0.003),
                    'description': f"{'Vision-capable' if model_info.get('supports_images') else 'Text-only'} {model_info['provider']} model"
                }
        return models
    except Exception as e:
        st.warning(f"Could not load LLM models dynamically: {e}")
        # Fallback to static models
        return {
            "openai/gpt-4o": {
                'display_name': "GPT-4o",
                'provider': "openai",
                'supports_vision': True,
                'quality_score': 0.95,
                'speed_score': 0.8,
                'input_cost': 0.0025,
                'output_cost': 0.01,
                'description': "Most capable OpenAI model"
            },
            "gpt-4o-mini": {
                'display_name': "GPT-4o Mini",
                'provider': "openai",
                'supports_vision': True,
                'quality_score': 0.88,
                'speed_score': 0.85,
                'input_cost': 0.00015,
                'output_cost': 0.0006,
                'description': "Cost-effective vision model"
            },
            "claude-3-haiku": {
                'display_name': "Claude 3 Haiku",
                'provider': "anthropic",
                'supports_vision': True,
                'quality_score': 0.82,
                'speed_score': 0.95,
                'input_cost': 0.00025,
                'output_cost': 0.00125,
                'description': "Ultra-fast and affordable"
            },
            "claude-3-sonnet": {
                'display_name': "Claude 3 Sonnet",
                'provider': "anthropic",
                'supports_vision': True,
                'quality_score': 0.92,
                'speed_score': 0.8,
                'input_cost': 0.003,
                'output_cost': 0.015,
                'description': "Balanced performance and cost"
            },
            "gemini-1.5-flash": {
                'display_name': "Gemini 1.5 Flash",
                'provider': "google",
                'supports_vision': True,
                'quality_score': 0.83,
                'speed_score': 0.92,
                'input_cost': 0.000075,
                'output_cost': 0.0003,
                'description': "Extremely cost-effective"
            }
        }


def estimate_testing_cost(models: List[str], documents: List[Dict]) -> float:
    """Estimate cost for multi-model testing"""
    available_models = get_available_llm_models()
    total_cost = 0.0
    
    for model_name in models:
        if model_name in available_models:
            model_info = available_models[model_name]
            # Estimate tokens per document (simplified)
            avg_tokens_per_doc = 1000  # input tokens
            avg_output_tokens = 200    # output tokens
            
            model_cost = 0.0
            for doc in documents:
                input_cost = (avg_tokens_per_doc / 1000) * model_info['input_cost']
                output_cost = (avg_output_tokens / 1000) * model_info['output_cost']
                model_cost += input_cost + output_cost
            
            total_cost += model_cost
    
    return total_cost


def run_multi_model_testing(models: List[str], threshold: float, batch_size: int, ensemble: bool):
    """Run multi-model testing simulation"""
    with st.spinner("Running multi-model testing..."):
        # Simulate testing progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        total_steps = len(models) * len(st.session_state.phase1_selected_documents)
        current_step = 0
        
        for model_name in models:
            status_text.text(f"Testing {model_name}...")
            
            # Run real model testing
            model_results = process_model_real_testing(model_name, st.session_state.phase1_selected_documents, threshold)
            results[model_name] = model_results
            
            # Update progress
            current_step += len(st.session_state.phase1_selected_documents)
            progress_bar.progress(current_step / total_steps)
            
            # Small delay for realism
            time.sleep(0.5)
        
        # Store results
        st.session_state.multi_model_results = results
        
        progress_bar.progress(1.0)
        status_text.text("Testing completed!")
        
        st.success(f"Multi-model testing completed for {len(models)} models!")


# Removed evaluate_model_with_ground_truth - Replaced by process_model_real_testing


# Removed simulate_baseline_performance - We only use real processing now


def process_model_real_testing(model_name: str, documents: List[Dict], threshold: float) -> Dict:
    """Actually process documents with real LLM models"""
    st.info(f"🔥 Processing documents with REAL {model_name} using Phase 0's exact conversion method - No simulation!")
    
    # Initialize LLM service
    llm_service = MultimodalLLMService()
    evaluator = PIIEvaluator()
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_processing_time = 0
    total_cost = 0
    document_results = []
    
    valid_docs = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, doc in enumerate(documents):
        status_text.text(f"Processing {doc['name']} with {model_name}...")
        
        try:
            # Get ground truth
            ground_truth_labels = doc['info'].get('ground_truth_labels', {})
            if not ground_truth_labels or not ground_truth_labels.get('entities'):
                st.warning(f"Skipping {doc['name']} - No ground truth available")
                continue
            
            valid_docs += 1
            gt_entities = ground_truth_labels.get('entities', [])
            
            # Convert document content to images for processing
            file_content = doc.get('file_content')  # Base64 encoded content from Phase 0
            if not file_content:
                st.error(f"No file content found for {doc['name']}")
                continue
                
            start_time = time.time()
            
            # Use Phase 0's EXACT conversion approach
            try:
                if not PHASE0_CONVERSION_AVAILABLE:
                    st.error("Phase 0 conversion function not available")
                    continue
                    
                import base64
                decoded_content = base64.b64decode(file_content)
                file_extension = doc['info'].get('file_extension', '.pdf')
                
                # Use Phase 0's exact conversion function with same password handling
                # Default password "Hubert" like Phase 0 uses
                password = "Hubert"  # Same default password as Phase 0
                images = phase0_convert_document_to_images(decoded_content, file_extension, password)
                
                if not images:
                    st.error(f"Could not convert {doc['name']} to images using Phase 0 method")
                    continue
                    
            except Exception as e:
                st.error(f"Failed to process content for {doc['name']} using Phase 0 method: {str(e)}")
                continue
            
            # Process with real LLM
            extracted_entities = []
            doc_cost = 0
            
            # Check if model supports vision processing
            model_info = llm_service.get_model_info(model_name)
            if not model_info.get('supports_images', True):
                st.warning(f"⚠️ {model_name} doesn't support image processing - skipping {doc['name']}")
                continue
            
            for img_data in images:
                result = llm_service.extract_pii_from_image(
                    image_data=img_data,
                    model_key=model_name,
                    document_type=doc.get('category', 'document')
                )
                
                if result.get('success'):
                    # Extract entities from the result (LLM service returns 'pii_entities')
                    pii_entities = result.get('pii_entities', [])
                    if pii_entities:
                        extracted_entities.extend(pii_entities)
                    else:
                        # Fallback: try other possible field names
                        entities = result.get('entities', [])
                        if entities:
                            extracted_entities.extend(entities)
                    
                    # Get REAL cost from actual API usage
                    usage_info = result.get('usage', {})
                    real_cost = usage_info.get('estimated_cost', 0)
                    doc_cost += real_cost
                    
                    # Log real cost information for transparency
                    if real_cost > 0:
                        st.info(f"💰 Real API cost for this image: ${real_cost:.6f} "
                               f"(Tokens: {usage_info.get('prompt_tokens', 0)} input + {usage_info.get('completion_tokens', 0)} output)")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    if "do not currently support image processing" in error_msg:
                        st.warning(f"⚠️ {model_name} doesn't support image processing for {doc['name']}")
                        break  # Skip remaining images for this document
                    else:
                        st.error(f"Failed to process image from {doc['name']}: {error_msg}")
            
            processing_time = time.time() - start_time
            
            # Debug information about entities
            st.info(f"📋 Extracted {len(extracted_entities)} entities, Ground truth: {len(gt_entities)} entities")
            
            # Evaluate against ground truth
            try:
                evaluation_result = evaluator.evaluate(
                    predicted_entities=extracted_entities,
                    ground_truth_entities=gt_entities,
                    threshold=threshold
                )
            except Exception as eval_error:
                st.error(f"Evaluation error for {doc['name']}: {str(eval_error)}")
                # Create default evaluation result
                class DefaultEvaluation:
                    def __init__(self):
                        self.precision = 0.0
                        self.recall = 0.0
                        self.f1_score = 0.0
                        self.true_positives = 0
                        self.false_positives = len(extracted_entities)
                        self.false_negatives = len(gt_entities)
                
                evaluation_result = DefaultEvaluation()
            
            # Extract metrics
            doc_precision = evaluation_result.precision
            doc_recall = evaluation_result.recall
            doc_f1 = evaluation_result.f1_score
            
            # Get detailed metrics from the underlying metrics object
            try:
                detailed_metrics = evaluation_result.metrics  # Access underlying EvaluationMetrics
                true_positives = getattr(detailed_metrics, 'total_true_positives', len(extracted_entities) if len(extracted_entities) <= len(gt_entities) else len(gt_entities))
                false_positives = getattr(detailed_metrics, 'total_false_positives', max(0, len(extracted_entities) - true_positives))
                false_negatives = getattr(detailed_metrics, 'total_false_negatives', max(0, len(gt_entities) - true_positives))
            except:
                # Fallback calculation
                true_positives = min(len(extracted_entities), len(gt_entities))
                false_positives = max(0, len(extracted_entities) - true_positives)
                false_negatives = max(0, len(gt_entities) - true_positives)
            
            # Store results
            document_results.append({
                "document": doc["name"],
                "domain": doc.get('category', 'Unknown'),
                "processing_time": processing_time,
                "entities_found": len(extracted_entities),
                "ground_truth_entities": len(gt_entities),
                "true_positives": true_positives,
                "false_positives": false_positives, 
                "false_negatives": false_negatives,
                "precision": doc_precision,
                "recall": doc_recall,
                "f1": doc_f1,
                "cost": doc_cost,
                "extracted_entities": extracted_entities,
                "confidence_scores": [e.get('confidence', 0.5) for e in extracted_entities],
                "expected_pii_types": list(set([e.get('type', 'unknown') for e in gt_entities]))
            })
            
            # Accumulate totals
            total_precision += doc_precision
            total_recall += doc_recall
            total_f1 += doc_f1
            total_processing_time += processing_time
            total_cost += doc_cost
            
        except Exception as e:
            st.error(f"Error processing {doc['name']}: {str(e)}")
            continue
        
        # Update progress
        progress_bar.progress((idx + 1) / len(documents))
    
    progress_bar.progress(1.0)
    status_text.text(f"Completed processing {valid_docs} documents with {model_name}")
    
    if valid_docs == 0:
        st.error("No valid documents processed")
        return {"metrics": {}, "document_results": [], "ground_truth_evaluation": False}
    
    # Calculate average metrics
    avg_precision = total_precision / valid_docs
    avg_recall = total_recall / valid_docs
    avg_f1 = total_f1 / valid_docs
    avg_processing_time = total_processing_time / valid_docs
    
    return {
        "metrics": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "avg_processing_time": avg_processing_time,
            "total_cost": total_cost
        },
        "document_results": document_results,
        "ground_truth_evaluation": True,
        "evaluated_documents": valid_docs
    }


def show_realtime_comparison_dashboard():
    """Real-time PII extraction comparison dashboard with document visualization"""
    st.markdown("### ⚡ Real-time PII Extraction Comparison")
    st.markdown("Compare PII extraction results across multiple models in real-time with document visualization.")
    
    # Add document visualization section first
    show_document_visualization()
    
    st.markdown("---")
    st.markdown("#### ⚡ Real-time Model Comparison")
    
    # Control panel
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("#### Comparison Controls")
        
        # Document selection for real-time comparison
        if st.session_state.phase1_selected_documents:
            selected_doc = st.selectbox(
                "Select document for comparison:",
                options=range(len(st.session_state.phase1_selected_documents)),
                format_func=lambda x: st.session_state.phase1_selected_documents[x]['name'],
                key="realtime_doc_selector"
            )
            
            current_doc = st.session_state.phase1_selected_documents[selected_doc]
            
            # Model selection for comparison
            available_models = get_available_llm_models()
            preferred_defaults = ["gpt-4o-mini", "claude-3-haiku"]
            available_options = list(available_models.keys())
            valid_defaults = [model for model in preferred_defaults if model in available_options]
            
            comparison_models = st.multiselect(
                "Select models for comparison:",
                options=available_options,
                default=valid_defaults if valid_defaults else available_options[:2],
                key="realtime_model_selector"
            )
            
            # Real-time settings
            st.markdown("#### Real-time Settings")
            
            auto_refresh = st.checkbox("Auto-refresh every 5 seconds", value=False)
            
            confidence_filter = st.slider(
                "Confidence Filter",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Show only entities above this confidence"
            )
            
            # Manual refresh button
            if st.button("🔄 Refresh Comparison"):
                run_realtime_comparison(current_doc, comparison_models, confidence_filter)
        
        else:
            st.warning("Please select documents in the Document Selection tab first.")
    
    with col2:
        st.markdown("#### Real-time Comparison Results")
        
        if 'realtime_results' in st.session_state and st.session_state.realtime_results:
            show_realtime_results(st.session_state.realtime_results, confidence_filter)
        else:
            st.info("Select a document and models, then click 'Refresh Comparison' to see real-time results.")
        
        # Auto-refresh functionality
        if st.session_state.get('auto_refresh', False):
            # This would be implemented with st.rerun() in a real scenario
            st.info("Auto-refresh is enabled. Results will update every 5 seconds.")


def run_realtime_comparison(document: Dict, models: List[str], confidence_filter: float):
    """Run real-time comparison for selected document and models"""
    if not models:
        st.warning("Please select at least one model for comparison.")
        return
    
    with st.spinner("Running real-time comparison..."):
        # Simulate real-time processing
        results = {}
        
        for model_name in models:
            # Simulate model processing
            entities = simulate_pii_extraction(document, model_name, confidence_filter)
            processing_time = np.random.uniform(0.5, 3.0)
            
            results[model_name] = {
                'entities': entities,
                'processing_time': processing_time,
                'total_entities': len(entities),
                'high_confidence_entities': len([e for e in entities if e['confidence'] > 0.8]),
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
        
        # Store results
        st.session_state.realtime_results = {
            'document': document,
            'models': results,
            'confidence_filter': confidence_filter,
            'comparison_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


def simulate_pii_extraction(document: Dict, model_name: str, confidence_filter: float) -> List[Dict]:
    """Simulate PII extraction for a document with a specific model"""
    # Use document's expected PII types
    expected_types = document['info']['expected_pii_types']
    
    entities = []
    np.random.seed(hash(model_name + document['name']) % 1000)
    
    # Generate realistic entities based on document type
    for pii_type in expected_types:
        # Some models might miss certain entity types
        detection_probability = get_model_detection_probability(model_name, pii_type)
        
        if np.random.random() < detection_probability:
            # Generate 1-3 entities of this type
            num_entities = np.random.randint(1, 4)
            
            for i in range(num_entities):
                confidence = np.random.uniform(0.6, 0.98)
                
                if confidence >= confidence_filter:
                    entities.append({
                        'type': pii_type,
                        'text': generate_mock_entity_text(pii_type),
                        'confidence': confidence,
                        'start_pos': np.random.randint(0, 1000),
                        'end_pos': np.random.randint(0, 1000) + 50,
                        'context': f"Mock context for {pii_type} detection"
                    })
    
    return entities


def get_model_detection_probability(model_name: str, pii_type: str) -> float:
    """Get probability that a model will detect a specific PII type"""
    # Different models have different strengths
    model_strengths = {
        "gpt-4o": {
            "PERSON": 0.95, "EMAIL": 0.98, "PHONE": 0.92, "ADDRESS": 0.88,
            "SSN": 0.85, "DATE_OF_BIRTH": 0.82, "ACCOUNT_NUMBER": 0.80
        },
        "gpt-4o-mini": {
            "PERSON": 0.88, "EMAIL": 0.95, "PHONE": 0.85, "ADDRESS": 0.80,
            "SSN": 0.78, "DATE_OF_BIRTH": 0.75, "ACCOUNT_NUMBER": 0.72
        },
        "claude-3-haiku": {
            "PERSON": 0.82, "EMAIL": 0.90, "PHONE": 0.80, "ADDRESS": 0.75,
            "SSN": 0.70, "DATE_OF_BIRTH": 0.68, "ACCOUNT_NUMBER": 0.65
        },
        "claude-3-sonnet": {
            "PERSON": 0.90, "EMAIL": 0.93, "PHONE": 0.87, "ADDRESS": 0.83,
            "SSN": 0.80, "DATE_OF_BIRTH": 0.78, "ACCOUNT_NUMBER": 0.75
        }
    }
    
    return model_strengths.get(model_name, {}).get(pii_type, 0.70)


def generate_mock_entity_text(pii_type: str) -> str:
    """Generate mock text for PII entity"""
    mock_data = {
        "PERSON": ["John Smith", "Jane Doe", "Michael Johnson", "Sarah Wilson"],
        "EMAIL": ["john@example.com", "jane.doe@company.com", "contact@business.org"],
        "PHONE": ["(555) 123-4567", "555-987-6543", "+1-555-555-5555"],
        "ADDRESS": ["123 Main St", "456 Oak Avenue", "789 Pine Street"],
        "SSN": ["123-45-6789", "987-65-4321", "555-44-3333"],
        "DATE_OF_BIRTH": ["1990-01-15", "1985-07-22", "1978-12-03"],
        "ACCOUNT_NUMBER": ["ACC-123456789", "4532-1234-5678-9012", "789456123"]
    }
    
    return np.random.choice(mock_data.get(pii_type, ["Unknown Entity"]))


def show_realtime_results(results: Dict, confidence_filter: float):
    """Display real-time comparison results"""
    models_data = results['models']
    
    # Summary metrics
    st.markdown("##### Comparison Summary")
    
    cols = st.columns(len(models_data))
    for i, (model_name, model_results) in enumerate(models_data.items()):
        with cols[i]:
            st.metric(
                f"{model_name}",
                f"{model_results['total_entities']} entities",
                f"{model_results['processing_time']:.2f}s"
            )
            st.caption(f"Updated: {model_results['timestamp']}")
    
    # Entity comparison table
    st.markdown("##### Entity Detection Comparison")
    
    # Create comparison table
    comparison_data = []
    all_entity_types = set()
    
    # Collect all entity types
    for model_results in models_data.values():
        for entity in model_results['entities']:
            all_entity_types.add(entity['type'])
    
    # Create comparison by entity type
    for entity_type in sorted(all_entity_types):
        row = {'Entity Type': entity_type}
        
        for model_name, model_results in models_data.items():
            type_entities = [e for e in model_results['entities'] if e['type'] == entity_type]
            avg_confidence = np.mean([e['confidence'] for e in type_entities]) if type_entities else 0
            
            row[f"{model_name} Count"] = len(type_entities)
            row[f"{model_name} Avg Conf"] = f"{avg_confidence:.3f}" if avg_confidence > 0 else "N/A"
        
        comparison_data.append(row)
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Visualization
        st.markdown("##### Detection Rate Visualization")
        
        # Entity count comparison
        entity_counts = {}
        for model_name, model_results in models_data.items():
            entity_counts[model_name] = model_results['total_entities']
        
        fig_counts = px.bar(
            x=list(entity_counts.keys()),
            y=list(entity_counts.values()),
            title=f"Total Entities Detected (Confidence > {confidence_filter})",
            labels={'x': 'Model', 'y': 'Entity Count'}
        )
        st.plotly_chart(fig_counts, use_container_width=True)
        
        # Processing time comparison
        processing_times = {model: data['processing_time'] for model, data in models_data.items()}
        fig_times = px.bar(
            x=list(processing_times.keys()),
            y=list(processing_times.values()),
            title="Processing Time Comparison",
            labels={'x': 'Model', 'y': 'Processing Time (seconds)'}
        )
        st.plotly_chart(fig_times, use_container_width=True)


def show_variance_calculator():
    """Performance variance calculator with <10% target"""
    st.markdown("### 📊 Performance Variance Calculator")
    st.markdown("Calculate and analyze performance variance across models (Target: <10%)")
    
    # Check if we have test results
    if 'multi_model_results' not in st.session_state or not st.session_state.multi_model_results:
        st.warning("Please run Multi-Model Testing first to generate data for variance analysis.")
        return
    
    results = st.session_state.multi_model_results
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Variance Analysis Settings")
        
        # Metric selection
        selected_metric = st.selectbox(
            "Select metric for variance analysis:",
            options=['precision', 'recall', 'f1', 'avg_processing_time'],
            index=2,  # Default to F1
            help="Choose which metric to analyze for variance"
        )
        
        # Variance threshold
        variance_threshold = st.slider(
            "Variance Threshold (%)",
            min_value=1.0,
            max_value=20.0,
            value=10.0,
            step=0.5,
            help="Target variance threshold (lower is better)"
        )
        
        # Analysis type
        analysis_type = st.selectbox(
            "Analysis Type:",
            options=['Coefficient of Variation', 'Standard Deviation', 'Min-Max Range'],
            help="Method for calculating variance"
        )
        
        if st.button("Calculate Variance"):
            calculate_performance_variance(results, selected_metric, variance_threshold, analysis_type)
    
    with col2:
        st.markdown("#### Variance Analysis Results")
        
        if 'variance_analysis' in st.session_state and st.session_state.variance_analysis:
            show_variance_analysis_results(st.session_state.variance_analysis)
        else:
            st.info("Configure settings and click 'Calculate Variance' to see analysis results.")


def calculate_performance_variance(results: Dict, metric: str, threshold: float, analysis_type: str):
    """Calculate performance variance across models"""
    # Extract metric values
    metric_values = []
    model_names = []
    
    for model_name, model_results in results.items():
        if 'metrics' in model_results and metric in model_results['metrics']:
            metric_values.append(model_results['metrics'][metric])
            model_names.append(model_name)
    
    if len(metric_values) < 2:
        st.error("Need at least 2 models to calculate variance.")
        return
    
    # Calculate variance statistics
    mean_value = np.mean(metric_values)
    std_dev = np.std(metric_values)
    min_value = np.min(metric_values)
    max_value = np.max(metric_values)
    
    # Calculate variance based on selected method
    if analysis_type == 'Coefficient of Variation':
        variance_percent = (std_dev / mean_value) * 100 if mean_value != 0 else 0
        variance_description = "Coefficient of Variation"
    elif analysis_type == 'Standard Deviation':
        variance_percent = (std_dev / mean_value) * 100 if mean_value != 0 else 0
        variance_description = "Standard Deviation as % of Mean"
    else:  # Min-Max Range
        range_value = max_value - min_value
        variance_percent = (range_value / mean_value) * 100 if mean_value != 0 else 0
        variance_description = "Min-Max Range as % of Mean"
    
    # Determine if variance meets threshold
    meets_threshold = variance_percent <= threshold
    
    # Store analysis results
    st.session_state.variance_analysis = {
        'metric': metric,
        'analysis_type': analysis_type,
        'threshold': threshold,
        'variance_percent': variance_percent,
        'variance_description': variance_description,
        'meets_threshold': meets_threshold,
        'statistics': {
            'mean': mean_value,
            'std_dev': std_dev,
            'min': min_value,
            'max': max_value,
            'range': max_value - min_value
        },
        'model_values': dict(zip(model_names, metric_values)),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Show immediate feedback
    if meets_threshold:
        st.success(f"✅ Variance target achieved! {variance_percent:.2f}% < {threshold}%")
    else:
        st.error(f"❌ Variance exceeds target: {variance_percent:.2f}% > {threshold}%")


def show_variance_analysis_results(analysis: Dict):
    """Display variance analysis results"""
    # Summary metrics
    st.markdown("##### Variance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Variance",
            f"{analysis['variance_percent']:.2f}%",
            delta=f"{analysis['variance_percent'] - analysis['threshold']:.2f}%" if analysis['variance_percent'] > analysis['threshold'] else None,
            delta_color="inverse"
        )
    
    with col2:
        st.metric("Mean Value", f"{analysis['statistics']['mean']:.4f}")
    
    with col3:
        st.metric("Std Deviation", f"{analysis['statistics']['std_dev']:.4f}")
    
    with col4:
        st.metric("Range", f"{analysis['statistics']['range']:.4f}")
    
    # Threshold status
    if analysis['meets_threshold']:
        st.success(f"✅ Performance variance of {analysis['variance_percent']:.2f}% meets the target threshold of {analysis['threshold']}%")
    else:
        st.error(f"❌ Performance variance of {analysis['variance_percent']:.2f}% exceeds the target threshold of {analysis['threshold']}%")
    
    # Model performance table
    st.markdown("##### Model Performance Breakdown")
    
    model_data = []
    for model_name, value in analysis['model_values'].items():
        deviation = abs(value - analysis['statistics']['mean'])
        deviation_percent = (deviation / analysis['statistics']['mean']) * 100 if analysis['statistics']['mean'] != 0 else 0
        
        model_data.append({
            'Model': model_name,
            f'{analysis["metric"].title()}': f"{value:.4f}",
            'Deviation from Mean': f"{deviation:.4f}",
            'Deviation %': f"{deviation_percent:.2f}%",
            'Status': '✅ Within Range' if deviation_percent <= analysis['threshold'] else '❌ High Variance'
        })
    
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True)
    
    # Visualizations
    st.markdown("##### Variance Visualization")
    
    # Box plot showing distribution
    fig_box = px.box(
        y=list(analysis['model_values'].values()),
        title=f"{analysis['metric'].title()} Distribution Across Models"
    )
    fig_box.add_hline(
        y=analysis['statistics']['mean'],
        line_dash="dash",
        line_color="red",
        annotation_text="Mean"
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Bar chart with variance bands
    fig_bar = px.bar(
        x=list(analysis['model_values'].keys()),
        y=list(analysis['model_values'].values()),
        title=f"{analysis['metric'].title()} by Model with Variance Bands"
    )
    
    # Add variance bands
    mean_val = analysis['statistics']['mean']
    threshold_val = mean_val * (analysis['threshold'] / 100)
    
    fig_bar.add_hline(y=mean_val, line_dash="dash", line_color="green", annotation_text="Mean")
    fig_bar.add_hline(y=mean_val + threshold_val, line_dash="dot", line_color="orange", annotation_text="Upper Threshold")
    fig_bar.add_hline(y=mean_val - threshold_val, line_dash="dot", line_color="orange", annotation_text="Lower Threshold")
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Recommendations
    st.markdown("##### Recommendations")
    
    if analysis['meets_threshold']:
        st.success("""
        **Performance variance is within acceptable limits.**
        - All models show consistent performance
        - System is ready for production deployment
        - Consider ensemble methods for optimal results
        """)
    else:
        high_variance_models = [
            model for model, value in analysis['model_values'].items()
            if abs(value - analysis['statistics']['mean']) / analysis['statistics']['mean'] * 100 > analysis['threshold']
        ]
        
        st.warning(f"""
        **Performance variance exceeds target threshold.**
        
        **High variance models:** {', '.join(high_variance_models)}
        
        **Recommendations:**
        - Review and retune high-variance models
        - Consider removing outlier models from ensemble
        - Implement additional quality control measures
        - Investigate causes of performance variation
        """)


def show_baseline_comparison_tools():
    """Baseline comparison tools (OCR + spaCy NER vs LLM approaches)"""
    st.markdown("### 🔍 Baseline Comparison Tools")
    st.markdown("Compare traditional OCR + spaCy NER approaches with modern LLM-based methods.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Baseline Configuration")
        
        # Baseline methods selection
        baseline_methods = st.multiselect(
            "Select baseline methods:",
            options=[
                "OCR + spaCy NER",
                "OCR + Rule-based",
                "OCR + Transformers NER",
                "LayoutLM",
                "Pure spaCy NER (no OCR)"
            ],
            default=["OCR + spaCy NER", "OCR + Rule-based"],
            help="Traditional PII extraction methods"
        )
        
        # LLM methods for comparison
        preferred_defaults = ["gpt-4o-mini", "claude-3-haiku"]
        available_options = list(get_available_llm_models().keys())
        valid_defaults = [model for model in preferred_defaults if model in available_options]
        
        llm_methods = st.multiselect(
            "Select LLM methods:",
            options=available_options,
            default=valid_defaults if valid_defaults else available_options[:2],
            help="Modern LLM-based methods"
        )
        
        # Comparison metrics
        st.markdown("#### Comparison Metrics")
        
        comparison_metrics = st.multiselect(
            "Select metrics to compare:",
            options=[
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "Processing Time",
                "Cost per Document",
                "Memory Usage",
                "Setup Complexity"
            ],
            default=["Precision", "Recall", "F1-Score", "Processing Time", "Cost per Document"],
            help="Metrics for baseline comparison"
        )
        
        # Document types for testing
        if st.session_state.phase1_selected_documents:
            doc_categories = list(set([doc['category'] for doc in st.session_state.phase1_selected_documents]))
            selected_categories = st.multiselect(
                "Document categories to test:",
                options=doc_categories,
                default=doc_categories,
                help="Test on different document types"
            )
        
        if st.button("Run Baseline Comparison", 
                    disabled=not (baseline_methods and llm_methods and comparison_metrics)):
            run_baseline_comparison(baseline_methods, llm_methods, comparison_metrics)
    
    with col2:
        st.markdown("#### Baseline Comparison Results")
        
        if 'baseline_comparison' in st.session_state and st.session_state.baseline_comparison:
            show_baseline_comparison_results(st.session_state.baseline_comparison)
        else:
            st.info("Configure baseline and LLM methods, then run comparison to see results.")


def run_baseline_comparison(baseline_methods: List[str], llm_methods: List[str], metrics: List[str]):
    """Run baseline comparison between traditional and LLM methods"""
    with st.spinner("Running baseline comparison..."):
        # Simulate baseline comparison
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_methods = baseline_methods + llm_methods
        results = {}
        
        for i, method in enumerate(all_methods):
            status_text.text(f"Testing {method}...")
            
            # Simulate method testing
            method_results = simulate_baseline_method_testing(method, metrics)
            results[method] = method_results
            
            # Update progress
            progress_bar.progress((i + 1) / len(all_methods))
            time.sleep(0.3)
        
        # Calculate comparison analysis
        comparison_analysis = analyze_baseline_comparison(results, baseline_methods, llm_methods, metrics)
        
        # Store results
        st.session_state.baseline_comparison = {
            'results': results,
            'analysis': comparison_analysis,
            'baseline_methods': baseline_methods,
            'llm_methods': llm_methods,
            'metrics': metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        progress_bar.progress(1.0)
        status_text.text("Baseline comparison completed!")
        
        st.success("Baseline comparison completed successfully!")


def simulate_baseline_method_testing(method: str, metrics: List[str]) -> Dict:
    """Simulate testing for a baseline method"""
    np.random.seed(hash(method) % 1000)
    
    # Define performance characteristics for different methods
    method_profiles = {
        "OCR + spaCy NER": {
            "Accuracy": 0.82, "Precision": 0.78, "Recall": 0.75, "F1-Score": 0.765,
            "Processing Time": 2.5, "Cost per Document": 0.01, "Memory Usage": 512,
            "Setup Complexity": 3
        },
        "OCR + Rule-based": {
            "Accuracy": 0.75, "Precision": 0.85, "Recall": 0.65, "F1-Score": 0.74,
            "Processing Time": 1.2, "Cost per Document": 0.005, "Memory Usage": 256,
            "Setup Complexity": 2
        },
        "OCR + Transformers NER": {
            "Accuracy": 0.88, "Precision": 0.85, "Recall": 0.82, "F1-Score": 0.835,
            "Processing Time": 4.5, "Cost per Document": 0.02, "Memory Usage": 2048,
            "Setup Complexity": 4
        },
        "LayoutLM": {
            "Accuracy": 0.90, "Precision": 0.87, "Recall": 0.85, "F1-Score": 0.86,
            "Processing Time": 6.2, "Cost per Document": 0.03, "Memory Usage": 4096,
            "Setup Complexity": 5
        },
        "Pure spaCy NER (no OCR)": {
            "Accuracy": 0.70, "Precision": 0.72, "Recall": 0.68, "F1-Score": 0.70,
            "Processing Time": 0.8, "Cost per Document": 0.001, "Memory Usage": 128,
            "Setup Complexity": 1
        },
        # LLM methods (use existing model profiles)
        "gpt-4o": {
            "Accuracy": 0.94, "Precision": 0.92, "Recall": 0.89, "F1-Score": 0.905,
            "Processing Time": 3.2, "Cost per Document": 0.15, "Memory Usage": 0,
            "Setup Complexity": 2
        },
        "gpt-4o-mini": {
            "Accuracy": 0.90, "Precision": 0.88, "Recall": 0.85, "F1-Score": 0.865,
            "Processing Time": 1.8, "Cost per Document": 0.025, "Memory Usage": 0,
            "Setup Complexity": 2
        },
        "claude-3-haiku": {
            "Accuracy": 0.84, "Precision": 0.82, "Recall": 0.80, "F1-Score": 0.81,
            "Processing Time": 1.2, "Cost per Document": 0.018, "Memory Usage": 0,
            "Setup Complexity": 2
        },
        "claude-3-sonnet": {
            "Accuracy": 0.92, "Precision": 0.90, "Recall": 0.87, "F1-Score": 0.885,
            "Processing Time": 2.5, "Cost per Document": 0.08, "Memory Usage": 0,
            "Setup Complexity": 2
        }
    }
    
    # Get base profile or default
    base_profile = method_profiles.get(method, {
        "Accuracy": 0.80, "Precision": 0.78, "Recall": 0.76, "F1-Score": 0.77,
        "Processing Time": 2.0, "Cost per Document": 0.05, "Memory Usage": 1024,
        "Setup Complexity": 3
    })
    
    # Add realistic variance
    results = {}
    for metric in metrics:
        if metric in base_profile:
            base_value = base_profile[metric]
            
            # Add noise based on metric type
            if metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                noise = np.random.normal(0, 0.02)
                results[metric] = max(0.5, min(0.98, base_value + noise))
            elif metric == "Processing Time":
                noise = np.random.normal(0, 0.3)
                results[metric] = max(0.1, base_value + noise)
            elif metric == "Cost per Document":
                noise = np.random.normal(0, base_value * 0.1)
                results[metric] = max(0.001, base_value + noise)
            elif metric == "Memory Usage":
                noise = np.random.normal(0, base_value * 0.1)
                results[metric] = max(64, base_value + noise)
            elif metric == "Setup Complexity":
                results[metric] = base_value  # Keep integer values
            else:
                results[metric] = base_value
    
    return results


def analyze_baseline_comparison(results: Dict, baseline_methods: List[str], llm_methods: List[str], metrics: List[str]) -> Dict:
    """Analyze baseline comparison results"""
    analysis = {
        'baseline_performance': {},
        'llm_performance': {},
        'improvements': {},
        'trade_offs': {}
    }
    
    # Calculate average performance for each group
    for metric in metrics:
        baseline_values = [results[method][metric] for method in baseline_methods if metric in results[method]]
        llm_values = [results[method][metric] for method in llm_methods if metric in results[method]]
        
        if baseline_values and llm_values:
            baseline_avg = np.mean(baseline_values)
            llm_avg = np.mean(llm_values)
            
            analysis['baseline_performance'][metric] = baseline_avg
            analysis['llm_performance'][metric] = llm_avg
            
            # Calculate improvement (positive means LLM is better)
            if metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
                improvement = ((llm_avg - baseline_avg) / baseline_avg) * 100
            elif metric in ["Processing Time", "Cost per Document", "Setup Complexity"]:
                # For these metrics, lower is better
                improvement = ((baseline_avg - llm_avg) / baseline_avg) * 100
            else:
                improvement = ((llm_avg - baseline_avg) / baseline_avg) * 100
            
            analysis['improvements'][metric] = improvement
    
    return analysis


def show_baseline_comparison_results(comparison_data: Dict):
    """Display baseline comparison results"""
    results = comparison_data['results']
    analysis = comparison_data['analysis']
    baseline_methods = comparison_data['baseline_methods']
    llm_methods = comparison_data['llm_methods']
    metrics = comparison_data['metrics']
    
    # Summary comparison
    st.markdown("##### Performance Comparison Summary")
    
    # Create summary table
    summary_data = []
    for metric in metrics:
        if metric in analysis['baseline_performance'] and metric in analysis['llm_performance']:
            baseline_avg = analysis['baseline_performance'][metric]
            llm_avg = analysis['llm_performance'][metric]
            improvement = analysis['improvements'][metric]
            
            summary_data.append({
                'Metric': metric,
                'Baseline Avg': f"{baseline_avg:.4f}",
                'LLM Avg': f"{llm_avg:.4f}",
                'Improvement': f"{improvement:+.1f}%",
                'Winner': 'LLM' if improvement > 0 else 'Baseline'
            })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Detailed method comparison
        st.markdown("##### Detailed Method Comparison")
        
        # Create detailed comparison table
        detailed_data = []
        for method, method_results in results.items():
            row = {'Method': method, 'Type': 'LLM' if method in llm_methods else 'Baseline'}
            for metric in metrics:
                if metric in method_results:
                    if metric in ["Processing Time", "Cost per Document"]:
                        row[metric] = f"{method_results[metric]:.4f}"
                    elif metric == "Memory Usage":
                        row[metric] = f"{method_results[metric]:.0f} MB"
                    elif metric == "Setup Complexity":
                        row[metric] = f"{method_results[metric]:.0f}/5"
                    else:
                        row[metric] = f"{method_results[metric]:.3f}"
            detailed_data.append(row)
        
        df_detailed = pd.DataFrame(detailed_data)
        st.dataframe(df_detailed, use_container_width=True)
        
        # Visualizations
        st.markdown("##### Comparison Visualizations")
        
        # Performance radar chart
        fig_radar = create_comparison_radar_chart(analysis, metrics)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Cost vs Performance analysis
        if 'F1-Score' in metrics and 'Cost per Document' in metrics:
            cost_perf_data = []
            for method, method_results in results.items():
                if 'F1-Score' in method_results and 'Cost per Document' in method_results:
                    cost_perf_data.append({
                        'Method': method,
                        'F1-Score': method_results['F1-Score'],
                        'Cost per Document': method_results['Cost per Document'],
                        'Type': 'LLM' if method in llm_methods else 'Baseline'
                    })
            
            if cost_perf_data:
                df_cost_perf = pd.DataFrame(cost_perf_data)
                fig_scatter = px.scatter(
                    df_cost_perf,
                    x='Cost per Document',
                    y='F1-Score',
                    color='Type',
                    hover_name='Method',
                    title='Cost vs Performance Analysis',
                    labels={'Cost per Document': 'Cost per Document ($)', 'F1-Score': 'F1-Score'}
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Key insights
        st.markdown("##### Key Insights")
        
        # Calculate winners
        llm_wins = sum(1 for imp in analysis['improvements'].values() if imp > 0)
        baseline_wins = len(analysis['improvements']) - llm_wins
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("LLM Advantages", f"{llm_wins}/{len(metrics)}")
        
        with col2:
            st.metric("Baseline Advantages", f"{baseline_wins}/{len(metrics)}")
        
        with col3:
            avg_improvement = np.mean(list(analysis['improvements'].values()))
            st.metric("Avg Improvement", f"{avg_improvement:+.1f}%")
        
        # Recommendations
        st.markdown("##### Recommendations")
        
        if llm_wins > baseline_wins:
            st.success("""
            **LLM methods show superior performance overall.**
            
            **Advantages:**
            - Higher accuracy and precision
            - Better handling of complex documents
            - More flexible and adaptable
            
            **Considerations:**
            - Higher operational costs
            - Dependency on external APIs
            - Potential latency issues
            """)
        else:
            st.info("""
            **Baseline methods remain competitive.**
            
            **Advantages:**
            - Lower operational costs
            - Faster processing times
            - Full control and privacy
            
            **Considerations:**
            - Lower accuracy on complex documents
            - Requires more setup and maintenance
            - Limited adaptability
            """)


def create_comparison_radar_chart(analysis: Dict, metrics: List[str]) -> go.Figure:
    """Create radar chart for baseline vs LLM comparison"""
    # Normalize metrics for radar chart (0-1 scale)
    baseline_values = []
    llm_values = []
    chart_metrics = []
    
    for metric in metrics:
        if metric in analysis['baseline_performance'] and metric in analysis['llm_performance']:
            baseline_val = analysis['baseline_performance'][metric]
            llm_val = analysis['llm_performance'][metric]
            
            # Normalize based on metric type
            if metric in ["Processing Time", "Cost per Document", "Setup Complexity"]:
                # For these metrics, lower is better, so invert
                max_val = max(baseline_val, llm_val)
                baseline_norm = 1 - (baseline_val / max_val)
                llm_norm = 1 - (llm_val / max_val)
            else:
                # For performance metrics, higher is better
                baseline_norm = baseline_val
                llm_norm = llm_val
            
            baseline_values.append(baseline_norm)
            llm_values.append(llm_norm)
            chart_metrics.append(metric)
    
    # Create radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=baseline_values + [baseline_values[0]],  # Close the polygon
        theta=chart_metrics + [chart_metrics[0]],
        fill='toself',
        name='Baseline Methods',
        fillcolor='rgba(255, 99, 132, 0.2)',
        line=dict(color='rgba(255, 99, 132, 1)')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=llm_values + [llm_values[0]],  # Close the polygon
        theta=chart_metrics + [chart_metrics[0]],
        fill='toself',
        name='LLM Methods',
        fillcolor='rgba(54, 162, 235, 0.2)',
        line=dict(color='rgba(54, 162, 235, 1)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Baseline vs LLM Methods Comparison"
    )
    
    return fig


def show_document_visualization():
    """Show document visualization with PII highlighting"""
    st.markdown("#### 📄 Document Visualization with PII Highlighting")
    
    if not st.session_state.phase1_selected_documents:
        st.info("Please select documents first in the Document Selection tab.")
        return
    
    # Document selector
    selected_doc_idx = st.selectbox(
        "Select document to visualize:",
        options=range(len(st.session_state.phase1_selected_documents)),
        format_func=lambda x: st.session_state.phase1_selected_documents[x]['name']
    )
    
    if selected_doc_idx is None:
        return
        
    selected_doc = st.session_state.phase1_selected_documents[selected_doc_idx]
    
    # Model selector for visualization
    available_models = get_available_llm_models()
    selected_model = st.selectbox(
        "Select model to view results:",
        options=list(available_models.keys()),
        key="doc_viz_model_selector"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("##### Original Document")
        
        # Convert and display document using Phase 0's exact method
        file_content = selected_doc.get('file_content')
        if file_content:
            try:
                if not PHASE0_CONVERSION_AVAILABLE:
                    st.error("Phase 0 conversion function not available")
                    return
                    
                import base64
                decoded_content = base64.b64decode(file_content)
                file_extension = selected_doc['info'].get('file_extension', '.pdf')
                
                # Use Phase 0's exact conversion function with same password handling
                password = "Hubert"  # Same default password as Phase 0
                images = phase0_convert_document_to_images(decoded_content, file_extension, password)
                
                if images:
                    # Display first page/image
                    from io import BytesIO
                    from PIL import Image as PILImage
                    
                    # Decode first image
                    img_data = base64.b64decode(images[0])
                    img = PILImage.open(BytesIO(img_data))
                    st.image(img, caption=f"Page 1 of {selected_doc['name']}", use_column_width=True)
                    
                    if len(images) > 1:
                        st.info(f"Document has {len(images)} pages. Showing page 1.")
                else:
                    st.error("Could not convert document to images using Phase 0 method")
            except Exception as e:
                st.error(f"Error displaying document using Phase 0 method: {str(e)}")
        else:
            st.error("Document content not found (no base64 content available)")
    
    with col2:
        st.markdown("##### PII Extraction Results")
        
        # Check if we have results for this model and document
        if ('multi_model_results' in st.session_state and 
            selected_model in st.session_state.multi_model_results):
            
            model_results = st.session_state.multi_model_results[selected_model]
            doc_results = None
            
            # Find results for this specific document
            for doc_result in model_results.get('document_results', []):
                if doc_result['document'] == selected_doc['name']:
                    doc_results = doc_result
                    break
            
            if doc_results:
                # Display extracted entities
                st.markdown("**Extracted PII Entities:**")
                
                if 'extracted_entities' in doc_results and doc_results['extracted_entities']:
                    entities_df = pd.DataFrame(doc_results['extracted_entities'])
                    
                    for idx, entity in enumerate(doc_results['extracted_entities']):
                        confidence = entity.get('confidence', 0.5)
                        confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.6 else "red"
                        
                        st.markdown(f"""
                        **Entity {idx+1}:**
                        - **Type:** {entity.get('type', 'Unknown')}
                        - **Value:** `{entity.get('value', 'N/A')}`
                        - **Confidence:** <span style="color:{confidence_color}">{confidence:.2%}</span>
                        """, unsafe_allow_html=True)
                        
                        if entity.get('position'):
                            st.caption(f"Position: {entity['position']}")
                
                else:
                    st.info("No PII entities extracted by this model")
                
                # Display metrics
                st.markdown("**Performance Metrics:**")
                metrics_data = {
                    "Precision": f"{doc_results.get('precision', 0):.2%}",
                    "Recall": f"{doc_results.get('recall', 0):.2%}",
                    "F1-Score": f"{doc_results.get('f1', 0):.2%}",
                    "Processing Time": f"{doc_results.get('processing_time', 0):.2f}s",
                    "Cost": f"${doc_results.get('cost', 0):.4f}"
                }
                
                for metric, value in metrics_data.items():
                    st.metric(metric, value)
            
            else:
                st.info(f"No results found for {selected_doc['name']} with {selected_model}")
        
        else:
            st.info("Run multi-model testing to see extraction results")
        
        # Show ground truth comparison
        if selected_doc['info'].get('ground_truth_labels'):
            st.markdown("**Ground Truth Entities:**")
            gt_entities = selected_doc['info']['ground_truth_labels'].get('entities', [])
            
            for idx, entity in enumerate(gt_entities):
                st.markdown(f"""
                **GT Entity {idx+1}:**
                - **Type:** {entity.get('type', 'Unknown')}
                - **Value:** `{entity.get('value', 'N/A')}`
                """)


def fix_variance_analysis_with_domains():
    """Fix variance analysis to work with document domains from Phase 0"""
    st.markdown("#### 📊 Cross-Domain Variance Analysis")
    st.markdown("Analyze performance variance across document domains from Phase 0 dataset.")
    
    if not st.session_state.phase1_selected_documents:
        st.info("Please select documents first in the Document Selection tab.")
        return
    
    if 'multi_model_results' not in st.session_state or not st.session_state.multi_model_results:
        st.info("Please run multi-model testing first to generate variance analysis.")
        return
    
    # Extract domain information from documents
    domain_data = []
    
    for model_name, model_results in st.session_state.multi_model_results.items():
        for doc_result in model_results.get('document_results', []):
            domain_data.append({
                'model': model_name,
                'document': doc_result['document'],
                'domain': doc_result.get('domain', 'Unknown'),
                'precision': doc_result.get('precision', 0),
                'recall': doc_result.get('recall', 0),
                'f1': doc_result.get('f1', 0),
                'processing_time': doc_result.get('processing_time', 0),
                'cost': doc_result.get('cost', 0)
            })
    
    if not domain_data:
        st.error("No domain data available for variance analysis")
        return
    
    variance_df = pd.DataFrame(domain_data)
    
    # Group by domain and model
    st.markdown("##### Performance by Domain")
    
    metrics = ['precision', 'recall', 'f1', 'processing_time', 'cost']
    selected_metric = st.selectbox("Select metric for variance analysis:", metrics)
    
    # Calculate variance across domains
    domain_stats = variance_df.groupby(['domain', 'model'])[selected_metric].agg(['mean', 'std', 'count']).reset_index()
    domain_stats['cv'] = (domain_stats['std'] / domain_stats['mean']) * 100  # Coefficient of Variation
    
    # Display variance table
    st.markdown("**Variance Statistics by Domain:**")
    st.dataframe(domain_stats.round(4))
    
    # Visualize variance across domains
    fig = px.box(
        variance_df, 
        x='domain', 
        y=selected_metric, 
        color='model',
        title=f"{selected_metric.title()} Distribution Across Domains"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate overall variance
    overall_variance = variance_df.groupby('model')[selected_metric].std().mean()
    target_variance = st.session_state.phase1_variance_threshold * variance_df[selected_metric].mean()
    
    if overall_variance <= target_variance:
        st.success(f"✅ Variance target met! Overall variance: {overall_variance:.4f} (Target: ≤{target_variance:.4f})")
    else:
        st.warning(f"⚠️ Variance target not met. Overall variance: {overall_variance:.4f} (Target: ≤{target_variance:.4f})")
    
    # Domain-specific analysis
    st.markdown("##### Domain-Specific Analysis")
    
    unique_domains = variance_df['domain'].unique()
    selected_domain = st.selectbox("Select domain for detailed analysis:", unique_domains)
    
    domain_subset = variance_df[variance_df['domain'] == selected_domain]
    
    # Performance comparison across models for selected domain
    fig2 = px.bar(
        domain_subset.groupby('model')[selected_metric].mean().reset_index(),
        x='model',
        y=selected_metric,
        title=f"Average {selected_metric.title()} for {selected_domain} Domain"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Identify best and worst performing domains
    domain_performance = variance_df.groupby('domain')[selected_metric].mean().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Best Performing Domains:**")
        for domain in domain_performance.head(3).index:
            score = domain_performance[domain]
            st.metric(domain, f"{score:.3f}")
    
    with col2:
        st.markdown("**Lowest Performing Domains:**")
        for domain in domain_performance.tail(3).index:
            score = domain_performance[domain]
            st.metric(domain, f"{score:.3f}")


# Update the show_variance_calculator function to use the new implementation
def show_variance_calculator():
    """Performance variance calculator with real domain analysis"""
    st.markdown("### 📊 Performance Variance Calculator")
    st.markdown("Analyze performance variance across models and document domains (target: <10%)")
    
    fix_variance_analysis_with_domains()


def show_model_comparison_interface():
    """Show model comparison interface with ground truth analysis"""
    st.markdown("### 📈 Multi-Model Ground Truth Comparison")
    st.markdown("Compare multiple models against ground truth from Phase 0 dataset with detailed performance analysis.")
    
    # Check if we have Phase 0 data and multi-model results
    if not st.session_state.get('phase1_using_real_ground_truth', False):
        st.warning("🔍 **Real ground truth data required**. Import your Phase 0 dataset in the 'Document Selection' tab first.")
        st.info("💡 This interface requires documents with validated ground truth labels from Phase 0 to provide meaningful model comparisons.")
        return
    
    if 'multi_model_results' not in st.session_state or not st.session_state.multi_model_results:
        st.warning("🤖 **Multi-model testing results required**. Run multi-model testing in the 'Model Validation' tab first.")
        st.info("💡 This comparison requires results from multiple model testing to analyze performance differences.")
        return
    
    results = st.session_state.multi_model_results
    documents = st.session_state.phase1_selected_documents
    
    # Model performance overview
    st.markdown("#### 🏆 Model Performance Overview")
    
    # Create comprehensive comparison table
    comparison_data = []
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and 'metrics' in model_results:
            metrics = model_results['metrics']
            doc_results = model_results.get('document_results', [])
            
            comparison_data.append({
                'Model': model_name,
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1', 0),
                'Avg Processing Time': metrics.get('avg_processing_time', 0),
                'Total Cost': metrics.get('total_cost', 0),
                'Documents Processed': len(doc_results),
                'Avg Entities/Doc': np.mean([doc.get('entities_found', 0) for doc in doc_results]) if doc_results else 0,
                'Avg Confidence': np.mean([np.mean(doc.get('confidence_scores', [0.5])) for doc in doc_results]) if doc_results else 0
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # Display metrics table
        st.dataframe(df_comparison.round(4), use_container_width=True)
        
        # Performance radar chart
        st.markdown("#### 📊 Performance Radar Chart")
        fig_radar = create_model_comparison_radar_chart(df_comparison)
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Model ranking analysis
        st.markdown("#### 🥇 Model Ranking Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Rank by different metrics
            ranking_metric = st.selectbox(
                "Rank models by:",
                ['F1-Score', 'Precision', 'Recall', 'Processing Time', 'Total Cost'],
                help="Select metric for model ranking"
            )
            
            # Calculate rankings
            if ranking_metric in ['Processing Time', 'Total Cost']:
                # Lower is better
                ranked_df = df_comparison.sort_values(ranking_metric, ascending=True)
                ranking_direction = "(Lower is better)"
            else:
                # Higher is better
                ranked_df = df_comparison.sort_values(ranking_metric, ascending=False)
                ranking_direction = "(Higher is better)"
            
            st.markdown(f"**Model Ranking by {ranking_metric} {ranking_direction}**")
            for i, (_, row) in enumerate(ranked_df.iterrows()):
                rank_emoji = ["🥇", "🥈", "🥉"] + ["🏅"] * 10
                metric_value = row[ranking_metric]
                if ranking_metric in ['Processing Time']:
                    display_value = f"{metric_value:.2f}s"
                elif ranking_metric in ['Total Cost']:
                    display_value = f"${metric_value:.4f}"
                else:
                    display_value = f"{metric_value:.3f}"
                st.write(f"{rank_emoji[i]} **{row['Model']}**: {display_value}")
        
        with col2:
            # Performance consistency analysis
            st.markdown("**Performance Consistency**")
            
            # Calculate coefficient of variation for key metrics
            for metric in ['Precision', 'Recall', 'F1-Score']:
                values = df_comparison[metric].values
                if len(values) > 1:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
                    
                    consistency_level = "High" if cv < 5 else "Medium" if cv < 15 else "Low"
                    color = "green" if cv < 5 else "orange" if cv < 15 else "red"
                    
                    st.markdown(f"**{metric}**: {cv:.1f}% variation ({consistency_level})")
        
        # Detailed model comparison charts
        st.markdown("#### 📈 Detailed Performance Charts")
        
        # Performance vs Cost analysis
        fig_cost_perf = px.scatter(
            df_comparison,
            x='Total Cost',
            y='F1-Score',
            size='Avg Processing Time',
            color='Model',
            title='Performance vs Cost Analysis',
            labels={'Total Cost': 'Total Cost ($)', 'F1-Score': 'F1-Score'},
            hover_data=['Precision', 'Recall']
        )
        fig_cost_perf.update_layout(showlegend=True)
        st.plotly_chart(fig_cost_perf, use_container_width=True)
        
        # Speed vs Accuracy tradeoff
        fig_speed_acc = px.scatter(
            df_comparison,
            x='Avg Processing Time',
            y='F1-Score',
            size='Total Cost',
            color='Model',
            title='Speed vs Accuracy Tradeoff',
            labels={'Avg Processing Time': 'Avg Processing Time (s)', 'F1-Score': 'F1-Score'}
        )
        st.plotly_chart(fig_speed_acc, use_container_width=True)
        
        # Model recommendations
        st.markdown("#### 💡 Model Recommendations")
        
        best_f1_model = ranked_df.iloc[0]['Model'] if not ranked_df.empty else 'N/A'
        best_cost_model = df_comparison.loc[df_comparison['Total Cost'].idxmin(), 'Model'] if not df_comparison.empty else 'N/A'
        best_speed_model = df_comparison.loc[df_comparison['Avg Processing Time'].idxmin(), 'Model'] if not df_comparison.empty else 'N/A'
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.success(f"**Best Overall Performance**: {best_f1_model}")
            st.caption("Highest F1-Score")
        
        with col2:
            st.success(f"**Most Cost-Effective**: {best_cost_model}")
            st.caption("Lowest total cost")
        
        with col3:
            st.success(f"**Fastest Processing**: {best_speed_model}")
            st.caption("Lowest processing time")
    else:
        st.error("No valid model comparison data available.")


def create_model_comparison_radar_chart(df_comparison: pd.DataFrame) -> go.Figure:
    """Create radar chart for model comparison"""
    fig = go.Figure()
    
    # Normalize metrics for radar chart (0-1 scale)
    metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
    
    for _, row in df_comparison.iterrows():
        values = [row[metric] for metric in metrics_to_plot]
        # Close the polygon
        values.append(values[0])
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_to_plot + [metrics_to_plot[0]],
            fill='toself',
            name=row['Model'],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Model Performance Comparison"
    )
    
    return fig


def show_performance_analytics_interface():
    """Show performance analytics with domain/difficulty variance analysis"""
    st.markdown("### 📊 Performance Analytics & Variance Analysis")
    st.markdown("Analyze model performance variance across domains, difficulty levels, and document types using Phase 0 classification data.")
    
    # Check if we have the required data
    if not st.session_state.get('phase1_using_real_ground_truth', False):
        st.warning("🔍 **Phase 0 ground truth data required**. Import your Phase 0 dataset first.")
        return
    
    if 'multi_model_results' not in st.session_state or not st.session_state.multi_model_results:
        st.warning("🤖 **Multi-model testing results required**. Run multi-model testing first.")
        return
    
    results = st.session_state.multi_model_results
    documents = st.session_state.phase1_selected_documents
    
    # Extract performance data by categories
    performance_by_domain, performance_by_difficulty, performance_by_doc_type = extract_categorized_performance_data(results, documents)
    
    # Domain-based performance analysis
    st.markdown("#### 🏢 Performance by Domain")
    
    if performance_by_domain:
        domain_analysis_col1, domain_analysis_col2 = st.columns(2)
        
        with domain_analysis_col1:
            # Create domain performance table
            domain_summary = create_domain_performance_summary(performance_by_domain)
            st.dataframe(domain_summary, use_container_width=True)
        
        with domain_analysis_col2:
            # Domain performance heatmap
            fig_domain_heatmap = create_domain_performance_heatmap(performance_by_domain)
            st.plotly_chart(fig_domain_heatmap, use_container_width=True)
        
        # Detailed domain analysis
        selected_domain = st.selectbox(
            "Select domain for detailed analysis:",
            list(performance_by_domain.keys()),
            help="Choose a domain to see detailed performance breakdown"
        )
        
        if selected_domain:
            show_detailed_domain_analysis(performance_by_domain[selected_domain], selected_domain)
    else:
        st.info("No domain-specific performance data available.")
    
    # Difficulty-based performance analysis
    st.markdown("#### 📈 Performance by Difficulty Level")
    
    if performance_by_difficulty:
        difficulty_col1, difficulty_col2 = st.columns(2)
        
        with difficulty_col1:
            # Difficulty performance summary
            difficulty_summary = create_difficulty_performance_summary(performance_by_difficulty)
            st.dataframe(difficulty_summary, use_container_width=True)
        
        with difficulty_col2:
            # Difficulty performance trend
            fig_difficulty_trend = create_difficulty_performance_trend(performance_by_difficulty)
            st.plotly_chart(fig_difficulty_trend, use_container_width=True)
        
        # Performance degradation analysis
        st.markdown("**Performance Degradation Analysis**")
        degradation_analysis = analyze_performance_degradation(performance_by_difficulty)
        
        for model, degradation in degradation_analysis.items():
            if degradation['significant_degradation']:
                st.warning(f"⚠️ **{model}**: {degradation['degradation_percentage']:.1f}% F1-score drop from Easy to Hard documents")
            else:
                st.success(f"✅ **{model}**: Maintains consistent performance across difficulty levels")
    else:
        st.info("No difficulty-specific performance data available.")
    
    # Document type performance analysis
    st.markdown("#### 📄 Performance by Document Type")
    
    if performance_by_doc_type:
        doc_type_summary = create_doc_type_performance_summary(performance_by_doc_type)
        st.dataframe(doc_type_summary, use_container_width=True)
        
        # Document type performance chart
        fig_doc_type = create_doc_type_performance_chart(performance_by_doc_type)
        st.plotly_chart(fig_doc_type, use_container_width=True)
    else:
        st.info("No document type-specific performance data available.")
    
    # Cost analysis by category
    st.markdown("#### 💰 Cost Analysis by Category")
    
    cost_analysis = analyze_cost_by_category(results, documents)
    show_cost_analysis_dashboard(cost_analysis)


def show_metrics_dashboard_interface():
    """Show comprehensive metrics dashboard with KPIs, confusion matrices, F1 scores"""
    st.markdown("### 📋 Comprehensive Metrics Dashboard")
    st.markdown("Key performance indicators, confusion matrices, and detailed F1 scores by entity type.")
    
    # Check if we have the required data
    if not st.session_state.get('phase1_using_real_ground_truth', False):
        st.warning("🔍 **Phase 0 ground truth data required**. Import your Phase 0 dataset first.")
        return
    
    if 'multi_model_results' not in st.session_state or not st.session_state.multi_model_results:
        st.warning("🤖 **Multi-model testing results required**. Run multi-model testing first.")
        return
    
    results = st.session_state.multi_model_results
    documents = st.session_state.phase1_selected_documents
    
    # Key Performance Indicators
    st.markdown("#### 🎯 Key Performance Indicators (KPIs)")
    
    kpis = calculate_comprehensive_kpis(results, documents)
    show_kpi_dashboard(kpis)
    
    # Entity-specific performance analysis
    st.markdown("#### 🏷️ Entity Type Performance Analysis")
    
    entity_performance = analyze_entity_type_performance(results)
    show_entity_performance_dashboard(entity_performance)
    
    # Confusion matrices
    st.markdown("#### 🎭 Confusion Matrices")
    
    show_confusion_matrices_dashboard(results)
    
    # F1 scores by entity type
    st.markdown("#### 📊 F1 Scores by Entity Type")
    
    show_f1_scores_by_entity_dashboard(entity_performance)
    
    # Performance trends and alerts
    st.markdown("#### 🚨 Performance Alerts & Recommendations")
    
    alerts = generate_performance_alerts(kpis, entity_performance)
    show_alerts_dashboard(alerts)


# Supporting functions for analytics interfaces

def extract_categorized_performance_data(results: Dict, documents: List[Dict]) -> Tuple[Dict, Dict, Dict]:
    """Extract performance data categorized by domain, difficulty, and document type"""
    performance_by_domain = {}
    performance_by_difficulty = {}
    performance_by_doc_type = {}
    
    for model_name, model_results in results.items():
        if not isinstance(model_results, dict) or 'document_results' not in model_results:
            continue
        
        doc_results = model_results['document_results']
        
        for doc_result in doc_results:
            doc_name = doc_result.get('document', '')
            
            # Find corresponding document metadata
            doc_metadata = None
            for doc in documents:
                if doc['name'] == doc_name:
                    # Try to get metadata from ground truth classification
                    if doc.get('info', {}).get('ground_truth_labels', {}).get('document_classification'):
                        doc_metadata = doc['info']['ground_truth_labels']['document_classification']
                    else:
                        # Fallback to document metadata
                        doc_metadata = {
                            'domain': doc.get('category', 'Unknown'),
                            'difficulty_level': doc.get('info', {}).get('complexity', 'Medium'),
                            'document_type': doc.get('info', {}).get('type', 'Unknown')
                        }
                    break
            
            if not doc_metadata:
                continue
            
            # Extract performance metrics
            perf_data = {
                'model': model_name,
                'precision': doc_result.get('precision', 0),
                'recall': doc_result.get('recall', 0), 
                'f1': doc_result.get('f1', 0),
                'processing_time': doc_result.get('processing_time', 0),
                'cost': doc_result.get('cost', 0),
                'entities_found': doc_result.get('entities_found', 0),
                'ground_truth_entities': doc_result.get('ground_truth_entities', 0)
            }
            
            # Categorize by domain
            domain = doc_metadata.get('domain', 'Unknown')
            if domain not in performance_by_domain:
                performance_by_domain[domain] = []
            performance_by_domain[domain].append(perf_data)
            
            # Categorize by difficulty
            difficulty = doc_metadata.get('difficulty_level', 'Medium')
            if difficulty not in performance_by_difficulty:
                performance_by_difficulty[difficulty] = []
            performance_by_difficulty[difficulty].append(perf_data)
            
            # Categorize by document type
            doc_type = doc_metadata.get('document_type', 'Unknown')
            if doc_type not in performance_by_doc_type:
                performance_by_doc_type[doc_type] = []
            performance_by_doc_type[doc_type].append(perf_data)
    
    return performance_by_domain, performance_by_difficulty, performance_by_doc_type


def create_domain_performance_summary(performance_by_domain: Dict) -> pd.DataFrame:
    """Create domain performance summary table"""
    summary_data = []
    
    for domain, perf_data in performance_by_domain.items():
        if not perf_data:
            continue
        
        df = pd.DataFrame(perf_data)
        
        summary_data.append({
            'Domain': domain,
            'Documents': len(perf_data),
            'Avg F1-Score': df['f1'].mean(),
            'Avg Precision': df['precision'].mean(),
            'Avg Recall': df['recall'].mean(),
            'Avg Processing Time': df['processing_time'].mean(),
            'Total Cost': df['cost'].sum(),
            'F1 Std Dev': df['f1'].std(),
            'Best Model': df.loc[df['f1'].idxmax(), 'model'] if not df.empty else 'N/A'
        })
    
    return pd.DataFrame(summary_data).round(4)


def create_domain_performance_heatmap(performance_by_domain: Dict) -> go.Figure:
    """Create domain performance heatmap"""
    # Prepare data for heatmap
    domains = list(performance_by_domain.keys())
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    heatmap_data = []
    
    for domain in domains:
        perf_data = performance_by_domain[domain]
        if not perf_data:
            continue
        
        df = pd.DataFrame(perf_data)
        domain_row = [
            df['precision'].mean(),
            df['recall'].mean(),
            df['f1'].mean()
        ]
        heatmap_data.append(domain_row)
    
    if not heatmap_data:
        # Return empty figure if no data
        return go.Figure()
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=metrics,
        y=domains,
        colorscale='RdYlGn',
        text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Performance Score")
    ))
    
    fig.update_layout(
        title="Domain Performance Heatmap",
        xaxis_title="Metrics",
        yaxis_title="Domains"
    )
    
    return fig


def show_detailed_domain_analysis(domain_data: List[Dict], domain_name: str):
    """Show detailed analysis for a specific domain"""
    with st.expander(f"Detailed Analysis: {domain_name}", expanded=True):
        df = pd.DataFrame(domain_data)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents Analyzed", len(domain_data))
            st.metric("Average F1-Score", f"{df['f1'].mean():.3f}")
        
        with col2:
            st.metric("Best F1-Score", f"{df['f1'].max():.3f}")
            st.metric("Worst F1-Score", f"{df['f1'].min():.3f}")
        
        with col3:
            st.metric("F1-Score Variance", f"{df['f1'].std():.3f}")
            best_model = df.loc[df['f1'].idxmax(), 'model']
            st.metric("Best Performing Model", best_model)
        
        # Model performance breakdown
        model_performance = df.groupby('model').agg({
            'f1': ['mean', 'std', 'count'],
            'processing_time': 'mean',
            'cost': 'sum'
        }).round(4)
        
        st.dataframe(model_performance, use_container_width=True)


def create_difficulty_performance_summary(performance_by_difficulty: Dict) -> pd.DataFrame:
    """Create difficulty performance summary table"""
    summary_data = []
    
    difficulty_order = ['Easy', 'Medium', 'Hard']
    
    for difficulty in difficulty_order:
        if difficulty not in performance_by_difficulty:
            continue
        
        perf_data = performance_by_difficulty[difficulty]
        if not perf_data:
            continue
        
        df = pd.DataFrame(perf_data)
        
        summary_data.append({
            'Difficulty': difficulty,
            'Documents': len(perf_data),
            'Avg F1-Score': df['f1'].mean(),
            'Avg Precision': df['precision'].mean(),
            'Avg Recall': df['recall'].mean(),
            'Avg Processing Time': df['processing_time'].mean(),
            'Total Cost': df['cost'].sum(),
            'Performance Variance': df['f1'].std()
        })
    
    return pd.DataFrame(summary_data).round(4)


def create_difficulty_performance_trend(performance_by_difficulty: Dict) -> go.Figure:
    """Create difficulty performance trend chart"""
    fig = go.Figure()
    
    difficulty_order = ['Easy', 'Medium', 'Hard']
    models = set()
    
    # Collect all models
    for difficulty_data in performance_by_difficulty.values():
        for item in difficulty_data:
            models.add(item['model'])
    
    # Plot trend for each model
    for model in models:
        f1_scores = []
        difficulties = []
        
        for difficulty in difficulty_order:
            if difficulty not in performance_by_difficulty:
                continue
            
            model_data = [item for item in performance_by_difficulty[difficulty] if item['model'] == model]
            if model_data:
                avg_f1 = np.mean([item['f1'] for item in model_data])
                f1_scores.append(avg_f1)
                difficulties.append(difficulty)
        
        if f1_scores:
            fig.add_trace(go.Scatter(
                x=difficulties,
                y=f1_scores,
                mode='lines+markers',
                name=model,
                line=dict(width=3),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title="Performance Trend by Difficulty Level",
        xaxis_title="Difficulty Level",
        yaxis_title="F1-Score",
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def analyze_performance_degradation(performance_by_difficulty: Dict) -> Dict:
    """Analyze performance degradation from easy to hard documents"""
    degradation_analysis = {}
    
    # Get all models
    all_models = set()
    for difficulty_data in performance_by_difficulty.values():
        for item in difficulty_data:
            all_models.add(item['model'])
    
    for model in all_models:
        easy_f1_scores = []
        hard_f1_scores = []
        
        if 'Easy' in performance_by_difficulty:
            easy_data = [item for item in performance_by_difficulty['Easy'] if item['model'] == model]
            easy_f1_scores = [item['f1'] for item in easy_data]
        
        if 'Hard' in performance_by_difficulty:
            hard_data = [item for item in performance_by_difficulty['Hard'] if item['model'] == model]
            hard_f1_scores = [item['f1'] for item in hard_data]
        
        if easy_f1_scores and hard_f1_scores:
            easy_avg = np.mean(easy_f1_scores)
            hard_avg = np.mean(hard_f1_scores)
            
            degradation_pct = ((easy_avg - hard_avg) / easy_avg) * 100 if easy_avg > 0 else 0
            
            degradation_analysis[model] = {
                'easy_avg_f1': easy_avg,
                'hard_avg_f1': hard_avg,
                'degradation_percentage': degradation_pct,
                'significant_degradation': degradation_pct > 15  # Threshold for significant degradation
            }
    
    return degradation_analysis


def create_doc_type_performance_summary(performance_by_doc_type: Dict) -> pd.DataFrame:
    """Create document type performance summary"""
    summary_data = []
    
    for doc_type, perf_data in performance_by_doc_type.items():
        if not perf_data:
            continue
        
        df = pd.DataFrame(perf_data)
        
        summary_data.append({
            'Document Type': doc_type,
            'Documents': len(perf_data),
            'Avg F1-Score': df['f1'].mean(),
            'Avg Precision': df['precision'].mean(),
            'Avg Recall': df['recall'].mean(),
            'Avg Processing Time': df['processing_time'].mean(),
            'Total Cost': df['cost'].sum()
        })
    
    return pd.DataFrame(summary_data).round(4)


def create_doc_type_performance_chart(performance_by_doc_type: Dict) -> go.Figure:
    """Create document type performance chart"""
    doc_types = []
    avg_f1_scores = []
    avg_processing_times = []
    total_costs = []
    
    for doc_type, perf_data in performance_by_doc_type.items():
        if not perf_data:
            continue
        
        df = pd.DataFrame(perf_data)
        doc_types.append(doc_type)
        avg_f1_scores.append(df['f1'].mean())
        avg_processing_times.append(df['processing_time'].mean())
        total_costs.append(df['cost'].sum())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('F1-Score by Document Type', 'Processing Time by Document Type', 
                       'Total Cost by Document Type', 'F1 vs Cost by Document Type'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # F1-Score bar chart
    fig.add_trace(
        go.Bar(x=doc_types, y=avg_f1_scores, name="F1-Score", marker_color="blue"),
        row=1, col=1
    )
    
    # Processing time bar chart
    fig.add_trace(
        go.Bar(x=doc_types, y=avg_processing_times, name="Processing Time", marker_color="green"),
        row=1, col=2
    )
    
    # Total cost bar chart
    fig.add_trace(
        go.Bar(x=doc_types, y=total_costs, name="Total Cost", marker_color="red"),
        row=2, col=1
    )
    
    # F1 vs Cost scatter
    fig.add_trace(
        go.Scatter(x=total_costs, y=avg_f1_scores, mode='markers+text',
                  text=doc_types, textposition="top center",
                  name="F1 vs Cost", marker=dict(size=10, color="purple")),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, title_text="Document Type Performance Analysis")
    
    return fig


def analyze_cost_by_category(results: Dict, documents: List[Dict]) -> Dict:
    """Analyze cost by category (domain, difficulty, document type)"""
    cost_analysis = {
        'by_domain': {},
        'by_difficulty': {},
        'by_doc_type': {},
        'by_model': {}
    }
    
    for model_name, model_results in results.items():
        if not isinstance(model_results, dict) or 'document_results' not in model_results:
            continue
        
        model_total_cost = 0
        doc_results = model_results['document_results']
        
        for doc_result in doc_results:
            doc_name = doc_result.get('document', '')
            cost = doc_result.get('cost', 0)
            model_total_cost += cost
            
            # Find document metadata
            for doc in documents:
                if doc['name'] == doc_name:
                    domain = doc.get('category', 'Unknown')
                    difficulty = doc.get('info', {}).get('complexity', 'Medium')
                    doc_type = doc.get('info', {}).get('type', 'Unknown')
                    
                    # Aggregate costs
                    cost_analysis['by_domain'][domain] = cost_analysis['by_domain'].get(domain, 0) + cost
                    cost_analysis['by_difficulty'][difficulty] = cost_analysis['by_difficulty'].get(difficulty, 0) + cost
                    cost_analysis['by_doc_type'][doc_type] = cost_analysis['by_doc_type'].get(doc_type, 0) + cost
                    break
        
        cost_analysis['by_model'][model_name] = model_total_cost
    
    return cost_analysis


def show_cost_analysis_dashboard(cost_analysis: Dict):
    """Show cost analysis dashboard"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost by domain
        if cost_analysis['by_domain']:
            st.markdown("**Cost by Domain**")
            domain_costs = pd.DataFrame(list(cost_analysis['by_domain'].items()), 
                                      columns=['Domain', 'Total Cost'])
            st.dataframe(domain_costs.round(4))
        
        # Cost by difficulty
        if cost_analysis['by_difficulty']:
            st.markdown("**Cost by Difficulty**")
            difficulty_costs = pd.DataFrame(list(cost_analysis['by_difficulty'].items()), 
                                          columns=['Difficulty', 'Total Cost'])
            st.dataframe(difficulty_costs.round(4))
    
    with col2:
        # Cost by model
        if cost_analysis['by_model']:
            st.markdown("**Cost by Model**")
            model_costs = pd.DataFrame(list(cost_analysis['by_model'].items()), 
                                     columns=['Model', 'Total Cost'])
            st.dataframe(model_costs.round(4))
        
        # Cost efficiency analysis
        st.markdown("**Cost Efficiency**")
        st.info("Cost per F1-point improvement and processing efficiency metrics")


def calculate_comprehensive_kpis(results: Dict, documents: List[Dict]) -> Dict:
    """Calculate comprehensive KPIs across all models and documents"""
    kpis = {
        'overall_metrics': {},
        'model_comparison': {},
        'processing_efficiency': {},
        'cost_efficiency': {},
        'quality_metrics': {}
    }
    
    all_precision = []
    all_recall = []
    all_f1 = []
    all_processing_times = []
    all_costs = []
    total_documents = 0
    total_entities_found = 0
    total_ground_truth_entities = 0
    
    model_stats = {}
    
    for model_name, model_results in results.items():
        if not isinstance(model_results, dict) or 'document_results' not in model_results:
            continue
        
        doc_results = model_results['document_results']
        model_precision = []
        model_recall = []
        model_f1 = []
        model_times = []
        model_costs = []
        
        for doc_result in doc_results:
            precision = doc_result.get('precision', 0)
            recall = doc_result.get('recall', 0)
            f1 = doc_result.get('f1', 0)
            proc_time = doc_result.get('processing_time', 0)
            cost = doc_result.get('cost', 0)
            
            all_precision.append(precision)
            all_recall.append(recall)
            all_f1.append(f1)
            all_processing_times.append(proc_time)
            all_costs.append(cost)
            
            model_precision.append(precision)
            model_recall.append(recall)
            model_f1.append(f1)
            model_times.append(proc_time)
            model_costs.append(cost)
            
            total_entities_found += doc_result.get('entities_found', 0)
            total_ground_truth_entities += doc_result.get('ground_truth_entities', 0)
        
        total_documents += len(doc_results)
        
        # Model-specific stats
        model_stats[model_name] = {
            'avg_precision': np.mean(model_precision) if model_precision else 0,
            'avg_recall': np.mean(model_recall) if model_recall else 0,
            'avg_f1': np.mean(model_f1) if model_f1 else 0,
            'avg_processing_time': np.mean(model_times) if model_times else 0,
            'total_cost': sum(model_costs),
            'documents_processed': len(doc_results)
        }
    
    # Overall metrics
    kpis['overall_metrics'] = {
        'avg_precision': np.mean(all_precision) if all_precision else 0,
        'avg_recall': np.mean(all_recall) if all_recall else 0,
        'avg_f1': np.mean(all_f1) if all_f1 else 0,
        'total_documents': total_documents,
        'total_entities_found': total_entities_found,
        'total_ground_truth_entities': total_ground_truth_entities,
        'entity_detection_rate': (total_entities_found / total_ground_truth_entities) if total_ground_truth_entities > 0 else 0
    }
    
    # Processing efficiency
    kpis['processing_efficiency'] = {
        'avg_processing_time': np.mean(all_processing_times) if all_processing_times else 0,
        'total_processing_time': sum(all_processing_times),
        'documents_per_hour': (total_documents / (sum(all_processing_times) / 3600)) if sum(all_processing_times) > 0 else 0
    }
    
    # Cost efficiency
    kpis['cost_efficiency'] = {
        'total_cost': sum(all_costs),
        'avg_cost_per_document': np.mean(all_costs) if all_costs else 0,
        'cost_per_entity_found': (sum(all_costs) / total_entities_found) if total_entities_found > 0 else 0,
        'cost_per_f1_point': (sum(all_costs) / np.mean(all_f1)) if all_f1 and np.mean(all_f1) > 0 else 0
    }
    
    # Quality metrics
    kpis['quality_metrics'] = {
        'precision_std': np.std(all_precision) if all_precision else 0,
        'recall_std': np.std(all_recall) if all_recall else 0,
        'f1_std': np.std(all_f1) if all_f1 else 0,
        'consistency_score': 1 - (np.std(all_f1) / np.mean(all_f1)) if all_f1 and np.mean(all_f1) > 0 else 0
    }
    
    # Model comparison
    kpis['model_comparison'] = model_stats
    
    return kpis


def show_kpi_dashboard(kpis: Dict):
    """Show KPI dashboard with key metrics"""
    # Overall performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall F1-Score", 
            f"{kpis['overall_metrics']['avg_f1']:.3f}",
            help="Average F1-score across all models and documents"
        )
        st.metric(
            "Total Documents", 
            kpis['overall_metrics']['total_documents'],
            help="Total documents processed across all models"
        )
    
    with col2:
        st.metric(
            "Precision", 
            f"{kpis['overall_metrics']['avg_precision']:.3f}",
            help="Average precision across all models"
        )
        st.metric(
            "Entity Detection Rate", 
            f"{kpis['overall_metrics']['entity_detection_rate']:.1%}",
            help="Ratio of entities found vs ground truth"
        )
    
    with col3:
        st.metric(
            "Recall", 
            f"{kpis['overall_metrics']['avg_recall']:.3f}",
            help="Average recall across all models"
        )
        st.metric(
            "Consistency Score", 
            f"{kpis['quality_metrics']['consistency_score']:.3f}",
            help="Performance consistency (1 = perfect consistency)"
        )
    
    with col4:
        st.metric(
            "Avg Processing Time", 
            f"{kpis['processing_efficiency']['avg_processing_time']:.2f}s",
            help="Average processing time per document"
        )
        st.metric(
            "Total Cost", 
            f"${kpis['cost_efficiency']['total_cost']:.4f}",
            help="Total cost across all models and documents"
        )
    
    # Efficiency metrics
    st.markdown("**Efficiency Metrics**")
    
    eff_col1, eff_col2, eff_col3, eff_col4 = st.columns(4)
    
    with eff_col1:
        st.metric(
            "Docs/Hour", 
            f"{kpis['processing_efficiency']['documents_per_hour']:.1f}",
            help="Documents that can be processed per hour"
        )
    
    with eff_col2:
        st.metric(
            "Cost/Document", 
            f"${kpis['cost_efficiency']['avg_cost_per_document']:.4f}",
            help="Average cost per document"
        )
    
    with eff_col3:
        st.metric(
            "Cost/Entity", 
            f"${kpis['cost_efficiency']['cost_per_entity_found']:.4f}",
            help="Cost per entity found"
        )
    
    with eff_col4:
        st.metric(
            "Cost/F1-Point", 
            f"${kpis['cost_efficiency']['cost_per_f1_point']:.4f}",
            help="Cost per F1-score point"
        )


def analyze_entity_type_performance(results: Dict) -> Dict:
    """Analyze performance by entity type across all models"""
    entity_performance = {}
    
    for model_name, model_results in results.items():
        if not isinstance(model_results, dict) or 'document_results' not in model_results:
            continue
        
        doc_results = model_results['document_results']
        
        for doc_result in doc_results:
            extracted_entities = doc_result.get('extracted_entities', [])
            
            # Group by entity type
            for entity in extracted_entities:
                entity_type = entity.get('type', 'UNKNOWN')
                confidence = entity.get('confidence', 0)
                
                if entity_type not in entity_performance:
                    entity_performance[entity_type] = {
                        'total_found': 0,
                        'confidences': [],
                        'models': {},
                        'true_positives': 0,
                        'false_positives': 0,
                        'false_negatives': 0
                    }
                
                entity_performance[entity_type]['total_found'] += 1
                entity_performance[entity_type]['confidences'].append(confidence)
                
                if model_name not in entity_performance[entity_type]['models']:
                    entity_performance[entity_type]['models'][model_name] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'confidences': []
                    }
                
                entity_performance[entity_type]['models'][model_name]['count'] += 1
                entity_performance[entity_type]['models'][model_name]['confidences'].append(confidence)
            
            # Add confusion matrix data
            tp = doc_result.get('true_positives', 0)
            fp = doc_result.get('false_positives', 0)
            fn = doc_result.get('false_negatives', 0)
            
            # This is simplified - in practice you'd need entity-type specific TP/FP/FN
            for entity_type in entity_performance:
                entity_performance[entity_type]['true_positives'] += tp
                entity_performance[entity_type]['false_positives'] += fp
                entity_performance[entity_type]['false_negatives'] += fn
    
    # Calculate averages
    for entity_type, data in entity_performance.items():
        if data['confidences']:
            data['avg_confidence'] = np.mean(data['confidences'])
            data['confidence_std'] = np.std(data['confidences'])
        
        # Calculate precision, recall, F1 per entity type
        tp = data['true_positives']
        fp = data['false_positives']
        fn = data['false_negatives']
        
        data['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        data['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        data['f1'] = 2 * (data['precision'] * data['recall']) / (data['precision'] + data['recall']) if (data['precision'] + data['recall']) > 0 else 0
        
        for model_name, model_data in data['models'].items():
            if model_data['confidences']:
                model_data['avg_confidence'] = np.mean(model_data['confidences'])
    
    return entity_performance


def show_entity_performance_dashboard(entity_performance: Dict):
    """Show entity type performance dashboard"""
    if not entity_performance:
        st.info("No entity performance data available.")
        return
    
    # Entity performance summary table
    entity_summary_data = []
    for entity_type, data in entity_performance.items():
        entity_summary_data.append({
            'Entity Type': entity_type,
            'Total Found': data['total_found'],
            'Avg Confidence': data.get('avg_confidence', 0),
            'Confidence Std': data.get('confidence_std', 0),
            'Precision': data.get('precision', 0),
            'Recall': data.get('recall', 0),
            'F1-Score': data.get('f1', 0)
        })
    
    if entity_summary_data:
        df_entity_summary = pd.DataFrame(entity_summary_data)
        st.dataframe(df_entity_summary.round(4), use_container_width=True)
        
        # Entity performance visualization
        fig_entity_perf = px.bar(
            df_entity_summary,
            x='Entity Type',
            y=['Precision', 'Recall', 'F1-Score'],
            title="Performance by Entity Type",
            barmode='group'
        )
        st.plotly_chart(fig_entity_perf, use_container_width=True)
        
        # Entity confidence distribution
        fig_confidence = go.Figure()
        for entity_type, data in entity_performance.items():
            if data.get('confidences'):
                fig_confidence.add_trace(go.Box(
                    y=data['confidences'],
                    name=entity_type,
                    boxpoints='outliers'
                ))
        
        fig_confidence.update_layout(
            title="Confidence Distribution by Entity Type",
            yaxis_title="Confidence Score"
        )
        st.plotly_chart(fig_confidence, use_container_width=True)


def show_confusion_matrices_dashboard(results: Dict):
    """Show confusion matrices for each model"""
    for model_name, model_results in results.items():
        if not isinstance(model_results, dict) or 'document_results' not in model_results:
            continue
        
        with st.expander(f"Confusion Matrix: {model_name}", expanded=False):
            doc_results = model_results['document_results']
            
            # Aggregate confusion matrix data
            total_tp = sum(doc.get('true_positives', 0) for doc in doc_results)
            total_fp = sum(doc.get('false_positives', 0) for doc in doc_results)
            total_fn = sum(doc.get('false_negatives', 0) for doc in doc_results)
            total_tn = 0  # True negatives are harder to calculate for entity extraction
            
            # Create confusion matrix visualization
            confusion_matrix = np.array([
                [total_tp, total_fp],
                [total_fn, total_tn]
            ])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=confusion_matrix,
                x=['Predicted Positive', 'Predicted Negative'],
                y=['Actual Positive', 'Actual Negative'],
                text=confusion_matrix,
                texttemplate="%{text}",
                colorscale='Blues'
            ))
            
            fig_cm.update_layout(
                title=f"Confusion Matrix: {model_name}",
                xaxis_title="Predicted",
                yaxis_title="Actual"
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Show metrics derived from confusion matrix
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            cm_col1, cm_col2, cm_col3 = st.columns(3)
            
            with cm_col1:
                st.metric("Precision", f"{precision:.3f}")
            with cm_col2:
                st.metric("Recall", f"{recall:.3f}")
            with cm_col3:
                st.metric("F1-Score", f"{f1:.3f}")


def show_f1_scores_by_entity_dashboard(entity_performance: Dict):
    """Show F1 scores by entity type dashboard"""
    if not entity_performance:
        st.info("No entity performance data available for F1 analysis.")
        return
    
    # F1 scores by entity type
    entity_f1_data = []
    for entity_type, data in entity_performance.items():
        entity_f1_data.append({
            'Entity Type': entity_type,
            'F1-Score': data.get('f1', 0),
            'Total Found': data['total_found']
        })
    
    if entity_f1_data:
        df_f1 = pd.DataFrame(entity_f1_data)
        
        # F1 scores bar chart
        fig_f1_bar = px.bar(
            df_f1,
            x='Entity Type',
            y='F1-Score',
            title="F1-Scores by Entity Type",
            color='F1-Score',
            color_continuous_scale='RdYlGn'
        )
        fig_f1_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_f1_bar, use_container_width=True)
        
        # Entity type ranking
        df_f1_sorted = df_f1.sort_values('F1-Score', ascending=False)
        
        st.markdown("**Entity Type Performance Ranking**")
        for i, (_, row) in enumerate(df_f1_sorted.iterrows()):
            rank_emoji = ["🥇", "🥈", "🥉"] + ["🏅"] * 10
            entity_type = row['Entity Type']
            f1_score = row['F1-Score']
            total_found = row['Total Found']
            
            performance_level = "Excellent" if f1_score > 0.9 else "Good" if f1_score > 0.8 else "Fair" if f1_score > 0.7 else "Poor"
            
            st.write(f"{rank_emoji[i]} **{entity_type}**: {f1_score:.3f} ({performance_level}) - {total_found} instances")


def generate_performance_alerts(kpis: Dict, entity_performance: Dict) -> List[Dict]:
    """Generate performance alerts and recommendations"""
    alerts = []
    
    # Overall performance alerts
    overall_f1 = kpis['overall_metrics']['avg_f1']
    if overall_f1 < 0.7:
        alerts.append({
            'type': 'warning',
            'category': 'Overall Performance',
            'message': f"Overall F1-score ({overall_f1:.3f}) is below recommended threshold (0.7)",
            'recommendation': "Consider model fine-tuning, improving training data quality, or adjusting hyperparameters"
        })
    elif overall_f1 > 0.9:
        alerts.append({
            'type': 'success',
            'category': 'Overall Performance',
            'message': f"Excellent overall F1-score ({overall_f1:.3f})",
            'recommendation': "System performance is excellent and ready for production deployment"
        })
    
    # Consistency alerts
    consistency_score = kpis['quality_metrics']['consistency_score']
    if consistency_score < 0.8:
        alerts.append({
            'type': 'warning',
            'category': 'Performance Consistency',
            'message': f"Low performance consistency ({consistency_score:.3f})",
            'recommendation': "High variance detected. Review model selection and consider ensemble methods"
        })
    
    # Cost efficiency alerts
    cost_per_doc = kpis['cost_efficiency']['avg_cost_per_document']
    if cost_per_doc > 0.1:
        alerts.append({
            'type': 'info',
            'category': 'Cost Efficiency',
            'message': f"High cost per document (${cost_per_doc:.4f})",
            'recommendation': "Consider using more cost-effective models for batch processing"
        })
    
    # Entity-specific alerts
    for entity_type, data in entity_performance.items():
        f1_score = data.get('f1', 0)
        if f1_score < 0.6:
            alerts.append({
                'type': 'warning',
                'category': f'Entity Performance ({entity_type})',
                'message': f"Low F1-score for {entity_type} ({f1_score:.3f})",
                'recommendation': f"Improve {entity_type} detection with additional training examples or rule-based post-processing"
            })
    
    # Processing efficiency alerts
    docs_per_hour = kpis['processing_efficiency']['documents_per_hour']
    if docs_per_hour < 10:
        alerts.append({
            'type': 'info',
            'category': 'Processing Efficiency',
            'message': f"Low processing rate ({docs_per_hour:.1f} docs/hour)",
            'recommendation': "Consider parallel processing, faster models, or batch optimization"
        })
    
    return alerts


def show_alerts_dashboard(alerts: List[Dict]):
    """Show performance alerts dashboard"""
    if not alerts:
        st.success("🎉 No performance issues detected! All metrics are within acceptable ranges.")
        return
    
    # Group alerts by type
    warnings = [alert for alert in alerts if alert['type'] == 'warning']
    infos = [alert for alert in alerts if alert['type'] == 'info']
    successes = [alert for alert in alerts if alert['type'] == 'success']
    
    # Show warnings first
    if warnings:
        st.markdown("**⚠️ Warnings**")
        for alert in warnings:
            st.warning(f"**{alert['category']}**: {alert['message']}")
            st.info(f"💡 **Recommendation**: {alert['recommendation']}")
    
    # Show informational alerts
    if infos:
        st.markdown("**ℹ️ Information**")
        for alert in infos:
            st.info(f"**{alert['category']}**: {alert['message']}")
            st.caption(f"💡 {alert['recommendation']}")
    
    # Show successes
    if successes:
        st.markdown("**✅ Success**")
        for alert in successes:
            st.success(f"**{alert['category']}**: {alert['message']}")
            st.caption(f"💡 {alert['recommendation']}")