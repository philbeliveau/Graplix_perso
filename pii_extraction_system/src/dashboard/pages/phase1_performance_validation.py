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


def show_page():
    """Main Phase 1 Performance Validation page"""
    st.markdown('<div class="section-header">🎯 Phase 1: Cross-Domain Performance Validation</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Phase 1 Objectives:**
    - Document selection interface (10+ diverse documents)
    - Multi-model testing panel (GPT-4o, GPT-4o-mini, other models)
    - Real-time PII extraction comparison
    - Performance variance calculator (target: <10%)
    - Baseline comparison (OCR + spaCy NER vs LLM approaches)
    """)
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Initialize session state for Phase 1
    initialize_phase1_session_state()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Document Selection",
        "🤖 Multi-Model Testing",
        "⚡ Real-time Comparison",
        "📊 Variance Analysis",
        "🔍 Baseline Comparison"
    ])
    
    with tab1:
        show_document_selection_interface()
    
    with tab2:
        show_multi_model_testing_panel()
    
    with tab3:
        show_realtime_comparison_dashboard()
    
    with tab4:
        show_variance_calculator()
    
    with tab5:
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


def show_document_selection_interface():
    """Document selection interface with 10+ diverse documents"""
    st.markdown("### 📄 Document Selection Interface")
    st.markdown("Select diverse documents for cross-domain validation testing.")
    
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
        preferred_defaults = ["gpt-4o-mini", "gpt-4o", "claude-3-haiku"]
        available_options = list(available_models.keys())
        valid_defaults = [model for model in preferred_defaults if model in available_options]
        
        selected_models = st.multiselect(
            "Select LLM models for testing:",
            options=available_options,
            default=valid_defaults[:3] if valid_defaults else available_options[:3],
            help="Choose multiple models for comparison"
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
    """Get available LLM models for testing"""
    try:
        # Import LLM configuration
        from llm.llm_config import LLMModelRegistry
        
        models = {}
        for model_name, model_config in LLMModelRegistry.MODELS.items():
            models[model_name] = {
                'display_name': model_config.display_name,
                'provider': model_config.provider.value,
                'supports_vision': model_config.supports_vision,
                'quality_score': model_config.quality_score,
                'speed_score': model_config.speed_score,
                'input_cost': model_config.input_cost_per_1k_tokens,
                'output_cost': model_config.output_cost_per_1k_tokens,
                'description': model_config.description
            }
        return models
    except ImportError:
        # Fallback to mock models if import fails
        return {
            "gpt-4o": {
                'display_name': "GPT-4o",
                'provider': "openai",
                'supports_vision': True,
                'quality_score': 0.95,
                'speed_score': 0.8,
                'input_cost': 0.005,
                'output_cost': 0.015,
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
            
            # Simulate model testing
            model_results = simulate_model_testing(model_name, st.session_state.phase1_selected_documents, threshold)
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


def simulate_model_testing(model_name: str, documents: List[Dict], threshold: float) -> Dict:
    """Simulate model testing and return results"""
    available_models = get_available_llm_models()
    model_info = available_models.get(model_name, {})
    
    # Base performance varies by model
    base_performance = {
        "gpt-4o": {"precision": 0.92, "recall": 0.89, "f1": 0.905},
        "gpt-4o-mini": {"precision": 0.88, "recall": 0.85, "f1": 0.865},
        "claude-3-haiku": {"precision": 0.82, "recall": 0.80, "f1": 0.81},
        "claude-3-sonnet": {"precision": 0.90, "recall": 0.87, "f1": 0.885},
        "gemini-1.5-flash": {"precision": 0.83, "recall": 0.81, "f1": 0.82}
    }
    
    # Get base performance or default
    base_perf = base_performance.get(model_name, {"precision": 0.80, "recall": 0.78, "f1": 0.79})
    
    # Add some realistic variance
    np.random.seed(hash(model_name) % 1000)  # Deterministic but different per model
    noise_factor = 0.03
    
    precision = max(0.6, min(0.98, base_perf["precision"] + np.random.normal(0, noise_factor)))
    recall = max(0.6, min(0.98, base_perf["recall"] + np.random.normal(0, noise_factor)))
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Processing time varies by model
    base_times = {
        "gpt-4o": 3.2,
        "gpt-4o-mini": 1.8,
        "claude-3-haiku": 1.2,
        "claude-3-sonnet": 2.5,
        "gemini-1.5-flash": 1.5
    }
    
    avg_processing_time = base_times.get(model_name, 2.0) + np.random.normal(0, 0.3)
    
    # Cost calculation
    total_cost = estimate_testing_cost([model_name], documents)
    
    return {
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_processing_time": abs(avg_processing_time),
            "total_cost": total_cost / len([model_name])  # Cost for this model only
        },
        "document_results": [
            {
                "document": doc["name"],
                "processing_time": abs(avg_processing_time + np.random.normal(0, 0.5)),
                "entities_found": np.random.randint(3, 12),
                "confidence_scores": [np.random.uniform(threshold, 0.98) for _ in range(np.random.randint(3, 12))]
            }
            for doc in documents
        ]
    }


def show_realtime_comparison_dashboard():
    """Real-time PII extraction comparison dashboard"""
    st.markdown("### ⚡ Real-time PII Extraction Comparison")
    st.markdown("Compare PII extraction results across multiple models in real-time.")
    
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