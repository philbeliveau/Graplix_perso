"""
Model Comparison Page - Compare different PII extraction strategies

This page enables systematic evaluation of different PII extraction strategies,
supporting data-driven optimization of system performance and accuracy.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import session_state, ui_components, auth

def show_page():
    """Main model comparison page"""
    st.markdown('<div class="section-header">🔬 Model Comparison</div>', 
                unsafe_allow_html=True)
    st.markdown("Compare different PII extraction strategies side-by-side.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Layout
    show_model_selection()
    
    if st.session_state.get('selected_models'):
        show_comparison_interface()
    else:
        st.info("Select models to compare from the sidebar.")

def show_model_selection():
    """Model selection interface"""
    with st.sidebar:
        st.markdown("### Model Selection")
        
        available_models = get_available_models()
        
        selected_models = st.multiselect(
            "Choose models to compare:",
            options=list(available_models.keys()),
            default=st.session_state.get('selected_models', []),
            help="Select 2-4 models for comparison"
        )
        
        # Update session state
        st.session_state.selected_models = selected_models
        
        # Configuration options
        st.markdown("### Comparison Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('confidence_threshold', 0.5),
            step=0.05
        )
        st.session_state.confidence_threshold = confidence_threshold
        
        evaluation_metric = st.selectbox(
            "Primary Evaluation Metric",
            ['F1-Score', 'Precision', 'Recall', 'Accuracy'],
            index=0
        )
        
        if st.button("Run Comparison", disabled=len(selected_models) < 2):
            run_model_comparison(selected_models, confidence_threshold, evaluation_metric)

def get_available_models() -> Dict[str, Dict]:
    """Get available PII extraction models"""
    return {
        'Rule-Based Extractor': {
            'type': 'rule_based',
            'description': 'Regex and dictionary-based extraction',
            'strengths': ['Fast', 'Deterministic', 'No training required'],
            'weaknesses': ['Limited recall', 'Language specific']
        },
        'spaCy NER': {
            'type': 'spacy_ner',
            'description': 'Pre-trained spaCy named entity recognition',
            'strengths': ['Good accuracy', 'Fast inference', 'Multi-language'],
            'weaknesses': ['General purpose', 'May miss domain-specific entities']
        },
        'Transformers NER': {
            'type': 'transformers_ner',
            'description': 'BERT-based NER from Hugging Face',
            'strengths': ['High accuracy', 'Context awareness', 'Fine-tunable'],
            'weaknesses': ['Slower inference', 'Resource intensive']
        },
        'LayoutLM': {
            'type': 'layoutlm',
            'description': 'Layout-aware document understanding',
            'strengths': ['Document structure awareness', 'High accuracy on forms'],
            'weaknesses': ['Complex setup', 'Requires visual features']
        },
        'Custom Fine-tuned': {
            'type': 'custom_finetuned',
            'description': 'Domain-specific fine-tuned model',
            'strengths': ['Domain optimized', 'High performance'],
            'weaknesses': ['Requires training data', 'Domain specific']
        },
        'Ensemble Method': {
            'type': 'ensemble',
            'description': 'Combination of multiple models',
            'strengths': ['Best of all models', 'Robust performance'],
            'weaknesses': ['Slower inference', 'Complex to maintain']
        }
    }

def show_comparison_interface():
    """Main comparison interface"""
    selected_models = st.session_state.get('selected_models', [])
    
    if len(selected_models) < 2:
        st.warning("Please select at least 2 models for comparison.")
        return
    
    # Tabs for different comparison views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Metrics", 
        "📄 Side-by-Side Results", 
        "📈 Statistical Analysis",
        "🔧 Configuration Comparison"
    ])
    
    with tab1:
        show_performance_metrics()
    
    with tab2:
        show_side_by_side_results()
    
    with tab3:
        show_statistical_analysis()
    
    with tab4:
        show_configuration_comparison()

def run_model_comparison(models: List[str], threshold: float, metric: str):
    """Run comparison between selected models"""
    with st.spinner("Running model comparison..."):
        # Mock comparison results (would integrate with actual models)
        comparison_results = generate_mock_comparison_results(models, threshold)
        
        # Store results in session state
        st.session_state.model_comparison_results = comparison_results
        
        st.success(f"Comparison completed for {len(models)} models!")

def generate_mock_comparison_results(models: List[str], threshold: float) -> Dict[str, Any]:
    """Generate mock comparison results for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    results = {
        'models': models,
        'threshold': threshold,
        'timestamp': st.session_state.get('current_time', 'now'),
        'performance_metrics': {},
        'entity_level_results': {},
        'processing_times': {},
        'confusion_matrices': {},
        'agreement_analysis': {}
    }
    
    # Generate performance metrics for each model
    base_scores = {
        'Rule-Based Extractor': {'precision': 0.85, 'recall': 0.65, 'f1': 0.73},
        'spaCy NER': {'precision': 0.78, 'recall': 0.82, 'f1': 0.80},
        'Transformers NER': {'precision': 0.92, 'recall': 0.88, 'f1': 0.90},
        'LayoutLM': {'precision': 0.89, 'recall': 0.85, 'f1': 0.87},
        'Custom Fine-tuned': {'precision': 0.94, 'recall': 0.91, 'f1': 0.92},
        'Ensemble Method': {'precision': 0.95, 'recall': 0.93, 'f1': 0.94}
    }
    
    for model in models:
        base = base_scores.get(model, {'precision': 0.75, 'recall': 0.75, 'f1': 0.75})
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.02, 3)
        metrics = {
            'precision': max(0, min(1, base['precision'] + noise[0])),
            'recall': max(0, min(1, base['recall'] + noise[1])),
            'f1': max(0, min(1, base['f1'] + noise[2]))
        }
        metrics['accuracy'] = (metrics['precision'] + metrics['recall']) / 2
        
        results['performance_metrics'][model] = metrics
        results['processing_times'][model] = np.random.uniform(0.5, 5.0)  # seconds
        
        # Generate entity-level results
        pii_categories = ['PERSON', 'EMAIL', 'PHONE', 'ADDRESS', 'SSN']
        entity_results = {}
        for category in pii_categories:
            entity_results[category] = {
                'true_positives': np.random.randint(5, 25),
                'false_positives': np.random.randint(0, 8),
                'false_negatives': np.random.randint(0, 10)
            }
        results['entity_level_results'][model] = entity_results
    
    return results

def show_performance_metrics():
    """Display performance metrics comparison"""
    st.markdown("### Performance Metrics Comparison")
    
    comparison_results = st.session_state.get('model_comparison_results', {})
    if not comparison_results:
        st.info("Run a comparison to see performance metrics.")
        return
    
    metrics_data = comparison_results.get('performance_metrics', {})
    if not metrics_data:
        st.warning("No performance data available.")
        return
    
    # Create comparison table
    df_metrics = pd.DataFrame(metrics_data).T
    df_metrics = df_metrics.round(3)
    
    # Format as percentages
    for col in df_metrics.columns:
        df_metrics[col] = df_metrics[col].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(df_metrics, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance radar chart
        raw_metrics = comparison_results.get('performance_metrics', {})
        if raw_metrics:
            model_name = st.selectbox("Select model for radar chart:", list(raw_metrics.keys()))
            if model_name:
                fig = ui_components.create_performance_metrics_chart(raw_metrics[model_name])
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Processing time comparison
        processing_times = comparison_results.get('processing_times', {})
        if processing_times:
            import plotly.express as px
            
            df_times = pd.DataFrame(list(processing_times.items()), 
                                  columns=['Model', 'Processing Time (s)'])
            fig = px.bar(df_times, x='Model', y='Processing Time (s)',
                        title='Processing Time Comparison')
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

def show_side_by_side_results():
    """Show side-by-side extraction results"""
    st.markdown("### Side-by-Side Results")
    
    # Select a document for comparison
    uploaded_docs = st.session_state.get('uploaded_documents', {})
    if not uploaded_docs:
        st.info("Upload documents in the Document Processing section to compare results.")
        return
    
    doc_options = {doc_id: info['name'] for doc_id, info in uploaded_docs.items()}
    selected_doc = st.selectbox("Select document for comparison:", 
                               options=list(doc_options.keys()),
                               format_func=lambda x: doc_options[x])
    
    if not selected_doc:
        return
    
    # Get processing results
    doc_results = session_state.get_processing_results(selected_doc)
    if not doc_results:
        st.warning("No processing results found. Process the document first.")
        return
    
    selected_models = st.session_state.get('selected_models', [])
    
    # Create columns for each model
    if len(selected_models) <= 2:
        cols = st.columns(len(selected_models))
    else:
        # Use tabs for more than 2 models
        tabs = st.tabs(selected_models)
        cols = tabs
    
    text_content = doc_results.get('text_content', '')
    
    for i, model_name in enumerate(selected_models):
        with cols[i]:
            st.markdown(f"**{model_name}**")
            
            # Generate mock results for this model
            mock_entities = generate_mock_entities_for_model(model_name, text_content)
            
            # Show highlighted text
            if mock_entities:
                highlighted_html = ui_components.display_pii_highlights(text_content, mock_entities)
                st.markdown(f'<div style="height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;">{highlighted_html}</div>', 
                           unsafe_allow_html=True)
            else:
                st.text_area(f"Text ({model_name})", text_content, height=300, disabled=True)
            
            # Entity count
            entity_counts = {}
            for entity in mock_entities:
                entity_type = entity.get('type', 'UNKNOWN')
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            if entity_counts:
                st.markdown("**Detected PII:**")
                for pii_type, count in entity_counts.items():
                    st.markdown(f"- {pii_type}: {count}")

def generate_mock_entities_for_model(model_name: str, text: str) -> List[Dict]:
    """Generate mock PII entities for a specific model"""
    import re
    
    # Different models have different patterns and accuracies
    base_entities = []
    
    # Email detection (all models should find these)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    for match in re.finditer(email_pattern, text):
        confidence = 0.95 if 'Rule-Based' in model_name else np.random.uniform(0.85, 0.98)
        base_entities.append({
            'type': 'EMAIL',
            'text': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': confidence
        })
    
    # Phone detection (varying accuracy)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    for match in re.finditer(phone_pattern, text):
        if 'Rule-Based' in model_name:
            confidence = 0.90
        elif 'Transformers' in model_name or 'Custom' in model_name:
            confidence = np.random.uniform(0.88, 0.95)
        else:
            confidence = np.random.uniform(0.75, 0.88)
        
        base_entities.append({
            'type': 'PHONE',
            'text': match.group(),
            'start': match.start(),
            'end': match.end(),
            'confidence': confidence
        })
    
    # Name detection (more variation between models)
    name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    for match in re.finditer(name_pattern, text):
        if match.group() not in ['Dear Sir', 'John Doe', 'Jane Doe']:
            if 'Rule-Based' in model_name:
                # Rule-based might miss some names
                if np.random.random() > 0.3:
                    continue
                confidence = 0.60
            elif 'Transformers' in model_name or 'Custom' in model_name:
                confidence = np.random.uniform(0.80, 0.92)
            else:
                confidence = np.random.uniform(0.65, 0.82)
            
            base_entities.append({
                'type': 'PERSON',
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'confidence': confidence
            })
    
    return base_entities

def show_statistical_analysis():
    """Show statistical analysis of model comparison"""
    st.markdown("### Statistical Analysis")
    
    comparison_results = st.session_state.get('model_comparison_results', {})
    if not comparison_results:
        st.info("Run a comparison to see statistical analysis.")
        return
    
    # Model agreement analysis
    st.markdown("#### Model Agreement Analysis")
    
    models = comparison_results.get('models', [])
    if len(models) >= 2:
        # Create agreement matrix
        agreement_data = []
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    # Mock agreement score
                    agreement = np.random.uniform(0.6, 0.9)
                    agreement_data.append({
                        'Model 1': model1,
                        'Model 2': model2,
                        'Agreement': agreement
                    })
        
        if agreement_data:
            df_agreement = pd.DataFrame(agreement_data)
            
            # Pivot for heatmap
            pivot_agreement = df_agreement.pivot(index='Model 1', columns='Model 2', values='Agreement')
            
            import plotly.express as px
            fig = px.imshow(pivot_agreement, 
                           title="Model Agreement Matrix",
                           color_continuous_scale='Blues',
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
    
    # Statistical significance testing
    st.markdown("#### Statistical Significance")
    
    metrics_data = comparison_results.get('performance_metrics', {})
    if len(metrics_data) >= 2:
        st.markdown("**McNemar's Test Results:**")
        
        model_pairs = [(models[i], models[j]) for i in range(len(models)) 
                      for j in range(i+1, len(models))]
        
        for model1, model2 in model_pairs:
            # Mock p-value
            p_value = np.random.uniform(0.01, 0.15)
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"{model1} vs {model2}")
            with col2:
                st.write(f"p = {p_value:.3f}")
            with col3:
                color = "green" if significance == "Significant" else "orange"
                st.markdown(f'<span style="color: {color}">{significance}</span>', 
                           unsafe_allow_html=True)

def show_configuration_comparison():
    """Show model configuration and capabilities comparison"""
    st.markdown("### Model Configuration Comparison")
    
    selected_models = st.session_state.get('selected_models', [])
    available_models = get_available_models()
    
    if not selected_models:
        st.info("Select models to see configuration comparison.")
        return
    
    # Create comparison table
    comparison_data = []
    for model in selected_models:
        model_info = available_models.get(model, {})
        comparison_data.append({
            'Model': model,
            'Type': model_info.get('type', 'unknown'),
            'Description': model_info.get('description', 'No description'),
            'Strengths': ', '.join(model_info.get('strengths', [])),
            'Weaknesses': ', '.join(model_info.get('weaknesses', []))
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Feature comparison matrix
    st.markdown("#### Feature Comparison")
    
    features = [
        'Speed', 'Accuracy', 'Language Support', 'Document Layout',
        'Training Required', 'Resource Usage', 'Customization'
    ]
    
    feature_matrix = []
    for model in selected_models:
        # Mock feature scores
        feature_scores = {}
        for feature in features:
            if 'Rule-Based' in model:
                scores = {'Speed': 5, 'Accuracy': 3, 'Language Support': 2, 
                         'Document Layout': 1, 'Training Required': 5, 
                         'Resource Usage': 5, 'Customization': 3}
            elif 'spaCy' in model:
                scores = {'Speed': 4, 'Accuracy': 4, 'Language Support': 4,
                         'Document Layout': 2, 'Training Required': 4,
                         'Resource Usage': 4, 'Customization': 3}
            elif 'Transformers' in model:
                scores = {'Speed': 2, 'Accuracy': 5, 'Language Support': 5,
                         'Document Layout': 3, 'Training Required': 3,
                         'Resource Usage': 2, 'Customization': 4}
            elif 'LayoutLM' in model:
                scores = {'Speed': 2, 'Accuracy': 5, 'Language Support': 3,
                         'Document Layout': 5, 'Training Required': 2,
                         'Resource Usage': 1, 'Customization': 3}
            else:
                scores = {f: 3 for f in features}  # Default scores
            
            feature_scores[feature] = scores.get(feature, 3)
        
        feature_scores['Model'] = model
        feature_matrix.append(feature_scores)
    
    df_features = pd.DataFrame(feature_matrix)
    df_features = df_features.set_index('Model')
    
    # Create heatmap
    import plotly.express as px
    fig = px.imshow(df_features.T, 
                   title="Feature Comparison Matrix (1=Poor, 5=Excellent)",
                   color_continuous_scale='RdYlGn',
                   aspect="auto")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)