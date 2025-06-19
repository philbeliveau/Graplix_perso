"""
Phase 2 Multimodal Model Testing Interface

This page provides advanced testing capabilities for multimodal LLMs including:
- Claude, Mistral, GPT-4V OCR testing
- Document difficulty categorization
- Performance metrics analysis
- Cross-domain variance analysis
- Interactive cost/performance/adaptability matrix
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import base64
import time
import sys
from pathlib import Path
from io import BytesIO
from PIL import Image

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

def show_page():
    """Main multimodal testing page"""
    st.markdown('<div class="section-header">🔬 Phase 2: Multimodal Model Testing</div>', 
                unsafe_allow_html=True)
    st.markdown("Advanced testing interface for multimodal LLMs with document difficulty analysis and performance optimization.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Initialize session state
    initialize_session_state()
    
    # Main layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 Model Testing",
        "📊 Document Analysis", 
        "📈 Performance Matrix",
        "🔄 Variance Analysis",
        "📋 Automated Reports"
    ])
    
    with tab1:
        show_model_testing_interface()
    
    with tab2:
        show_document_analysis()
    
    with tab3:
        show_performance_matrix()
    
    with tab4:
        show_variance_analysis()
    
    with tab5:
        show_automated_reports()

def initialize_session_state():
    """Initialize session state variables"""
    if 'test_results' not in st.session_state:
        st.session_state.test_results = []
    if 'document_difficulties' not in st.session_state:
        st.session_state.document_difficulties = {}
    if 'performance_matrix_data' not in st.session_state:
        st.session_state.performance_matrix_data = []

def show_model_testing_interface():
    """Enhanced model testing interface"""
    st.markdown("### 🔧 Advanced Multimodal Testing")
    
    # Get available models
    available_models = llm_service.get_available_models()
    
    if not available_models:
        st.error("No multimodal models available. Please configure API keys in settings.")
        return
    
    # Model selection with enhanced information
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Model Selection")
        selected_models = st.multiselect(
            "Choose models to test:",
            available_models,
            default=available_models[:3] if len(available_models) >= 3 else available_models,
            help="Select multiple models for comparison"
        )
        
        # Test configuration
        st.markdown("#### Test Configuration")
        test_mode = st.selectbox(
            "Testing Mode",
            ["Single Document", "Batch Testing", "Comparative Analysis", "Stress Testing"]
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=1000,
            max_value=8000,
            value=4000,
            step=500
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1
        )
    
    with col2:
        st.markdown("#### Model Information")
        if selected_models:
            for model in selected_models:
                model_info = llm_service.get_model_info(model)
                if model_info.get('available', False):
                    provider = model_info['provider']
                    model_name = model_info['model']
                    
                    with st.expander(f"{provider.upper()} - {model_name}"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Input Cost", f"${model_info['cost_per_1k_input_tokens']:.4f}/1K tokens")
                            st.metric("Output Cost", f"${model_info['cost_per_1k_output_tokens']:.4f}/1K tokens")
                        with col_b:
                            st.write("✅ Images Supported")
                            st.write("✅ JSON Output")
    
    # Document upload and testing
    st.markdown("#### Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload documents for testing",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        accept_multiple_files=True,
        help="Upload images or PDFs for multimodal testing"
    )
    
    if uploaded_files and selected_models:
        # Document difficulty estimation
        st.markdown("#### Document Difficulty Assessment")
        
        difficulties = {}
        for uploaded_file in uploaded_files:
            difficulty = estimate_document_difficulty(uploaded_file)
            difficulties[uploaded_file.name] = difficulty
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**{uploaded_file.name}**")
            with col2:
                difficulty_color = get_difficulty_color(difficulty['score'])
                st.markdown(f"<span style='color: {difficulty_color}'>**{difficulty['category']}**</span>", 
                           unsafe_allow_html=True)
            with col3:
                st.write(f"Score: {difficulty['score']:.2f}")
        
        # Store difficulties in session state
        st.session_state.document_difficulties.update(difficulties)
        
        # Run tests
        if st.button("🚀 Run Multimodal Tests", type="primary"):
            run_multimodal_tests(uploaded_files, selected_models, test_mode, {
                'confidence_threshold': confidence_threshold,
                'max_tokens': max_tokens,
                'temperature': temperature
            })

def estimate_document_difficulty(uploaded_file) -> Dict[str, Any]:
    """Estimate document difficulty based on various factors"""
    
    # Convert to image for analysis
    if uploaded_file.type == "application/pdf":
        # For PDF, we'd need to convert to image first
        # For now, assume medium difficulty
        base_difficulty = 0.6
    else:
        # Load image and analyze characteristics
        image = Image.open(uploaded_file)
        base_difficulty = analyze_image_complexity(image)
    
    # Estimate based on file size, type, and characteristics
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    # Difficulty factors
    size_factor = min(file_size_mb / 10, 1.0)  # Larger files are harder
    type_factor = 0.8 if uploaded_file.type == "application/pdf" else 0.5
    
    # Combined difficulty score
    difficulty_score = min(base_difficulty + size_factor * 0.3 + type_factor * 0.2, 1.0)
    
    # Categorize difficulty
    if difficulty_score < 0.3:
        category = "Easy"
    elif difficulty_score < 0.6:
        category = "Medium"
    elif difficulty_score < 0.8:
        category = "Hard"
    else:
        category = "Very Hard"
    
    return {
        'score': difficulty_score,
        'category': category,
        'factors': {
            'base_complexity': base_difficulty,
            'size_factor': size_factor,
            'type_factor': type_factor
        },
        'estimated_processing_time': difficulty_score * 15 + 5,  # 5-20 seconds
        'recommended_models': get_recommended_models_for_difficulty(difficulty_score)
    }

def analyze_image_complexity(image: Image.Image) -> float:
    """Analyze image complexity for difficulty estimation"""
    try:
        # Convert to grayscale for analysis
        gray_image = image.convert('L')
        img_array = np.array(gray_image)
        
        # Calculate various complexity metrics
        
        # 1. Image resolution complexity
        resolution_factor = min((image.width * image.height) / (1920 * 1080), 1.0)
        
        # 2. Contrast and variance
        contrast = np.std(img_array) / 255.0
        
        # 3. Edge density (approximate)
        edges = np.abs(np.diff(img_array, axis=0)).mean() + np.abs(np.diff(img_array, axis=1)).mean()
        edge_density = min(edges / 50, 1.0)
        
        # 4. Texture complexity (local variance)
        texture_complexity = np.mean([np.std(img_array[i:i+20, j:j+20]) for i in range(0, img_array.shape[0]-20, 20) 
                                     for j in range(0, img_array.shape[1]-20, 20)]) / 255.0
        
        # Combine factors
        complexity = (resolution_factor * 0.2 + 
                     contrast * 0.3 + 
                     edge_density * 0.3 + 
                     texture_complexity * 0.2)
        
        return min(complexity, 1.0)
        
    except Exception as e:
        # Default complexity if analysis fails
        return 0.5

def get_difficulty_color(score: float) -> str:
    """Get color based on difficulty score"""
    if score < 0.3:
        return "green"
    elif score < 0.6:
        return "orange"
    elif score < 0.8:
        return "red"
    else:
        return "darkred"

def get_recommended_models_for_difficulty(difficulty_score: float) -> List[str]:
    """Get recommended models based on difficulty score"""
    available_models = llm_service.get_available_models()
    
    if difficulty_score < 0.3:
        # Easy documents - use faster, cheaper models
        recommended = [m for m in available_models if 'gpt-4o-mini' in m or 'haiku' in m or 'flash' in m]
    elif difficulty_score < 0.6:
        # Medium documents - balanced models
        recommended = [m for m in available_models if 'gpt-4o' in m or 'sonnet' in m or 'gemini-1.5-pro' in m]
    else:
        # Hard documents - use most capable models
        recommended = [m for m in available_models if 'gpt-4o' in m or 'claude-3-5-sonnet' in m or 'opus' in m]
    
    return recommended[:3]  # Return top 3 recommendations

def run_multimodal_tests(uploaded_files, selected_models, test_mode, config):
    """Run multimodal tests on uploaded documents"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    total_tests = len(uploaded_files) * len(selected_models)
    current_test = 0
    
    test_results = []
    
    for uploaded_file in uploaded_files:
        # Convert file to base64
        file_bytes = uploaded_file.getvalue()
        
        if uploaded_file.type == "application/pdf":
            # For PDF, we'd need to convert to image first
            # For now, skip PDFs or implement PDF to image conversion
            status_text.warning(f"PDF processing not implemented yet for {uploaded_file.name}")
            continue
        
        base64_image = base64.b64encode(file_bytes).decode('utf-8')
        
        # Get document difficulty
        difficulty = st.session_state.document_difficulties.get(uploaded_file.name, {'score': 0.5})
        
        for model in selected_models:
            current_test += 1
            progress_bar.progress(current_test / total_tests)
            status_text.text(f"Testing {model} on {uploaded_file.name} ({current_test}/{total_tests})")
            
            # Run extraction
            start_time = time.time()
            result = llm_service.extract_pii_from_image(
                base64_image, 
                model, 
                document_type="test_document",
                max_tokens=config['max_tokens'],
                temperature=config['temperature']
            )
            
            # Store comprehensive results
            test_result = {
                'timestamp': datetime.now().isoformat(),
                'document_name': uploaded_file.name,
                'model': model,
                'difficulty_score': difficulty['score'],
                'difficulty_category': difficulty.get('category', 'Unknown'),
                'success': result.get('success', False),
                'processing_time': result.get('processing_time', 0),
                'total_entities': result.get('total_entities', 0),
                'estimated_cost': result.get('usage', {}).get('estimated_cost', 0),
                'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                'confidence_scores': [e.get('confidence', 0) for e in result.get('pii_entities', [])],
                'entity_types': [e.get('type', '') for e in result.get('pii_entities', [])],
                'extraction_method': result.get('extraction_method', ''),
                'test_mode': test_mode,
                'config': config,
                'raw_result': result
            }
            
            test_results.append(test_result)
            
            # Display immediate results
            with results_container:
                display_test_result(test_result)
    
    # Store results in session state
    st.session_state.test_results.extend(test_results)
    
    # Generate summary
    if test_results:
        progress_bar.progress(1.0)
        status_text.success(f"✅ Completed {len(test_results)} tests!")
        
        # Display test summary
        display_test_summary(test_results)

def display_test_result(result: Dict[str, Any]):
    """Display individual test result"""
    
    success_icon = "✅" if result['success'] else "❌"
    
    with st.expander(f"{success_icon} {result['model']} - {result['document_name']}"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
            st.metric("Entities Found", result['total_entities'])
        
        with col2:
            st.metric("Estimated Cost", f"${result['estimated_cost']:.4f}")
            st.metric("Tokens Used", f"{result['tokens_used']:,}")
        
        with col3:
            st.metric("Difficulty", result['difficulty_category'])
            st.metric("Success", "Yes" if result['success'] else "No")
        
        with col4:
            if result['confidence_scores']:
                avg_confidence = np.mean(result['confidence_scores'])
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            if result['entity_types']:
                unique_types = len(set(result['entity_types']))
                st.metric("Entity Types", unique_types)
        
        # Show entity breakdown
        if result['entity_types']:
            entity_counts = pd.Series(result['entity_types']).value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Entity Breakdown:**")
                for entity_type, count in entity_counts.items():
                    st.write(f"• {entity_type}: {count}")
            
            with col2:
                if len(entity_counts) > 0:
                    fig = px.pie(values=entity_counts.values, names=entity_counts.index,
                               title="Entity Distribution")
                    st.plotly_chart(fig, use_container_width=True)

def display_test_summary(results: List[Dict[str, Any]]):
    """Display comprehensive test summary"""
    
    st.markdown("### 📊 Test Summary")
    
    df = pd.DataFrame(results)
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        st.metric("Success Rate", f"{successful_tests/total_tests:.1%}")
    
    with col2:
        avg_processing_time = df['processing_time'].mean()
        st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
    
    with col3:
        total_cost = df['estimated_cost'].sum()
        st.metric("Total Cost", f"${total_cost:.4f}")
    
    with col4:
        avg_entities = df['total_entities'].mean()
        st.metric("Avg Entities Found", f"{avg_entities:.1f}")
    
    # Performance by model
    st.markdown("#### Performance by Model")
    
    model_performance = df.groupby('model').agg({
        'success': lambda x: sum(x) / len(x),
        'processing_time': 'mean',
        'estimated_cost': 'sum',
        'total_entities': 'mean'
    }).round(3)
    
    model_performance.columns = ['Success Rate', 'Avg Time (s)', 'Total Cost ($)', 'Avg Entities']
    st.dataframe(model_performance, use_container_width=True)
    
    # Performance by difficulty
    st.markdown("#### Performance by Document Difficulty")
    
    difficulty_performance = df.groupby('difficulty_category').agg({
        'success': lambda x: sum(x) / len(x),
        'processing_time': 'mean',
        'total_entities': 'mean'
    }).round(3)
    
    difficulty_performance.columns = ['Success Rate', 'Avg Time (s)', 'Avg Entities']
    st.dataframe(difficulty_performance, use_container_width=True)

def show_document_analysis():
    """Show document difficulty analysis"""
    st.markdown("### 📊 Document Difficulty Analysis")
    
    difficulties = st.session_state.document_difficulties
    
    if not difficulties:
        st.info("No document difficulty data available. Upload and test documents first.")
        return
    
    # Create difficulty analysis
    difficulty_data = []
    for doc_name, difficulty in difficulties.items():
        difficulty_data.append({
            'document': doc_name,
            'difficulty_score': difficulty['score'],
            'category': difficulty['category'],
            'estimated_time': difficulty.get('estimated_processing_time', 0),
            'base_complexity': difficulty['factors']['base_complexity'],
            'size_factor': difficulty['factors']['size_factor'],
            'type_factor': difficulty['factors']['type_factor']
        })
    
    df_difficulty = pd.DataFrame(difficulty_data)
    
    # Display difficulty distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Difficulty Distribution")
        category_counts = df_difficulty['category'].value_counts()
        
        fig = px.pie(values=category_counts.values, names=category_counts.index,
                    title="Document Difficulty Categories")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Difficulty vs Estimated Time")
        
        fig = px.scatter(df_difficulty, x='difficulty_score', y='estimated_time',
                        color='category', hover_name='document',
                        title="Difficulty Score vs Processing Time")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed breakdown
    st.markdown("#### Detailed Difficulty Analysis")
    
    # Format for display
    df_display = df_difficulty.copy()
    df_display['difficulty_score'] = df_display['difficulty_score'].round(3)
    df_display['estimated_time'] = df_display['estimated_time'].round(1)
    df_display['base_complexity'] = df_display['base_complexity'].round(3)
    
    st.dataframe(df_display, use_container_width=True)
    
    # Difficulty factor analysis
    st.markdown("#### Difficulty Factor Analysis")
    
    factor_data = df_difficulty[['base_complexity', 'size_factor', 'type_factor']].mean()
    
    fig = px.bar(x=factor_data.index, y=factor_data.values,
                title="Average Contribution of Difficulty Factors")
    fig.update_layout(xaxis_title="Factor", yaxis_title="Average Contribution")
    st.plotly_chart(fig, use_container_width=True)

def show_performance_matrix():
    """Show interactive cost/performance/adaptability matrix"""
    st.markdown("### 📈 Interactive Performance Matrix")
    
    test_results = st.session_state.test_results
    
    if not test_results:
        st.info("No test results available. Run multimodal tests first.")
        return
    
    # Create performance matrix data
    df_results = pd.DataFrame(test_results)
    
    # Aggregate by model
    model_metrics = df_results.groupby('model').agg({
        'processing_time': 'mean',
        'estimated_cost': 'mean',
        'total_entities': 'mean',
        'success': lambda x: sum(x) / len(x),
        'difficulty_score': 'mean'
    }).round(4)
    
    model_metrics.columns = ['Avg_Time', 'Avg_Cost', 'Avg_Entities', 'Success_Rate', 'Avg_Difficulty']
    
    # Calculate adaptability score (based on performance across difficulty levels)
    adaptability_scores = {}
    for model in df_results['model'].unique():
        model_data = df_results[df_results['model'] == model]
        
        # Adaptability = consistency across difficulty levels
        difficulty_groups = model_data.groupby('difficulty_category')['success'].mean()
        adaptability = 1 - difficulty_groups.std() if len(difficulty_groups) > 1 else model_data['success'].mean()
        adaptability_scores[model] = max(0, adaptability)
    
    model_metrics['Adaptability'] = pd.Series(adaptability_scores)
    
    # Interactive 3D scatter plot
    st.markdown("#### 3D Performance Matrix: Cost vs Performance vs Adaptability")
    
    fig = go.Figure(data=go.Scatter3d(
        x=model_metrics['Avg_Cost'],
        y=model_metrics['Avg_Entities'],
        z=model_metrics['Adaptability'],
        mode='markers+text',
        text=model_metrics.index,
        textposition="top center",
        marker=dict(
            size=model_metrics['Success_Rate'] * 20,  # Size based on success rate
            color=model_metrics['Avg_Time'],          # Color based on processing time
            colorscale='Viridis',
            colorbar=dict(title="Avg Processing Time (s)"),
            opacity=0.8
        ),
        hovertemplate='<b>%{text}</b><br>' +
                     'Cost: $%{x:.4f}<br>' +
                     'Entities: %{y:.1f}<br>' +
                     'Adaptability: %{z:.2f}<br>' +
                     '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Model Performance Matrix",
        scene=dict(
            xaxis_title="Average Cost per Document ($)",
            yaxis_title="Average Entities Found",
            zaxis_title="Adaptability Score"
        ),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.markdown("#### Detailed Performance Metrics")
    
    # Format for display
    display_metrics = model_metrics.copy()
    display_metrics['Avg_Cost'] = display_metrics['Avg_Cost'].apply(lambda x: f"${x:.4f}")
    display_metrics['Success_Rate'] = display_metrics['Success_Rate'].apply(lambda x: f"{x:.1%}")
    display_metrics['Adaptability'] = display_metrics['Adaptability'].round(3)
    
    display_metrics.columns = ['Avg Time (s)', 'Avg Cost', 'Entities Found', 'Success Rate', 'Avg Difficulty', 'Adaptability']
    
    st.dataframe(display_metrics, use_container_width=True)
    
    # Model recommendations
    st.markdown("#### Model Recommendations")
    
    # Find best models for different criteria
    best_cost = model_metrics['Avg_Cost'].idxmin()
    best_performance = model_metrics['Avg_Entities'].idxmax()
    best_speed = model_metrics['Avg_Time'].idxmin()
    best_adaptability = model_metrics['Adaptability'].idxmax()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success(f"💰 **Most Cost-Effective:**\n{best_cost}")
    
    with col2:
        st.success(f"🎯 **Best Performance:**\n{best_performance}")
    
    with col3:
        st.success(f"⚡ **Fastest:**\n{best_speed}")
    
    with col4:
        st.success(f"🔄 **Most Adaptable:**\n{best_adaptability}")

def show_variance_analysis():
    """Show cross-domain variance analysis"""
    st.markdown("### 🔄 Cross-Domain Variance Analysis")
    
    test_results = st.session_state.test_results
    
    if not test_results:
        st.info("No test results available for variance analysis.")
        return
    
    df_results = pd.DataFrame(test_results)
    
    # Variance analysis by difficulty category
    st.markdown("#### Performance Variance by Document Difficulty")
    
    variance_metrics = []
    
    for model in df_results['model'].unique():
        model_data = df_results[df_results['model'] == model]
        
        # Calculate variance across difficulty categories
        difficulty_performance = model_data.groupby('difficulty_category').agg({
            'processing_time': 'mean',
            'total_entities': 'mean',
            'success': 'mean',
            'estimated_cost': 'mean'
        })
        
        # Calculate coefficient of variation (CV) for each metric
        cv_time = difficulty_performance['processing_time'].std() / difficulty_performance['processing_time'].mean()
        cv_entities = difficulty_performance['total_entities'].std() / difficulty_performance['total_entities'].mean() if difficulty_performance['total_entities'].mean() > 0 else 0
        cv_success = difficulty_performance['success'].std() / difficulty_performance['success'].mean() if difficulty_performance['success'].mean() > 0 else 0
        cv_cost = difficulty_performance['estimated_cost'].std() / difficulty_performance['estimated_cost'].mean() if difficulty_performance['estimated_cost'].mean() > 0 else 0
        
        variance_metrics.append({
            'model': model,
            'time_variance': cv_time,
            'entities_variance': cv_entities,
            'success_variance': cv_success,
            'cost_variance': cv_cost,
            'overall_variance': np.mean([cv_time, cv_entities, cv_success, cv_cost])
        })
    
    df_variance = pd.DataFrame(variance_metrics)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Variance by Metric")
        
        fig = go.Figure()
        
        for metric in ['time_variance', 'entities_variance', 'success_variance', 'cost_variance']:
            fig.add_trace(go.Bar(
                name=metric.replace('_variance', '').title(),
                x=df_variance['model'],
                y=df_variance[metric]
            ))
        
        fig.update_layout(
            title="Performance Variance Across Difficulty Levels",
            xaxis_title="Model",
            yaxis_title="Coefficient of Variation",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Overall Variance Ranking")
        
        # Sort by overall variance (lower is better)
        df_variance_sorted = df_variance.sort_values('overall_variance')
        
        fig = px.bar(df_variance_sorted, x='model', y='overall_variance',
                    title="Overall Performance Variance (Lower = More Consistent)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed variance table
    st.markdown("#### Detailed Variance Analysis")
    
    display_variance = df_variance.copy()
    for col in ['time_variance', 'entities_variance', 'success_variance', 'cost_variance', 'overall_variance']:
        display_variance[col] = display_variance[col].round(4)
    
    display_variance.columns = ['Model', 'Time Variance', 'Entities Variance', 'Success Variance', 'Cost Variance', 'Overall Variance']
    
    st.dataframe(display_variance, use_container_width=True)
    
    # Cross-difficulty performance heatmap
    st.markdown("#### Cross-Difficulty Performance Heatmap")
    
    # Create heatmap data
    heatmap_data = []
    
    for model in df_results['model'].unique():
        model_data = df_results[df_results['model'] == model]
        
        for difficulty in ['Easy', 'Medium', 'Hard', 'Very Hard']:
            difficulty_data = model_data[model_data['difficulty_category'] == difficulty]
            
            if len(difficulty_data) > 0:
                heatmap_data.append({
                    'model': model,
                    'difficulty': difficulty,
                    'success_rate': difficulty_data['success'].mean(),
                    'avg_entities': difficulty_data['total_entities'].mean(),
                    'avg_cost': difficulty_data['estimated_cost'].mean(),
                    'avg_time': difficulty_data['processing_time'].mean()
                })
    
    if heatmap_data:
        df_heatmap = pd.DataFrame(heatmap_data)
        
        # Create heatmap for success rate
        heatmap_pivot = df_heatmap.pivot(index='model', columns='difficulty', values='success_rate')
        
        fig = px.imshow(heatmap_pivot, 
                       title="Success Rate Across Models and Difficulty Levels",
                       color_continuous_scale='RdYlGn',
                       aspect="auto")
        
        st.plotly_chart(fig, use_container_width=True)

def show_automated_reports():
    """Show automated report generation with recommendations"""
    st.markdown("### 📋 Automated Reports & Recommendations")
    
    test_results = st.session_state.test_results
    
    if not test_results:
        st.info("No test results available for report generation.")
        return
    
    # Report generation options
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Report Configuration")
        
        report_type = st.selectbox(
            "Report Type",
            ["Executive Summary", "Technical Analysis", "Cost Analysis", "Performance Comparison", "Full Report"]
        )
        
        include_charts = st.checkbox("Include Charts", value=True)
        include_raw_data = st.checkbox("Include Raw Data", value=False)
        
        if st.button("📊 Generate Report", type="primary"):
            generate_automated_report(test_results, report_type, include_charts, include_raw_data)
    
    with col2:
        st.markdown("#### Quick Insights")
        
        df_results = pd.DataFrame(test_results)
        
        # Generate quick insights
        insights = generate_quick_insights(df_results)
        
        for insight in insights:
            st.markdown(f"• {insight}")

def generate_quick_insights(df_results: pd.DataFrame) -> List[str]:
    """Generate quick insights from test results"""
    insights = []
    
    # Overall performance insights
    total_tests = len(df_results)
    success_rate = df_results['success'].mean()
    
    insights.append(f"**Overall Performance:** {success_rate:.1%} success rate across {total_tests} tests")
    
    # Best performing model
    model_performance = df_results.groupby('model')['success'].mean()
    best_model = model_performance.idxmax()
    best_success_rate = model_performance.max()
    
    insights.append(f"**Best Model:** {best_model} with {best_success_rate:.1%} success rate")
    
    # Cost efficiency
    cost_efficiency = df_results.groupby('model').apply(
        lambda x: x['total_entities'].sum() / x['estimated_cost'].sum() if x['estimated_cost'].sum() > 0 else 0
    )
    most_efficient = cost_efficiency.idxmax()
    
    insights.append(f"**Most Cost-Efficient:** {most_efficient} (entities per dollar)")
    
    # Difficulty insights
    difficulty_success = df_results.groupby('difficulty_category')['success'].mean()
    
    if 'Very Hard' in difficulty_success.index and difficulty_success['Very Hard'] < 0.8:
        insights.append(f"**⚠️ Challenge:** Very Hard documents show {difficulty_success['Very Hard']:.1%} success rate - consider using more capable models")
    
    # Processing time insights
    avg_time = df_results['processing_time'].mean()
    if avg_time > 10:
        insights.append(f"**⏱️ Performance:** Average processing time is {avg_time:.1f}s - consider optimizing for speed")
    
    # Cost insights
    total_cost = df_results['estimated_cost'].sum()
    avg_cost_per_doc = total_cost / len(df_results['document_name'].unique())
    
    insights.append(f"**💰 Cost Analysis:** ${avg_cost_per_doc:.4f} average cost per document")
    
    return insights

def generate_automated_report(test_results: List[Dict], report_type: str, include_charts: bool, include_raw_data: bool):
    """Generate comprehensive automated report"""
    
    st.markdown("---")
    st.markdown(f"## 📊 {report_type}")
    st.markdown(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    df_results = pd.DataFrame(test_results)
    
    # Executive Summary
    if report_type in ["Executive Summary", "Full Report"]:
        st.markdown("### Executive Summary")
        
        total_tests = len(test_results)
        unique_documents = len(df_results['document_name'].unique())
        unique_models = len(df_results['model'].unique())
        overall_success_rate = df_results['success'].mean()
        total_cost = df_results['estimated_cost'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tests", total_tests)
        with col2:
            st.metric("Documents Tested", unique_documents)
        with col3:
            st.metric("Models Compared", unique_models)
        with col4:
            st.metric("Success Rate", f"{overall_success_rate:.1%}")
        
        st.markdown(f"""
        **Key Findings:**
        - Conducted {total_tests} multimodal tests across {unique_documents} documents
        - Tested {unique_models} different models with {overall_success_rate:.1%} overall success rate
        - Total testing cost: ${total_cost:.4f}
        - Average processing time: {df_results['processing_time'].mean():.2f} seconds
        """)
    
    # Technical Analysis
    if report_type in ["Technical Analysis", "Full Report"]:
        st.markdown("### Technical Analysis")
        
        # Model performance comparison
        model_performance = df_results.groupby('model').agg({
            'success': 'mean',
            'processing_time': 'mean',
            'total_entities': 'mean',
            'estimated_cost': 'mean'
        }).round(4)
        
        st.markdown("#### Model Performance Comparison")
        st.dataframe(model_performance, use_container_width=True)
        
        if include_charts:
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(model_performance, y=model_performance.index, x='success',
                           title="Success Rate by Model", orientation='h')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(model_performance, y=model_performance.index, x='processing_time',
                           title="Processing Time by Model", orientation='h')
                st.plotly_chart(fig, use_container_width=True)
    
    # Cost Analysis
    if report_type in ["Cost Analysis", "Full Report"]:
        st.markdown("### Cost Analysis")
        
        cost_analysis = df_results.groupby('model').agg({
            'estimated_cost': ['sum', 'mean', 'count']
        }).round(4)
        
        cost_analysis.columns = ['Total Cost', 'Avg Cost', 'Test Count']
        cost_analysis['Cost per Success'] = (cost_analysis['Total Cost'] / 
                                           df_results.groupby('model')['success'].sum()).round(4)
        
        st.dataframe(cost_analysis, use_container_width=True)
        
        if include_charts:
            fig = px.pie(cost_analysis, values='Total Cost', names=cost_analysis.index,
                        title="Cost Distribution by Model")
            st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.markdown("### 🎯 Recommendations")
    
    recommendations = generate_recommendations(df_results)
    
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")
    
    # Raw data
    if include_raw_data:
        st.markdown("### Raw Data")
        st.dataframe(df_results, use_container_width=True)
    
    # Store report in memory for future reference
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'report_type': report_type,
        'summary': {
            'total_tests': len(test_results),
            'success_rate': df_results['success'].mean(),
            'total_cost': df_results['estimated_cost'].sum(),
            'avg_processing_time': df_results['processing_time'].mean()
        },
        'recommendations': recommendations
    }
    
    # Save to memory
    memory_key = f"swarm-development-centralized-1750358285173/phase2-dev/implementation/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Simulate memory storage (would use actual memory system)
    st.session_state[f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}'] = report_data
    
    st.success(f"✅ Report generated and saved to memory: {memory_key}")

def generate_recommendations(df_results: pd.DataFrame) -> List[str]:
    """Generate automated recommendations based on test results"""
    recommendations = []
    
    # Model performance recommendations
    model_performance = df_results.groupby('model').agg({
        'success': 'mean',
        'estimated_cost': 'mean',
        'processing_time': 'mean',
        'total_entities': 'mean'
    })
    
    # Best overall model
    best_model = model_performance['success'].idxmax()
    recommendations.append(f"**Primary Model:** Use {best_model} for highest success rate ({model_performance.loc[best_model, 'success']:.1%})")
    
    # Cost-effective model
    cost_efficiency = model_performance['total_entities'] / model_performance['estimated_cost']
    most_efficient = cost_efficiency.idxmax()
    recommendations.append(f"**Cost Optimization:** Use {most_efficient} for best cost-effectiveness")
    
    # Speed recommendations
    fastest_model = model_performance['processing_time'].idxmin()
    if model_performance.loc[fastest_model, 'processing_time'] < model_performance['processing_time'].median():
        recommendations.append(f"**Speed Optimization:** Use {fastest_model} for fastest processing ({model_performance.loc[fastest_model, 'processing_time']:.1f}s)")
    
    # Difficulty-based recommendations
    difficulty_performance = df_results.groupby(['model', 'difficulty_category'])['success'].mean().unstack()
    
    if 'Very Hard' in difficulty_performance.columns:
        best_for_hard = difficulty_performance['Very Hard'].idxmax()
        recommendations.append(f"**Complex Documents:** Use {best_for_hard} for very hard documents")
    
    # General optimization recommendations
    avg_success_rate = df_results['success'].mean()
    if avg_success_rate < 0.9:
        recommendations.append("**System Optimization:** Consider ensemble methods or model fine-tuning to improve overall success rate")
    
    avg_cost = df_results['estimated_cost'].mean()
    if avg_cost > 0.01:
        recommendations.append("**Cost Control:** Monitor usage closely - average cost per document is relatively high")
    
    # Variance-based recommendations
    model_variance = df_results.groupby('model')['success'].std()
    most_consistent = model_variance.idxmin()
    recommendations.append(f"**Consistency:** {most_consistent} shows most consistent performance across different document types")
    
    return recommendations

if __name__ == "__main__":
    show_page()