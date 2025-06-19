"""
Interactive Cost/Performance/Adaptability Matrix

This module provides a comprehensive 3D visualization and analysis system for optimizing
model selection across multiple performance dimensions including cost, accuracy, latency,
and adaptability.
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
import sys
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import session_state, ui_components, auth, run_history
from llm.multimodal_llm_service import llm_service
from utils.variance_analysis import variance_analyzer, VarianceDimension, VarianceMetric

logger = logging.getLogger(__name__)

def show_page():
    """Main interactive performance matrix page"""
    st.markdown('<div class="section-header">🎯 Performance Optimization Matrix</div>', 
                unsafe_allow_html=True)
    st.markdown("Interactive 3D analysis for optimal model selection across cost, performance, and adaptability dimensions.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Initialize matrix state
    initialize_matrix_state()
    
    # Configuration sidebar
    show_matrix_configuration()
    
    # Main matrix visualization
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 3D Performance Matrix",
        "📊 Multi-Criteria Analysis", 
        "🔍 Optimization Explorer",
        "📋 Selection Wizard"
    ])
    
    with tab1:
        show_3d_performance_matrix()
    
    with tab2:
        show_multi_criteria_analysis()
    
    with tab3:
        show_optimization_explorer()
    
    with tab4:
        show_selection_wizard()

def initialize_matrix_state():
    """Initialize performance matrix session state"""
    if 'matrix_config' not in st.session_state:
        st.session_state.matrix_config = {
            'cost_weight': 0.33,
            'performance_weight': 0.33,
            'adaptability_weight': 0.34,
            'selected_models': [],
            'optimization_target': 'balanced',
            'constraints': {}
        }
    
    if 'matrix_data' not in st.session_state:
        st.session_state.matrix_data = None

def show_matrix_configuration():
    """Show configuration sidebar for matrix analysis"""
    with st.sidebar:
        st.markdown("### ⚙️ Matrix Configuration")
        
        # Weight configuration
        st.markdown("#### Criterion Weights")
        
        cost_weight = st.slider(
            "Cost Importance",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.matrix_config['cost_weight'],
            step=0.05,
            help="Higher weight = cost is more important in selection"
        )
        
        performance_weight = st.slider(
            "Performance Importance",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.matrix_config['performance_weight'],
            step=0.05,
            help="Higher weight = accuracy/quality is more important"
        )
        
        adaptability_weight = st.slider(
            "Adaptability Importance",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.matrix_config['adaptability_weight'],
            step=0.05,
            help="Higher weight = consistency across domains is more important"
        )
        
        # Normalize weights to sum to 1
        total_weight = cost_weight + performance_weight + adaptability_weight
        if total_weight > 0:
            cost_weight /= total_weight
            performance_weight /= total_weight
            adaptability_weight /= total_weight
        
        # Update session state
        st.session_state.matrix_config.update({
            'cost_weight': cost_weight,
            'performance_weight': performance_weight,
            'adaptability_weight': adaptability_weight
        })
        
        # Display normalized weights
        st.markdown("**Normalized Weights:**")
        st.write(f"Cost: {cost_weight:.1%}")
        st.write(f"Performance: {performance_weight:.1%}")
        st.write(f"Adaptability: {adaptability_weight:.1%}")
        
        st.markdown("---")
        
        # Optimization target
        st.markdown("#### Optimization Target")
        
        optimization_target = st.selectbox(
            "Primary Goal",
            ["balanced", "cost_minimize", "performance_maximize", "adaptability_maximize", "custom"],
            index=0,
            help="What should the system optimize for?"
        )
        
        st.session_state.matrix_config['optimization_target'] = optimization_target
        
        # Constraints
        st.markdown("#### Constraints")
        
        max_cost = st.number_input(
            "Max Cost per Document ($)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.001,
            format="%.4f"
        )
        
        min_accuracy = st.slider(
            "Min Accuracy Required",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        max_latency = st.number_input(
            "Max Latency (seconds)",
            min_value=1,
            max_value=60,
            value=30,
            step=1
        )
        
        st.session_state.matrix_config['constraints'] = {
            'max_cost': max_cost,
            'min_accuracy': min_accuracy,
            'max_latency': max_latency
        }
        
        # Update analysis button
        if st.button("🔄 Update Analysis", type="primary"):
            st.session_state.matrix_data = None  # Force refresh
            st.rerun()

def get_matrix_data() -> Optional[pd.DataFrame]:
    """Get and prepare performance matrix data"""
    
    # Check cache first
    if st.session_state.matrix_data is not None:
        return st.session_state.matrix_data
    
    # Get recent performance data
    performance_data = get_recent_performance_data()
    
    if not performance_data:
        return None
    
    # Create DataFrame and calculate matrix metrics
    df = pd.DataFrame(performance_data)
    
    # Calculate performance metrics for each model
    if 'model' not in df.columns:
        return None
    
    matrix_data = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        if len(model_data) < 3:  # Need minimum data points
            continue
        
        # Calculate core metrics
        avg_accuracy = model_data['accuracy_score'].mean() if 'accuracy_score' in model_data else 0
        avg_cost = model_data['cost'].mean() if 'cost' in model_data else 0
        avg_latency = model_data['latency'].mean() if 'latency' in model_data else 0
        
        # Calculate adaptability score
        adaptability = calculate_model_adaptability(model_data)
        
        # Calculate efficiency metrics
        cost_efficiency = avg_accuracy / avg_cost if avg_cost > 0 else 0
        speed_efficiency = avg_accuracy / avg_latency if avg_latency > 0 else 0
        
        # Calculate quality metrics
        accuracy_std = model_data['accuracy_score'].std() if 'accuracy_score' in model_data else 0
        consistency = 1 - min(accuracy_std, 1.0)  # Higher consistency for lower std
        
        # Calculate robustness (% of tests with accuracy > threshold)
        robustness = (model_data['accuracy_score'] > 0.7).mean() if 'accuracy_score' in model_data else 0
        
        # Normalize metrics for comparison
        normalized_metrics = calculate_normalized_metrics(model_data)
        
        matrix_entry = {
            'model': model,
            'provider': model.split('/')[0] if '/' in model else 'unknown',
            'model_name': model.split('/')[1] if '/' in model else model,
            
            # Core metrics
            'accuracy': avg_accuracy,
            'cost': avg_cost,
            'latency': avg_latency,
            'adaptability': adaptability,
            
            # Efficiency metrics
            'cost_efficiency': cost_efficiency,
            'speed_efficiency': speed_efficiency,
            
            # Quality metrics
            'consistency': consistency,
            'robustness': robustness,
            
            # Normalized metrics (0-1 scale)
            'accuracy_norm': normalized_metrics.get('accuracy_norm', 0),
            'cost_norm': 1 - normalized_metrics.get('cost_norm', 0),  # Inverted (lower cost = better)
            'latency_norm': 1 - normalized_metrics.get('latency_norm', 0),  # Inverted (lower latency = better)
            'adaptability_norm': normalized_metrics.get('adaptability_norm', 0),
            
            # Sample size
            'sample_size': len(model_data),
            
            # Raw data for detailed analysis
            'raw_data': model_data.to_dict('records')
        }
        
        # Calculate composite scores
        matrix_entry.update(calculate_composite_scores(matrix_entry))
        
        matrix_data.append(matrix_entry)
    
    if not matrix_data:
        return None
    
    # Convert to DataFrame
    matrix_df = pd.DataFrame(matrix_data)
    
    # Cache the results
    st.session_state.matrix_data = matrix_df
    
    return matrix_df

def get_recent_performance_data(hours: int = 168) -> List[Dict[str, Any]]:
    """Get recent performance data with enhanced metrics"""
    performance_data = []
    
    # Get data from test results
    test_results = st.session_state.get('test_results', [])
    
    # Filter recent data
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    for result in test_results:
        if 'timestamp' in result:
            try:
                result_time = datetime.fromisoformat(result['timestamp'])
                if result_time >= cutoff_time:
                    enhanced_result = enhance_result_with_matrix_metrics(result)
                    performance_data.append(enhanced_result)
            except:
                continue
    
    # Get data from run history if available
    try:
        run_data = run_history.run_history_manager.get_recent_runs(limit=100)
        
        for run in run_data:
            try:
                run_time = datetime.fromisoformat(run['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
                if run_time >= cutoff_time:
                    enhanced_run = enhance_run_with_matrix_metrics(run)
                    performance_data.append(enhanced_run)
            except:
                continue
    except:
        pass
    
    return performance_data

def enhance_result_with_matrix_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance test result with matrix-specific metrics"""
    enhanced = result.copy()
    
    # Calculate accuracy score based on confidence and success
    if result.get('success', False):
        confidence_scores = result.get('confidence_scores', [])
        if confidence_scores:
            accuracy_score = np.mean(confidence_scores)
        else:
            accuracy_score = 0.8  # Default for successful extractions
    else:
        accuracy_score = 0.0
    
    enhanced['accuracy_score'] = accuracy_score
    enhanced['cost'] = result.get('estimated_cost', 0)
    enhanced['latency'] = result.get('processing_time', 0)
    enhanced['success_rate'] = 1.0 if result.get('success', False) else 0.0
    enhanced['entity_count'] = result.get('total_entities', 0)
    enhanced['difficulty_score'] = result.get('difficulty_score', 0.5)
    
    return enhanced

def enhance_run_with_matrix_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance run history data with matrix metrics"""
    enhanced = run.copy()
    
    # Calculate accuracy based on entities found vs expected
    total_entities = run.get('total_entities', 0)
    total_documents = run.get('total_documents', 1)
    
    # Estimate accuracy (this would be better with ground truth data)
    entities_per_doc = total_entities / total_documents
    accuracy_score = min(entities_per_doc / 5, 1.0)  # Assume 5 entities per doc is perfect
    
    enhanced['accuracy_score'] = accuracy_score
    enhanced['cost'] = run.get('total_cost', 0) / total_documents
    enhanced['latency'] = run.get('total_processing_time', 0) / total_documents
    enhanced['success_rate'] = 1.0  # Assume run history only includes successful runs
    enhanced['entity_count'] = entities_per_doc
    enhanced['difficulty_score'] = 0.5  # Unknown
    
    return enhanced

def calculate_model_adaptability(model_data: pd.DataFrame) -> float:
    """Calculate adaptability score for a model"""
    try:
        # Adaptability based on performance consistency across different conditions
        
        # If we have difficulty levels, calculate variance across them
        if 'difficulty_level' in model_data.columns:
            difficulty_groups = model_data.groupby('difficulty_level')['accuracy_score']
            if len(difficulty_groups) > 1:
                group_means = difficulty_groups.mean()
                cv = group_means.std() / group_means.mean() if group_means.mean() > 0 else 1.0
                adaptability = max(0, 1 - cv)  # Lower CV = higher adaptability
                return adaptability
        
        # Otherwise, use general performance consistency
        if 'accuracy_score' in model_data.columns:
            accuracy_values = model_data['accuracy_score']
            if len(accuracy_values) > 1:
                cv = accuracy_values.std() / accuracy_values.mean() if accuracy_values.mean() > 0 else 1.0
                adaptability = max(0, 1 - cv)
                return adaptability
        
        return 0.7  # Default adaptability score
        
    except Exception as e:
        logger.warning(f"Error calculating adaptability: {e}")
        return 0.5

def calculate_normalized_metrics(model_data: pd.DataFrame) -> Dict[str, float]:
    """Calculate normalized metrics for comparison"""
    normalized = {}
    
    # Get global min/max for normalization (would be better to store these globally)
    all_data = get_recent_performance_data()
    if all_data:
        all_df = pd.DataFrame(all_data)
        
        for metric in ['accuracy_score', 'cost', 'latency']:
            if metric in model_data.columns and metric in all_df.columns:
                metric_values = model_data[metric]
                global_min = all_df[metric].min()
                global_max = all_df[metric].max()
                
                if global_max > global_min:
                    normalized_value = (metric_values.mean() - global_min) / (global_max - global_min)
                    normalized[f"{metric.replace('_score', '')}_norm"] = min(max(normalized_value, 0), 1)
                else:
                    normalized[f"{metric.replace('_score', '')}_norm"] = 0.5
    
    # Calculate adaptability normalization
    adaptability = calculate_model_adaptability(model_data)
    normalized['adaptability_norm'] = adaptability
    
    return normalized

def calculate_composite_scores(matrix_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate composite performance scores"""
    config = st.session_state.matrix_config
    
    # Weighted composite score
    composite_score = (
        matrix_entry['accuracy_norm'] * config['performance_weight'] +
        matrix_entry['cost_norm'] * config['cost_weight'] +
        matrix_entry['adaptability_norm'] * config['adaptability_weight']
    )
    
    # Efficiency score (accuracy per unit cost)
    efficiency_score = matrix_entry['cost_efficiency'] if matrix_entry['cost'] > 0 else 0
    
    # Quality score (combination of accuracy, consistency, robustness)
    quality_score = (
        matrix_entry['accuracy'] * 0.5 +
        matrix_entry['consistency'] * 0.3 +
        matrix_entry['robustness'] * 0.2
    )
    
    # Pareto efficiency (is this model on the efficient frontier?)
    pareto_efficient = calculate_pareto_efficiency(matrix_entry)
    
    return {
        'composite_score': composite_score,
        'efficiency_score': efficiency_score,
        'quality_score': quality_score,
        'pareto_efficient': pareto_efficient
    }

def calculate_pareto_efficiency(matrix_entry: Dict[str, Any]) -> bool:
    """Determine if a model is Pareto efficient"""
    # This would need all models to compare properly
    # For now, use a simple heuristic
    
    # A model is considered efficient if it's in the top 25% for at least one metric
    # and not in the bottom 25% for any metric
    
    metrics = ['accuracy_norm', 'cost_norm', 'adaptability_norm']
    scores = [matrix_entry.get(metric, 0.5) for metric in metrics]
    
    # Check if any score is in top 25% (> 0.75)
    has_strength = any(score > 0.75 for score in scores)
    
    # Check if any score is in bottom 25% (< 0.25)
    has_weakness = any(score < 0.25 for score in scores)
    
    return has_strength and not has_weakness

def show_3d_performance_matrix():
    """Show interactive 3D performance matrix"""
    st.markdown("### 🎯 3D Performance Matrix")
    
    matrix_data = get_matrix_data()
    
    if matrix_data is None or len(matrix_data) == 0:
        st.info("No performance data available for matrix analysis. Run some multimodal tests first.")
        return
    
    # Matrix visualization controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("#### Visualization Controls")
        
        x_axis = st.selectbox("X-Axis", ["cost", "latency", "accuracy", "consistency"], index=0)
        y_axis = st.selectbox("Y-Axis", ["accuracy", "cost", "latency", "adaptability"], index=0)
        z_axis = st.selectbox("Z-Axis", ["adaptability", "accuracy", "cost", "consistency"], index=0)
        
        color_by = st.selectbox("Color By", ["composite_score", "provider", "efficiency_score", "quality_score"], index=0)
        size_by = st.selectbox("Size By", ["sample_size", "robustness", "consistency", "accuracy"], index=0)
        
        show_pareto = st.checkbox("Highlight Pareto Efficient", value=True)
        show_constraints = st.checkbox("Show Constraint Violations", value=True)
    
    with col1:
        # Create 3D scatter plot
        fig = create_3d_scatter_plot(matrix_data, x_axis, y_axis, z_axis, color_by, size_by, show_pareto)
        
        # Add constraint violations if enabled
        if show_constraints:
            fig = add_constraint_visualization(fig, matrix_data)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance matrix table
    st.markdown("#### Performance Matrix Summary")
    
    # Format for display
    display_columns = ['model', 'accuracy', 'cost', 'latency', 'adaptability', 'composite_score', 'pareto_efficient']
    display_data = matrix_data[display_columns].copy()
    
    # Format numeric columns
    numeric_columns = ['accuracy', 'cost', 'latency', 'adaptability', 'composite_score']
    for col in numeric_columns:
        if col == 'cost':
            display_data[col] = display_data[col].apply(lambda x: f"${x:.4f}")
        elif col in ['accuracy', 'adaptability', 'composite_score']:
            display_data[col] = display_data[col].apply(lambda x: f"{x:.1%}")
        else:
            display_data[col] = display_data[col].apply(lambda x: f"{x:.1f}s")
    
    # Sort by composite score
    display_data = display_data.sort_values('composite_score', ascending=False)
    
    st.dataframe(display_data, use_container_width=True)
    
    # Model recommendations
    show_matrix_recommendations(matrix_data)

def create_3d_scatter_plot(data: pd.DataFrame, x_axis: str, y_axis: str, z_axis: str, 
                          color_by: str, size_by: str, show_pareto: bool) -> go.Figure:
    """Create 3D scatter plot for performance matrix"""
    
    # Prepare data
    hover_text = []
    for _, row in data.iterrows():
        hover_info = f"<b>{row['model']}</b><br>"
        hover_info += f"Accuracy: {row['accuracy']:.1%}<br>"
        hover_info += f"Cost: ${row['cost']:.4f}<br>"
        hover_info += f"Latency: {row['latency']:.1f}s<br>"
        hover_info += f"Adaptability: {row['adaptability']:.1%}<br>"
        hover_info += f"Composite: {row['composite_score']:.1%}<br>"
        hover_info += f"Sample Size: {row['sample_size']}"
        hover_text.append(hover_info)
    
    # Create base scatter plot
    fig = go.Figure()
    
    # Regular points
    regular_mask = ~data['pareto_efficient'] if show_pareto else [True] * len(data)
    
    if any(regular_mask):
        fig.add_trace(go.Scatter3d(
            x=data.loc[regular_mask, x_axis],
            y=data.loc[regular_mask, y_axis],
            z=data.loc[regular_mask, z_axis],
            mode='markers',
            marker=dict(
                size=data.loc[regular_mask, size_by] * 3 + 5,
                color=data.loc[regular_mask, color_by],
                colorscale='Viridis',
                opacity=0.7,
                colorbar=dict(title=color_by.replace('_', ' ').title())
            ),
            text=[data.loc[i, 'model'] for i in data.index if regular_mask[i]],
            hovertext=[hover_text[i] for i in range(len(hover_text)) if regular_mask[i]],
            hovertemplate='%{hovertext}<extra></extra>',
            name='Models'
        ))
    
    # Pareto efficient points (highlighted)
    if show_pareto:
        pareto_mask = data['pareto_efficient']
        
        if any(pareto_mask):
            fig.add_trace(go.Scatter3d(
                x=data.loc[pareto_mask, x_axis],
                y=data.loc[pareto_mask, y_axis],
                z=data.loc[pareto_mask, z_axis],
                mode='markers',
                marker=dict(
                    size=data.loc[pareto_mask, size_by] * 3 + 8,
                    color='red',
                    symbol='diamond',
                    opacity=0.9,
                    line=dict(color='darkred', width=2)
                ),
                text=[data.loc[i, 'model'] for i in data.index if pareto_mask[i]],
                hovertext=[hover_text[i] for i in range(len(hover_text)) if pareto_mask[i]],
                hovertemplate='%{hovertext}<br><b>Pareto Efficient</b><extra></extra>',
                name='Pareto Efficient'
            ))
    
    # Update layout
    fig.update_layout(
        title=f"3D Performance Matrix: {x_axis.title()} vs {y_axis.title()} vs {z_axis.title()}",
        scene=dict(
            xaxis_title=x_axis.replace('_', ' ').title(),
            yaxis_title=y_axis.replace('_', ' ').title(),
            zaxis_title=z_axis.replace('_', ' ').title(),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600,
        showlegend=True
    )
    
    return fig

def add_constraint_visualization(fig: go.Figure, data: pd.DataFrame) -> go.Figure:
    """Add constraint violation visualization to the plot"""
    constraints = st.session_state.matrix_config['constraints']
    
    # Identify constraint violations
    violations = []
    
    for _, row in data.iterrows():
        violated_constraints = []
        
        if row['cost'] > constraints['max_cost']:
            violated_constraints.append('Cost')
        
        if row['accuracy'] < constraints['min_accuracy']:
            violated_constraints.append('Accuracy')
        
        if row['latency'] > constraints['max_latency']:
            violated_constraints.append('Latency')
        
        if violated_constraints:
            violations.append({
                'model': row['model'],
                'violations': violated_constraints,
                'x': row['cost'],
                'y': row['accuracy'],
                'z': row['adaptability']
            })
    
    # Add constraint violation indicators
    if violations:
        violation_text = [f"{v['model']}<br>Violations: {', '.join(v['violations'])}" for v in violations]
        
        fig.add_trace(go.Scatter3d(
            x=[v['x'] for v in violations],
            y=[v['y'] for v in violations],
            z=[v['z'] for v in violations],
            mode='markers',
            marker=dict(
                size=15,
                color='orange',
                symbol='x',
                opacity=1.0,
                line=dict(color='red', width=3)
            ),
            text=violation_text,
            hovertemplate='%{text}<extra></extra>',
            name='Constraint Violations'
        ))
    
    return fig

def show_matrix_recommendations(matrix_data: pd.DataFrame):
    """Show recommendations based on matrix analysis"""
    st.markdown("#### 🎯 Matrix Recommendations")
    
    config = st.session_state.matrix_config
    target = config['optimization_target']
    
    # Get top performers by different criteria
    top_composite = matrix_data.loc[matrix_data['composite_score'].idxmax()]
    top_accuracy = matrix_data.loc[matrix_data['accuracy'].idxmax()]
    top_efficiency = matrix_data.loc[matrix_data['cost_efficiency'].idxmax()]
    most_adaptable = matrix_data.loc[matrix_data['adaptability'].idxmax()]
    
    # Pareto efficient models
    pareto_models = matrix_data[matrix_data['pareto_efficient']]['model'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🏆 Top Performers")
        
        st.success(f"**Overall Best**: {top_composite['model']}")
        st.write(f"Composite Score: {top_composite['composite_score']:.1%}")
        
        st.info(f"**Most Accurate**: {top_accuracy['model']}")
        st.write(f"Accuracy: {top_accuracy['accuracy']:.1%}")
        
        st.info(f"**Most Efficient**: {top_efficiency['model']}")
        st.write(f"Cost Efficiency: {top_efficiency['cost_efficiency']:.2f}")
        
        st.info(f"**Most Adaptable**: {most_adaptable['model']}")
        st.write(f"Adaptability: {most_adaptable['adaptability']:.1%}")
    
    with col2:
        st.markdown("##### 📊 Strategic Insights")
        
        if pareto_models:
            st.success(f"**Pareto Efficient Models**: {len(pareto_models)}")
            for model in pareto_models[:3]:  # Show top 3
                st.write(f"• {model}")
        
        # Constraint analysis
        constraints = config['constraints']
        compliant_models = matrix_data[
            (matrix_data['cost'] <= constraints['max_cost']) &
            (matrix_data['accuracy'] >= constraints['min_accuracy']) &
            (matrix_data['latency'] <= constraints['max_latency'])
        ]
        
        st.metric("Constraint Compliant", f"{len(compliant_models)}/{len(matrix_data)}")
        
        # Optimization recommendations
        recommendations = generate_matrix_recommendations(matrix_data, config)
        
        st.markdown("**Recommendations:**")
        for rec in recommendations[:3]:
            st.write(f"• {rec}")

def generate_matrix_recommendations(matrix_data: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations based on matrix analysis"""
    recommendations = []
    
    target = config['optimization_target']
    
    if target == 'cost_minimize':
        cheapest = matrix_data.loc[matrix_data['cost'].idxmin()]
        recommendations.append(f"For cost minimization, use {cheapest['model']} (${cheapest['cost']:.4f}/doc)")
    
    elif target == 'performance_maximize':
        best_accuracy = matrix_data.loc[matrix_data['accuracy'].idxmax()]
        recommendations.append(f"For maximum accuracy, use {best_accuracy['model']} ({best_accuracy['accuracy']:.1%})")
    
    elif target == 'adaptability_maximize':
        most_adaptable = matrix_data.loc[matrix_data['adaptability'].idxmax()]
        recommendations.append(f"For maximum adaptability, use {most_adaptable['model']} ({most_adaptable['adaptability']:.1%})")
    
    else:  # balanced
        best_composite = matrix_data.loc[matrix_data['composite_score'].idxmax()]
        recommendations.append(f"For balanced performance, use {best_composite['model']} (composite: {best_composite['composite_score']:.1%})")
    
    # Provider diversity recommendations
    provider_performance = matrix_data.groupby('provider')['composite_score'].mean().sort_values(ascending=False)
    best_provider = provider_performance.index[0]
    recommendations.append(f"{best_provider.upper()} models show best overall performance")
    
    # Efficiency recommendations
    if matrix_data['cost'].max() / matrix_data['cost'].min() > 5:  # High cost variance
        recommendations.append("Consider tiered approach: use cheaper models for simple documents")
    
    # Consistency recommendations
    low_consistency_models = matrix_data[matrix_data['consistency'] < 0.7]
    if len(low_consistency_models) > 0:
        recommendations.append(f"{len(low_consistency_models)} models show low consistency - consider ensemble methods")
    
    return recommendations

def show_multi_criteria_analysis():
    """Show multi-criteria decision analysis"""
    st.markdown("### 📊 Multi-Criteria Analysis")
    
    matrix_data = get_matrix_data()
    
    if matrix_data is None or len(matrix_data) == 0:
        st.info("No data available for multi-criteria analysis.")
        return
    
    # TOPSIS analysis
    show_topsis_analysis(matrix_data)
    
    # Sensitivity analysis
    show_sensitivity_analysis(matrix_data)
    
    # Trade-off analysis
    show_tradeoff_analysis(matrix_data)

def show_topsis_analysis(matrix_data: pd.DataFrame):
    """Show TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) analysis"""
    st.markdown("#### TOPSIS Ranking Analysis")
    
    # Prepare criteria matrix
    criteria = ['accuracy', 'cost', 'latency', 'adaptability']
    weights = [
        st.session_state.matrix_config['performance_weight'],
        st.session_state.matrix_config['cost_weight'],
        st.session_state.matrix_config['cost_weight'],  # Use cost weight for latency
        st.session_state.matrix_config['adaptability_weight']
    ]
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Extract criteria values
    criteria_matrix = matrix_data[criteria].values
    
    # Normalize the matrix
    normalized_matrix = criteria_matrix / np.sqrt((criteria_matrix**2).sum(axis=0))
    
    # Apply weights
    weighted_matrix = normalized_matrix * weights
    
    # Determine ideal and negative-ideal solutions
    # For cost and latency, lower is better (negative criteria)
    ideal_solution = np.array([
        weighted_matrix[:, 0].max(),  # accuracy (higher better)
        weighted_matrix[:, 1].min(),  # cost (lower better)
        weighted_matrix[:, 2].min(),  # latency (lower better)
        weighted_matrix[:, 3].max()   # adaptability (higher better)
    ])
    
    negative_ideal_solution = np.array([
        weighted_matrix[:, 0].min(),  # accuracy
        weighted_matrix[:, 1].max(),  # cost
        weighted_matrix[:, 2].max(),  # latency
        weighted_matrix[:, 3].min()   # adaptability
    ])
    
    # Calculate distances
    distance_to_ideal = np.sqrt(((weighted_matrix - ideal_solution)**2).sum(axis=1))
    distance_to_negative_ideal = np.sqrt(((weighted_matrix - negative_ideal_solution)**2).sum(axis=1))
    
    # Calculate TOPSIS scores
    topsis_scores = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
    
    # Create TOPSIS results DataFrame
    topsis_results = matrix_data[['model', 'accuracy', 'cost', 'latency', 'adaptability']].copy()
    topsis_results['topsis_score'] = topsis_scores
    topsis_results['topsis_rank'] = topsis_results['topsis_score'].rank(ascending=False)
    
    # Sort by TOPSIS score
    topsis_results = topsis_results.sort_values('topsis_score', ascending=False)
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**TOPSIS Rankings:**")
        
        # Format for display
        display_topsis = topsis_results.copy()
        display_topsis['accuracy'] = display_topsis['accuracy'].apply(lambda x: f"{x:.1%}")
        display_topsis['cost'] = display_topsis['cost'].apply(lambda x: f"${x:.4f}")
        display_topsis['latency'] = display_topsis['latency'].apply(lambda x: f"{x:.1f}s")
        display_topsis['adaptability'] = display_topsis['adaptability'].apply(lambda x: f"{x:.1%}")
        display_topsis['topsis_score'] = display_topsis['topsis_score'].apply(lambda x: f"{x:.3f}")
        display_topsis['topsis_rank'] = display_topsis['topsis_rank'].astype(int)
        
        st.dataframe(display_topsis, use_container_width=True)
    
    with col2:
        st.markdown("**TOPSIS Score Distribution:**")
        
        fig = px.bar(topsis_results, x='model', y='topsis_score',
                    title='TOPSIS Scores by Model')
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def show_sensitivity_analysis(matrix_data: pd.DataFrame):
    """Show sensitivity analysis for weight changes"""
    st.markdown("#### Sensitivity Analysis")
    
    st.markdown("Analyze how rankings change with different weight configurations:")
    
    # Create weight variation scenarios
    scenarios = {
        'Cost-Focused': {'cost_weight': 0.6, 'performance_weight': 0.25, 'adaptability_weight': 0.15},
        'Performance-Focused': {'cost_weight': 0.15, 'performance_weight': 0.6, 'adaptability_weight': 0.25},
        'Adaptability-Focused': {'cost_weight': 0.15, 'performance_weight': 0.25, 'adaptability_weight': 0.6},
        'Equal Weights': {'cost_weight': 0.33, 'performance_weight': 0.33, 'adaptability_weight': 0.34}
    }
    
    sensitivity_results = {}
    
    for scenario_name, weights in scenarios.items():
        # Calculate composite scores for this scenario
        composite_scores = (
            matrix_data['accuracy_norm'] * weights['performance_weight'] +
            matrix_data['cost_norm'] * weights['cost_weight'] +
            matrix_data['adaptability_norm'] * weights['adaptability_weight']
        )
        
        sensitivity_results[scenario_name] = composite_scores.rank(ascending=False)
    
    # Create sensitivity DataFrame
    sensitivity_df = pd.DataFrame(sensitivity_results, index=matrix_data['model'])
    
    # Calculate rank stability (standard deviation of ranks)
    sensitivity_df['rank_stability'] = sensitivity_df.std(axis=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Rank Changes Across Scenarios:**")
        st.dataframe(sensitivity_df, use_container_width=True)
    
    with col2:
        st.markdown("**Rank Stability:**")
        
        stability_sorted = sensitivity_df.sort_values('rank_stability')
        
        fig = px.bar(x=stability_sorted.index, y=stability_sorted['rank_stability'],
                    title='Ranking Stability (Lower = More Stable)')
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def show_tradeoff_analysis(matrix_data: pd.DataFrame):
    """Show trade-off analysis between different criteria"""
    st.markdown("#### Trade-off Analysis")
    
    # Performance vs Cost trade-off
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Performance vs Cost Trade-off:**")
        
        fig = px.scatter(matrix_data, x='cost', y='accuracy',
                        size='adaptability', hover_name='model',
                        title='Accuracy vs Cost (bubble size = adaptability)',
                        labels={'cost': 'Cost per Document ($)', 'accuracy': 'Accuracy'})
        
        # Add Pareto frontier
        pareto_points = matrix_data[matrix_data['pareto_efficient']]
        if len(pareto_points) > 0:
            fig.add_trace(go.Scatter(
                x=pareto_points['cost'],
                y=pareto_points['accuracy'],
                mode='markers',
                marker=dict(color='red', size=12, symbol='diamond'),
                name='Pareto Efficient'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Speed vs Quality Trade-off:**")
        
        fig = px.scatter(matrix_data, x='latency', y='accuracy',
                        color='cost', hover_name='model',
                        title='Accuracy vs Latency (color = cost)',
                        labels={'latency': 'Latency (seconds)', 'accuracy': 'Accuracy'})
        
        st.plotly_chart(fig, use_container_width=True)

def show_optimization_explorer():
    """Show optimization exploration interface"""
    st.markdown("### 🔍 Optimization Explorer")
    
    matrix_data = get_matrix_data()
    
    if matrix_data is None or len(matrix_data) == 0:
        st.info("No data available for optimization exploration.")
        return
    
    # Interactive optimization interface
    show_interactive_optimizer(matrix_data)
    
    # Scenario comparison
    show_scenario_comparison(matrix_data)
    
    # Optimization recommendations
    show_optimization_recommendations(matrix_data)

def show_interactive_optimizer(matrix_data: pd.DataFrame):
    """Show interactive optimization interface"""
    st.markdown("#### Interactive Optimization")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Optimization Parameters:**")
        
        # Dynamic weight adjustment
        st.markdown("*Adjust weights to see impact:*")
        
        cost_importance = st.slider("Cost Importance", 0.0, 1.0, 0.33, 0.05, key="opt_cost")
        perf_importance = st.slider("Performance Importance", 0.0, 1.0, 0.33, 0.05, key="opt_perf")
        adapt_importance = st.slider("Adaptability Importance", 0.0, 1.0, 0.34, 0.05, key="opt_adapt")
        
        # Normalize
        total = cost_importance + perf_importance + adapt_importance
        if total > 0:
            cost_w = cost_importance / total
            perf_w = perf_importance / total
            adapt_w = adapt_importance / total
        else:
            cost_w = perf_w = adapt_w = 1/3
        
        # Constraints
        st.markdown("*Constraints:*")
        max_cost_opt = st.number_input("Max Cost ($)", 0.0, 1.0, 0.1, 0.001, format="%.4f", key="opt_max_cost")
        min_acc_opt = st.slider("Min Accuracy", 0.0, 1.0, 0.7, 0.05, key="opt_min_acc")
        max_lat_opt = st.number_input("Max Latency (s)", 1, 60, 30, 1, key="opt_max_lat")
    
    with col2:
        # Calculate optimized scores
        optimized_scores = (
            matrix_data['accuracy_norm'] * perf_w +
            matrix_data['cost_norm'] * cost_w +
            matrix_data['adaptability_norm'] * adapt_w
        )
        
        # Apply constraints
        constraint_mask = (
            (matrix_data['cost'] <= max_cost_opt) &
            (matrix_data['accuracy'] >= min_acc_opt) &
            (matrix_data['latency'] <= max_lat_opt)
        )
        
        # Create results
        opt_results = matrix_data.copy()
        opt_results['optimized_score'] = optimized_scores
        opt_results['constraint_compliant'] = constraint_mask
        
        # Filter and sort
        compliant_models = opt_results[opt_results['constraint_compliant']].sort_values('optimized_score', ascending=False)
        
        if len(compliant_models) > 0:
            st.markdown("**Optimization Results:**")
            
            # Top recommendation
            top_model = compliant_models.iloc[0]
            st.success(f"**Recommended Model**: {top_model['model']}")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Accuracy", f"{top_model['accuracy']:.1%}")
            with col_b:
                st.metric("Cost", f"${top_model['cost']:.4f}")
            with col_c:
                st.metric("Latency", f"{top_model['latency']:.1f}s")
            
            # Results table
            display_cols = ['model', 'optimized_score', 'accuracy', 'cost', 'latency', 'adaptability']
            display_opt = compliant_models[display_cols].head(5)
            
            # Format for display
            display_opt['optimized_score'] = display_opt['optimized_score'].apply(lambda x: f"{x:.3f}")
            display_opt['accuracy'] = display_opt['accuracy'].apply(lambda x: f"{x:.1%}")
            display_opt['cost'] = display_opt['cost'].apply(lambda x: f"${x:.4f}")
            display_opt['latency'] = display_opt['latency'].apply(lambda x: f"{x:.1f}s")
            display_opt['adaptability'] = display_opt['adaptability'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_opt, use_container_width=True)
        
        else:
            st.warning("No models meet the specified constraints. Consider relaxing the requirements.")

def show_scenario_comparison(matrix_data: pd.DataFrame):
    """Show scenario comparison analysis"""
    st.markdown("#### Scenario Comparison")
    
    # Predefined scenarios
    scenarios = {
        'Production (Balanced)': {
            'weights': [0.3, 0.4, 0.3],  # cost, performance, adaptability
            'constraints': {'max_cost': 0.05, 'min_accuracy': 0.8, 'max_latency': 15}
        },
        'Research (Performance Focus)': {
            'weights': [0.1, 0.7, 0.2],
            'constraints': {'max_cost': 0.2, 'min_accuracy': 0.9, 'max_latency': 60}
        },
        'High-Volume (Cost Focus)': {
            'weights': [0.6, 0.2, 0.2],
            'constraints': {'max_cost': 0.01, 'min_accuracy': 0.7, 'max_latency': 10}
        },
        'Enterprise (Reliability Focus)': {
            'weights': [0.2, 0.3, 0.5],
            'constraints': {'max_cost': 0.1, 'min_accuracy': 0.85, 'max_latency': 20}
        }
    }
    
    scenario_results = {}
    
    for scenario_name, config in scenarios.items():
        weights = config['weights']
        constraints = config['constraints']
        
        # Calculate scores
        scores = (
            matrix_data['accuracy_norm'] * weights[1] +
            matrix_data['cost_norm'] * weights[0] +
            matrix_data['adaptability_norm'] * weights[2]
        )
        
        # Apply constraints
        constraint_mask = (
            (matrix_data['cost'] <= constraints['max_cost']) &
            (matrix_data['accuracy'] >= constraints['min_accuracy']) &
            (matrix_data['latency'] <= constraints['max_latency'])
        )
        
        compliant_models = matrix_data[constraint_mask]
        
        if len(compliant_models) > 0:
            best_model_idx = scores[constraint_mask].idxmax()
            best_model = matrix_data.loc[best_model_idx, 'model']
            best_score = scores[best_model_idx]
        else:
            best_model = "None (constraints too strict)"
            best_score = 0
        
        scenario_results[scenario_name] = {
            'best_model': best_model,
            'score': best_score,
            'compliant_count': len(compliant_models)
        }
    
    # Display scenario comparison
    scenario_df = pd.DataFrame(scenario_results).T
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Scenario Results:**")
        st.dataframe(scenario_df, use_container_width=True)
    
    with col2:
        st.markdown("**Scenario Analysis:**")
        
        fig = px.bar(x=scenario_df.index, y=scenario_df['score'],
                    title='Optimization Scores by Scenario')
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def show_optimization_recommendations(matrix_data: pd.DataFrame):
    """Show optimization recommendations"""
    st.markdown("#### 💡 Optimization Recommendations")
    
    # Analyze current performance distribution
    recommendations = []
    
    # Performance gap analysis
    top_performer = matrix_data.loc[matrix_data['composite_score'].idxmax()]
    avg_performance = matrix_data['composite_score'].mean()
    performance_gap = top_performer['composite_score'] - avg_performance
    
    if performance_gap > 0.2:
        recommendations.append(
            f"🎯 **Performance Gap**: {performance_gap:.1%} gap between best and average model. "
            f"Standardizing on {top_performer['model']} could improve overall performance."
        )
    
    # Cost efficiency opportunities
    cost_range = matrix_data['cost'].max() / matrix_data['cost'].min()
    if cost_range > 3:
        cheapest = matrix_data.loc[matrix_data['cost'].idxmin()]
        recommendations.append(
            f"💰 **Cost Optimization**: {cost_range:.1f}x cost difference between models. "
            f"Using {cheapest['model']} for simple tasks could reduce costs significantly."
        )
    
    # Adaptability insights
    adaptability_variance = matrix_data['adaptability'].std()
    if adaptability_variance > 0.15:
        most_adaptable = matrix_data.loc[matrix_data['adaptability'].idxmax()]
        recommendations.append(
            f"🔄 **Adaptability Variance**: High variance in adaptability ({adaptability_variance:.1%}). "
            f"{most_adaptable['model']} shows most consistent performance across domains."
        )
    
    # Portfolio recommendations
    pareto_count = sum(matrix_data['pareto_efficient'])
    if pareto_count > 1:
        recommendations.append(
            f"📊 **Portfolio Approach**: {pareto_count} Pareto-efficient models identified. "
            f"Consider a portfolio approach using different models for different use cases."
        )
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")
    
    if not recommendations:
        st.info("Performance appears well-optimized across all dimensions.")

def show_selection_wizard():
    """Show guided model selection wizard"""
    st.markdown("### 📋 Model Selection Wizard")
    
    matrix_data = get_matrix_data()
    
    if matrix_data is None or len(matrix_data) == 0:
        st.info("No data available for selection wizard.")
        return
    
    st.markdown("Answer a few questions to get personalized model recommendations:")
    
    # Step-by-step questionnaire
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Use case selection
        st.markdown("#### 1. What's your primary use case?")
        use_case = st.selectbox(
            "Select use case:",
            ["Production Processing", "Research & Development", "High-Volume Processing", 
             "Enterprise Applications", "Proof of Concept", "Cost-Sensitive Deployment"]
        )
        
        # Volume expectations
        st.markdown("#### 2. Expected processing volume?")
        volume = st.selectbox(
            "Documents per day:",
            ["< 100", "100 - 1,000", "1,000 - 10,000", "> 10,000"]
        )
        
        # Quality requirements
        st.markdown("#### 3. Quality requirements?")
        quality_req = st.selectbox(
            "Minimum accuracy required:",
            ["60% (Acceptable)", "70% (Good)", "80% (High)", "90% (Critical)"]
        )
        
        # Budget constraints
        st.markdown("#### 4. Budget considerations?")
        budget = st.selectbox(
            "Cost sensitivity:",
            ["Very Cost-Sensitive", "Moderate Budget", "Cost Not a Concern"]
        )
        
        # Performance requirements
        st.markdown("#### 5. Performance requirements?")
        latency_req = st.selectbox(
            "Maximum acceptable latency:",
            ["< 5 seconds", "< 15 seconds", "< 30 seconds", "< 60 seconds"]
        )
        
        if st.button("🎯 Get Recommendations", type="primary"):
            recommendations = generate_wizard_recommendations(
                matrix_data, use_case, volume, quality_req, budget, latency_req
            )
            
            st.session_state['wizard_recommendations'] = recommendations
    
    with col2:
        if 'wizard_recommendations' in st.session_state:
            recs = st.session_state['wizard_recommendations']
            
            st.markdown("### 🎯 Your Recommendations")
            
            # Primary recommendation
            if recs['primary']:
                st.success(f"**Primary Choice**: {recs['primary']['model']}")
                st.write(f"Match Score: {recs['primary']['score']:.1%}")
                st.write(f"Reasoning: {recs['primary']['reasoning']}")
            
            # Alternative recommendations
            if recs['alternatives']:
                st.markdown("**Alternatives:**")
                for alt in recs['alternatives'][:2]:
                    st.info(f"**{alt['model']}** (Score: {alt['score']:.1%})")
                    st.write(f"• {alt['reasoning']}")
            
            # Warnings
            if recs['warnings']:
                st.markdown("**⚠️ Considerations:**")
                for warning in recs['warnings']:
                    st.warning(warning)

def generate_wizard_recommendations(matrix_data: pd.DataFrame, use_case: str, volume: str, 
                                  quality_req: str, budget: str, latency_req: str) -> Dict[str, Any]:
    """Generate personalized recommendations based on wizard inputs"""
    
    # Parse requirements
    min_accuracy = {
        "60% (Acceptable)": 0.6,
        "70% (Good)": 0.7,
        "80% (High)": 0.8,
        "90% (Critical)": 0.9
    }[quality_req]
    
    max_latency = {
        "< 5 seconds": 5,
        "< 15 seconds": 15,
        "< 30 seconds": 30,
        "< 60 seconds": 60
    }[latency_req]
    
    cost_weight = {
        "Very Cost-Sensitive": 0.6,
        "Moderate Budget": 0.3,
        "Cost Not a Concern": 0.1
    }[budget]
    
    # Adjust weights based on use case
    if use_case == "Research & Development":
        performance_weight = 0.6
        adaptability_weight = 0.4 - cost_weight
    elif use_case == "High-Volume Processing":
        cost_weight = max(cost_weight, 0.5)
        performance_weight = 0.3
        adaptability_weight = 0.2
    elif use_case == "Enterprise Applications":
        adaptability_weight = 0.5
        performance_weight = 0.3
        cost_weight = min(cost_weight, 0.2)
    else:  # Production, PoC, Cost-Sensitive
        performance_weight = 0.4
        adaptability_weight = 0.6 - cost_weight
    
    # Filter models by constraints
    filtered_models = matrix_data[
        (matrix_data['accuracy'] >= min_accuracy) &
        (matrix_data['latency'] <= max_latency)
    ].copy()
    
    if len(filtered_models) == 0:
        return {
            'primary': None,
            'alternatives': [],
            'warnings': ["No models meet your requirements. Consider relaxing constraints."]
        }
    
    # Calculate recommendation scores
    scores = (
        filtered_models['accuracy_norm'] * performance_weight +
        filtered_models['cost_norm'] * cost_weight +
        filtered_models['adaptability_norm'] * adaptability_weight
    )
    
    filtered_models['recommendation_score'] = scores
    filtered_models = filtered_models.sort_values('recommendation_score', ascending=False)
    
    # Generate recommendations
    primary = filtered_models.iloc[0]
    primary_rec = {
        'model': primary['model'],
        'score': primary['recommendation_score'],
        'reasoning': generate_reasoning(primary, use_case, budget, quality_req)
    }
    
    # Alternative recommendations
    alternatives = []
    for _, model in filtered_models.iloc[1:4].iterrows():  # Next 3 models
        alternatives.append({
            'model': model['model'],
            'score': model['recommendation_score'],
            'reasoning': generate_alternative_reasoning(model, primary)
        })
    
    # Generate warnings
    warnings = []
    
    if primary['cost'] > 0.05:
        warnings.append(f"Cost may be high for volume processing (${primary['cost']:.4f}/doc)")
    
    if primary['adaptability'] < 0.7:
        warnings.append("Model may not perform consistently across all document types")
    
    if len(filtered_models) < 3:
        warnings.append("Limited model options available with your constraints")
    
    return {
        'primary': primary_rec,
        'alternatives': alternatives,
        'warnings': warnings
    }

def generate_reasoning(model_data: pd.Series, use_case: str, budget: str, quality_req: str) -> str:
    """Generate reasoning for recommendation"""
    reasons = []
    
    if model_data['accuracy'] > 0.85:
        reasons.append("high accuracy")
    
    if model_data['cost'] < 0.01:
        reasons.append("cost-effective")
    
    if model_data['latency'] < 10:
        reasons.append("fast processing")
    
    if model_data['adaptability'] > 0.8:
        reasons.append("consistent across document types")
    
    if model_data['pareto_efficient']:
        reasons.append("Pareto-efficient choice")
    
    reasoning = f"Best for {use_case.lower()} due to " + ", ".join(reasons)
    
    return reasoning

def generate_alternative_reasoning(model_data: pd.Series, primary_data: pd.Series) -> str:
    """Generate reasoning for alternative recommendations"""
    if model_data['cost'] < primary_data['cost']:
        return "Lower cost alternative"
    elif model_data['accuracy'] > primary_data['accuracy']:
        return "Higher accuracy option"
    elif model_data['latency'] < primary_data['latency']:
        return "Faster processing alternative"
    elif model_data['adaptability'] > primary_data['adaptability']:
        return "More adaptable across domains"
    else:
        return "Balanced alternative option"

if __name__ == "__main__":
    show_page()