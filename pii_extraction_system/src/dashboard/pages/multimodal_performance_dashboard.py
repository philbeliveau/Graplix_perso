"""
Phase 2 Multimodal Performance Dashboard

Advanced performance metrics dashboard specifically designed for multimodal LLM analysis,
providing deep insights into accuracy, cost, latency, and adaptability across different
document types and complexity levels.
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
import logging

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from dashboard.utils import session_state, ui_components, auth, run_history
from llm.multimodal_llm_service import llm_service

logger = logging.getLogger(__name__)

def show_page():
    """Main multimodal performance dashboard"""
    st.markdown('<div class="section-header">📊 Multimodal Performance Dashboard</div>', 
                unsafe_allow_html=True)
    st.markdown("Advanced performance analytics for multimodal LLM systems with real-time monitoring and predictive insights.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Initialize dashboard
    initialize_dashboard_state()
    
    # Header metrics
    show_header_metrics()
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Accuracy Analytics",
        "💰 Cost Intelligence", 
        "⚡ Latency Analysis",
        "🔄 Adaptability Metrics",
        "📈 Predictive Insights"
    ])
    
    with tab1:
        show_accuracy_analytics()
    
    with tab2:
        show_cost_intelligence()
    
    with tab3:
        show_latency_analysis()
    
    with tab4:
        show_adaptability_metrics()
    
    with tab5:
        show_predictive_insights()

def initialize_dashboard_state():
    """Initialize dashboard session state"""
    if 'dashboard_data' not in st.session_state:
        st.session_state.dashboard_data = {}
    
    if 'performance_cache' not in st.session_state:
        st.session_state.performance_cache = {}
    
    # Auto-refresh mechanism
    if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
        st.rerun()

def show_header_metrics():
    """Show key performance indicators at the top"""
    # Get recent performance data
    performance_data = get_recent_performance_data()
    
    if not performance_data:
        st.info("No recent performance data available. Run some multimodal tests to see metrics.")
        return
    
    # Calculate header metrics
    total_tests = len(performance_data)
    avg_accuracy = np.mean([p.get('accuracy_score', 0) for p in performance_data])
    total_cost = sum([p.get('cost', 0) for p in performance_data])
    avg_latency = np.mean([p.get('latency', 0) for p in performance_data])
    adaptability_score = calculate_adaptability_score(performance_data)
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Tests (24h)", 
            total_tests,
            delta=f"+{total_tests - get_previous_metric('tests', 0)}"
        )
    
    with col2:
        st.metric(
            "Avg Accuracy", 
            f"{avg_accuracy:.1%}",
            delta=f"{avg_accuracy - get_previous_metric('accuracy', 0.5):.1%}"
        )
    
    with col3:
        st.metric(
            "Total Cost", 
            f"${total_cost:.3f}",
            delta=f"${total_cost - get_previous_metric('cost', 0):.3f}"
        )
    
    with col4:
        st.metric(
            "Avg Latency", 
            f"{avg_latency:.1f}s",
            delta=f"{avg_latency - get_previous_metric('latency', 5):.1f}s",
            delta_color="inverse"
        )
    
    with col5:
        st.metric(
            "Adaptability", 
            f"{adaptability_score:.1%}",
            delta=f"{adaptability_score - get_previous_metric('adaptability', 0.5):.1%}"
        )
    
    # Store current metrics for next comparison
    store_current_metrics({
        'tests': total_tests,
        'accuracy': avg_accuracy,
        'cost': total_cost,
        'latency': avg_latency,
        'adaptability': adaptability_score
    })

def get_recent_performance_data(hours: int = 24) -> List[Dict[str, Any]]:
    """Get recent performance data from various sources"""
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
                    # Enhance result with calculated metrics
                    enhanced_result = enhance_result_with_metrics(result)
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
                    enhanced_run = enhance_run_with_metrics(run)
                    performance_data.append(enhanced_run)
            except:
                continue
    except:
        pass
    
    return performance_data

def enhance_result_with_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance test result with calculated performance metrics"""
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
    enhanced['model_provider'] = result.get('model', '').split('/')[0] if '/' in result.get('model', '') else 'unknown'
    enhanced['difficulty_level'] = result.get('difficulty_category', 'Unknown')
    
    return enhanced

def enhance_run_with_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance run history data with calculated metrics"""
    enhanced = run.copy()
    
    # Calculate accuracy based on entities found vs expected
    total_entities = run.get('total_entities', 0)
    total_documents = run.get('total_documents', 1)
    
    # Estimate accuracy (this would be better with ground truth data)
    entities_per_doc = total_entities / total_documents
    accuracy_score = min(entities_per_doc / 5, 1.0)  # Assume 5 entities per doc is perfect
    
    enhanced['accuracy_score'] = accuracy_score
    enhanced['cost'] = run.get('total_cost', 0)
    enhanced['latency'] = run.get('total_processing_time', 0) / total_documents
    enhanced['model_provider'] = run.get('model_used', '').split('/')[0] if '/' in run.get('model_used', '') else 'unknown'
    enhanced['difficulty_level'] = 'Unknown'  # Would need to be calculated
    
    return enhanced

def calculate_adaptability_score(performance_data: List[Dict[str, Any]]) -> float:
    """Calculate adaptability score based on performance consistency"""
    if not performance_data:
        return 0.0
    
    # Group by difficulty level and calculate variance
    difficulty_groups = {}
    for data in performance_data:
        difficulty = data.get('difficulty_level', 'Unknown')
        if difficulty not in difficulty_groups:
            difficulty_groups[difficulty] = []
        difficulty_groups[difficulty].append(data.get('accuracy_score', 0))
    
    if len(difficulty_groups) < 2:
        return np.mean([d.get('accuracy_score', 0) for d in performance_data])
    
    # Calculate coefficient of variation across difficulty levels
    group_means = [np.mean(scores) for scores in difficulty_groups.values()]
    overall_mean = np.mean(group_means)
    cv = np.std(group_means) / overall_mean if overall_mean > 0 else 1.0
    
    # Convert to adaptability (lower CV = higher adaptability)
    adaptability = max(0, 1 - cv)
    
    return adaptability

def get_previous_metric(metric_name: str, default: float) -> float:
    """Get previous metric value for delta calculation"""
    return st.session_state.get(f'prev_{metric_name}', default)

def store_current_metrics(metrics: Dict[str, float]):
    """Store current metrics for next comparison"""
    for name, value in metrics.items():
        st.session_state[f'prev_{name}'] = value

def show_accuracy_analytics():
    """Show comprehensive accuracy analysis"""
    st.markdown("### 🎯 Accuracy Analytics")
    
    performance_data = get_recent_performance_data(hours=168)  # 7 days
    
    if not performance_data:
        st.info("No accuracy data available.")
        return
    
    df = pd.DataFrame(performance_data)
    
    # Accuracy over time
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Accuracy Trends")
        
        # Create time series if timestamps available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Daily accuracy trend
            daily_accuracy = df.groupby(df['timestamp'].dt.date)['accuracy_score'].mean().reset_index()
            
            fig = px.line(daily_accuracy, x='timestamp', y='accuracy_score',
                         title='Daily Average Accuracy',
                         labels={'accuracy_score': 'Accuracy', 'timestamp': 'Date'})
            fig.update_yaxis(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Accuracy Distribution")
        
        fig = px.histogram(df, x='accuracy_score', nbins=20,
                          title='Accuracy Score Distribution',
                          labels={'accuracy_score': 'Accuracy Score'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Model comparison
    st.markdown("#### Accuracy by Model")
    
    if 'model' in df.columns:
        model_accuracy = df.groupby('model').agg({
            'accuracy_score': ['mean', 'std', 'count'],
            'cost': 'mean',
            'latency': 'mean'
        }).round(3)
        
        model_accuracy.columns = ['Avg Accuracy', 'Accuracy Std', 'Test Count', 'Avg Cost', 'Avg Latency']
        
        # Display table
        st.dataframe(model_accuracy, use_container_width=True)
        
        # Accuracy vs Cost scatter plot
        model_summary = df.groupby('model').agg({
            'accuracy_score': 'mean',
            'cost': 'mean',
            'latency': 'mean'
        }).reset_index()
        
        fig = px.scatter(model_summary, x='cost', y='accuracy_score',
                        size='latency', hover_name='model',
                        title='Accuracy vs Cost (bubble size = latency)',
                        labels={'cost': 'Average Cost ($)', 'accuracy_score': 'Average Accuracy'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Difficulty-based accuracy analysis
    if 'difficulty_level' in df.columns:
        st.markdown("#### Accuracy by Document Difficulty")
        
        difficulty_accuracy = df.groupby('difficulty_level')['accuracy_score'].agg(['mean', 'std', 'count']).reset_index()
        difficulty_accuracy.columns = ['Difficulty', 'Mean Accuracy', 'Std Dev', 'Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(difficulty_accuracy, x='Difficulty', y='Mean Accuracy',
                        error_y='Std Dev', title='Accuracy by Difficulty Level')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(difficulty_accuracy, use_container_width=True)
    
    # Accuracy insights
    st.markdown("#### 💡 Accuracy Insights")
    
    insights = generate_accuracy_insights(df)
    for insight in insights:
        st.markdown(f"• {insight}")

def generate_accuracy_insights(df: pd.DataFrame) -> List[str]:
    """Generate accuracy insights"""
    insights = []
    
    if len(df) == 0:
        return ["No data available for analysis."]
    
    avg_accuracy = df['accuracy_score'].mean()
    
    if avg_accuracy > 0.9:
        insights.append(f"🟢 **Excellent Performance**: {avg_accuracy:.1%} average accuracy")
    elif avg_accuracy > 0.8:
        insights.append(f"🟡 **Good Performance**: {avg_accuracy:.1%} average accuracy")
    else:
        insights.append(f"🔴 **Performance Alert**: {avg_accuracy:.1%} average accuracy - investigate low-performing models")
    
    # Model-specific insights
    if 'model' in df.columns:
        model_accuracy = df.groupby('model')['accuracy_score'].mean()
        best_model = model_accuracy.idxmax()
        worst_model = model_accuracy.idxmin()
        
        insights.append(f"🏆 **Best Model**: {best_model} ({model_accuracy[best_model]:.1%})")
        
        if model_accuracy[worst_model] < 0.7:
            insights.append(f"⚠️ **Underperforming**: {worst_model} ({model_accuracy[worst_model]:.1%}) needs attention")
    
    # Variability insights
    accuracy_std = df['accuracy_score'].std()
    if accuracy_std > 0.2:
        insights.append(f"📊 **High Variability**: {accuracy_std:.1%} accuracy standard deviation indicates inconsistent performance")
    
    return insights

def show_cost_intelligence():
    """Show advanced cost analysis and optimization"""
    st.markdown("### 💰 Cost Intelligence")
    
    performance_data = get_recent_performance_data(hours=168)  # 7 days
    
    if not performance_data:
        st.info("No cost data available.")
        return
    
    df = pd.DataFrame(performance_data)
    
    # Cost overview
    total_cost = df['cost'].sum()
    avg_cost_per_test = df['cost'].mean()
    cost_efficiency = df['accuracy_score'].sum() / total_cost if total_cost > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Cost (7d)", f"${total_cost:.3f}")
    
    with col2:
        st.metric("Avg Cost/Test", f"${avg_cost_per_test:.4f}")
    
    with col3:
        st.metric("Cost Efficiency", f"{cost_efficiency:.1f} acc/$ ×1000", 
                 help="Accuracy points per thousand dollars")
    
    # Cost trends and analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Cost Trends")
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            daily_cost = df.groupby(df['timestamp'].dt.date)['cost'].sum().reset_index()
            
            fig = px.line(daily_cost, x='timestamp', y='cost',
                         title='Daily Cost Trend', markers=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Cost Distribution by Model")
        
        if 'model' in df.columns:
            model_cost = df.groupby('model')['cost'].sum().reset_index()
            
            fig = px.pie(model_cost, values='cost', names='model',
                        title='Cost Share by Model')
            st.plotly_chart(fig, use_container_width=True)
    
    # Cost optimization analysis
    st.markdown("#### Cost Optimization Analysis")
    
    if 'model' in df.columns:
        # Calculate cost efficiency by model
        model_metrics = df.groupby('model').agg({
            'cost': ['sum', 'mean'],
            'accuracy_score': 'mean',
            'latency': 'mean'
        }).round(4)
        
        model_metrics.columns = ['Total Cost', 'Avg Cost', 'Avg Accuracy', 'Avg Latency']
        model_metrics['Cost Efficiency'] = (model_metrics['Avg Accuracy'] / model_metrics['Avg Cost']).round(1)
        model_metrics['Speed Efficiency'] = (model_metrics['Avg Accuracy'] / model_metrics['Avg Latency']).round(2)
        
        # Sort by cost efficiency
        model_metrics = model_metrics.sort_values('Cost Efficiency', ascending=False)
        
        st.dataframe(model_metrics, use_container_width=True)
        
        # Cost vs Performance matrix
        fig = go.Figure()
        
        for model in model_metrics.index:
            fig.add_trace(go.Scatter(
                x=[model_metrics.loc[model, 'Avg Cost']],
                y=[model_metrics.loc[model, 'Avg Accuracy']],
                mode='markers+text',
                text=[model],
                textposition="top center",
                name=model,
                marker=dict(
                    size=model_metrics.loc[model, 'Avg Latency'] * 3,  # Size by latency
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            title="Cost vs Accuracy Matrix (bubble size = latency)",
            xaxis_title="Average Cost ($)",
            yaxis_title="Average Accuracy",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost forecasting
    st.markdown("#### Cost Forecasting")
    
    if len(df) > 7:
        # Simple linear forecast for next 7 days
        if 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            daily_cost = df_sorted.groupby(df_sorted['timestamp'].dt.date)['cost'].sum()
            
            # Calculate trend
            x = np.arange(len(daily_cost))
            y = daily_cost.values
            
            if len(y) > 1:
                trend = np.polyfit(x, y, 1)[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    current_daily_avg = daily_cost.mean()
                    forecast_7d = current_daily_avg * 7 + trend * 7 * 3.5  # Trend effect
                    
                    st.metric("7-Day Forecast", f"${forecast_7d:.3f}",
                             delta=f"${forecast_7d - daily_cost.sum():.3f}")
                
                with col2:
                    trend_direction = "📈 Increasing" if trend > 0 else "📉 Decreasing"
                    st.metric("Cost Trend", trend_direction,
                             delta=f"${trend:.4f}/day")
    
    # Cost optimization recommendations
    st.markdown("#### 💡 Cost Optimization Recommendations")
    
    recommendations = generate_cost_recommendations(df)
    for rec in recommendations:
        st.markdown(f"• {rec}")

def generate_cost_recommendations(df: pd.DataFrame) -> List[str]:
    """Generate cost optimization recommendations"""
    recommendations = []
    
    if len(df) == 0:
        return ["No data available for cost analysis."]
    
    # Model efficiency analysis
    if 'model' in df.columns:
        model_efficiency = df.groupby('model').agg({
            'cost': 'mean',
            'accuracy_score': 'mean'
        })
        model_efficiency['efficiency'] = model_efficiency['accuracy_score'] / model_efficiency['cost']
        
        best_efficiency = model_efficiency['efficiency'].idxmax()
        worst_efficiency = model_efficiency['efficiency'].idxmin()
        
        recommendations.append(f"🎯 **Most Efficient**: Use {best_efficiency} for cost-effective processing")
        
        if model_efficiency.loc[worst_efficiency, 'efficiency'] < model_efficiency['efficiency'].median():
            recommendations.append(f"💸 **Cost Alert**: {worst_efficiency} shows poor cost efficiency - consider alternatives")
    
    # Usage pattern recommendations
    avg_cost = df['cost'].mean()
    if avg_cost > 0.01:
        recommendations.append("💰 **High Cost Alert**: Average cost per test > $0.01 - consider using smaller models for simple documents")
    
    # Difficulty-based recommendations
    if 'difficulty_level' in df.columns:
        difficulty_cost = df.groupby('difficulty_level')['cost'].mean()
        
        if 'Easy' in difficulty_cost.index and difficulty_cost['Easy'] > 0.005:
            recommendations.append("⚡ **Easy Documents**: Consider using faster, cheaper models for easy documents")
    
    return recommendations

def show_latency_analysis():
    """Show latency analysis and performance optimization"""
    st.markdown("### ⚡ Latency Analysis")
    
    performance_data = get_recent_performance_data(hours=168)
    
    if not performance_data:
        st.info("No latency data available.")
        return
    
    df = pd.DataFrame(performance_data)
    
    # Latency overview
    avg_latency = df['latency'].mean()
    p95_latency = df['latency'].quantile(0.95)
    min_latency = df['latency'].min()
    max_latency = df['latency'].max()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Latency", f"{avg_latency:.1f}s")
    
    with col2:
        st.metric("P95 Latency", f"{p95_latency:.1f}s")
    
    with col3:
        st.metric("Min Latency", f"{min_latency:.1f}s")
    
    with col4:
        st.metric("Max Latency", f"{max_latency:.1f}s")
    
    # Latency distribution and trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Latency Distribution")
        
        fig = px.histogram(df, x='latency', nbins=20,
                          title='Processing Time Distribution',
                          labels={'latency': 'Latency (seconds)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Latency by Model")
        
        if 'model' in df.columns:
            model_latency = df.groupby('model')['latency'].agg(['mean', 'std']).reset_index()
            model_latency.columns = ['Model', 'Mean Latency', 'Std Dev']
            
            fig = px.bar(model_latency, x='Model', y='Mean Latency',
                        error_y='Std Dev', title='Average Latency by Model')
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Performance correlation analysis
    st.markdown("#### Performance Correlation Analysis")
    
    # Create correlation matrix
    numeric_cols = ['latency', 'cost', 'accuracy_score']
    if all(col in df.columns for col in numeric_cols):
        correlation_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                       title="Performance Metrics Correlation")
        st.plotly_chart(fig, use_container_width=True)
    
    # Latency vs Accuracy scatter
    if 'accuracy_score' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(df, x='latency', y='accuracy_score',
                           color='model' if 'model' in df.columns else None,
                           title='Latency vs Accuracy',
                           labels={'latency': 'Latency (s)', 'accuracy_score': 'Accuracy'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Efficiency score (accuracy per second)
            df['efficiency'] = df['accuracy_score'] / df['latency']
            
            if 'model' in df.columns:
                model_efficiency = df.groupby('model')['efficiency'].mean().reset_index()
                model_efficiency = model_efficiency.sort_values('efficiency', ascending=False)
                
                fig = px.bar(model_efficiency, x='efficiency', y='model',
                           orientation='h', title='Processing Efficiency (Accuracy/Second)')
                st.plotly_chart(fig, use_container_width=True)
    
    # Latency insights
    st.markdown("#### ⚡ Latency Insights")
    
    insights = generate_latency_insights(df)
    for insight in insights:
        st.markdown(f"• {insight}")

def generate_latency_insights(df: pd.DataFrame) -> List[str]:
    """Generate latency optimization insights"""
    insights = []
    
    if len(df) == 0:
        return ["No latency data available."]
    
    avg_latency = df['latency'].mean()
    
    if avg_latency > 15:
        insights.append(f"🔴 **High Latency Alert**: {avg_latency:.1f}s average - consider optimization")
    elif avg_latency > 10:
        insights.append(f"🟡 **Moderate Latency**: {avg_latency:.1f}s average - room for improvement")
    else:
        insights.append(f"🟢 **Good Performance**: {avg_latency:.1f}s average latency")
    
    # Model performance
    if 'model' in df.columns:
        model_latency = df.groupby('model')['latency'].mean()
        fastest_model = model_latency.idxmin()
        slowest_model = model_latency.idxmax()
        
        insights.append(f"⚡ **Fastest Model**: {fastest_model} ({model_latency[fastest_model]:.1f}s)")
        
        if model_latency[slowest_model] > model_latency.median() * 2:
            insights.append(f"🐌 **Performance Alert**: {slowest_model} is significantly slower ({model_latency[slowest_model]:.1f}s)")
    
    # Efficiency insights
    if 'accuracy_score' in df.columns:
        df['efficiency'] = df['accuracy_score'] / df['latency']
        avg_efficiency = df['efficiency'].mean()
        
        insights.append(f"📊 **Processing Efficiency**: {avg_efficiency:.2f} accuracy points per second")
    
    return insights

def show_adaptability_metrics():
    """Show adaptability and cross-domain performance"""
    st.markdown("### 🔄 Adaptability Metrics")
    
    performance_data = get_recent_performance_data(hours=168)
    
    if not performance_data:
        st.info("No adaptability data available.")
        return
    
    df = pd.DataFrame(performance_data)
    
    # Overall adaptability score
    adaptability = calculate_adaptability_score(performance_data)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Adaptability", f"{adaptability:.1%}")
    
    with col2:
        consistency_score = 1 - df['accuracy_score'].std() if len(df) > 1 else 1.0
        st.metric("Consistency Score", f"{consistency_score:.1%}")
    
    with col3:
        robustness_score = (df['accuracy_score'] > 0.7).mean() if len(df) > 0 else 0.0
        st.metric("Robustness (>70%)", f"{robustness_score:.1%}")
    
    # Adaptability by dimension
    if 'model' in df.columns and 'difficulty_level' in df.columns:
        st.markdown("#### Cross-Domain Performance Matrix")
        
        # Create performance matrix
        performance_matrix = df.groupby(['model', 'difficulty_level'])['accuracy_score'].mean().unstack(fill_value=0)
        
        if not performance_matrix.empty:
            fig = px.imshow(performance_matrix, 
                           title="Model Performance Across Difficulty Levels",
                           color_continuous_scale='RdYlGn',
                           aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            
            # Variance analysis
            st.markdown("#### Performance Variance Analysis")
            
            model_variance = performance_matrix.var(axis=1).reset_index()
            model_variance.columns = ['Model', 'Performance Variance']
            model_variance = model_variance.sort_values('Performance Variance')
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(model_variance, x='Performance Variance', y='Model',
                           orientation='h', title='Performance Variance by Model (Lower = More Adaptable)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(model_variance, use_container_width=True)
    
    # Difficulty progression analysis
    if 'difficulty_level' in df.columns:
        st.markdown("#### Performance by Difficulty Level")
        
        difficulty_order = ['Easy', 'Medium', 'Hard', 'Very Hard']
        difficulty_performance = df.groupby('difficulty_level').agg({
            'accuracy_score': ['mean', 'std', 'count']
        }).round(3)
        
        difficulty_performance.columns = ['Mean Accuracy', 'Std Dev', 'Count']
        difficulty_performance = difficulty_performance.reindex(
            [d for d in difficulty_order if d in difficulty_performance.index]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(x=difficulty_performance.index, y=difficulty_performance['Mean Accuracy'],
                         title='Accuracy Drop-off by Difficulty',
                         markers=True)
            fig.update_layout(xaxis_title='Difficulty Level', yaxis_title='Mean Accuracy')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(difficulty_performance, use_container_width=True)
    
    # Adaptability recommendations
    st.markdown("#### 🔄 Adaptability Insights")
    
    insights = generate_adaptability_insights(df)
    for insight in insights:
        st.markdown(f"• {insight}")

def generate_adaptability_insights(df: pd.DataFrame) -> List[str]:
    """Generate adaptability insights"""
    insights = []
    
    if len(df) == 0:
        return ["No adaptability data available."]
    
    # Overall adaptability assessment
    adaptability = calculate_adaptability_score(df.to_dict('records'))
    
    if adaptability > 0.8:
        insights.append(f"🟢 **High Adaptability**: {adaptability:.1%} - models perform consistently across domains")
    elif adaptability > 0.6:
        insights.append(f"🟡 **Moderate Adaptability**: {adaptability:.1%} - some performance variation across domains")
    else:
        insights.append(f"🔴 **Low Adaptability**: {adaptability:.1%} - significant performance drops in challenging domains")
    
    # Model-specific insights
    if 'model' in df.columns and 'difficulty_level' in df.columns:
        model_adaptability = {}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            difficulty_scores = model_data.groupby('difficulty_level')['accuracy_score'].mean()
            
            if len(difficulty_scores) > 1:
                model_variance = difficulty_scores.std()
                model_adaptability[model] = 1 - min(model_variance, 1.0)
        
        if model_adaptability:
            most_adaptable = max(model_adaptability.keys(), key=lambda k: model_adaptability[k])
            least_adaptable = min(model_adaptability.keys(), key=lambda k: model_adaptability[k])
            
            insights.append(f"🏆 **Most Adaptable**: {most_adaptable} ({model_adaptability[most_adaptable]:.1%})")
            
            if model_adaptability[least_adaptable] < 0.6:
                insights.append(f"⚠️ **Domain Sensitivity**: {least_adaptable} shows high variance across difficulty levels")
    
    return insights

def show_predictive_insights():
    """Show predictive analytics and future performance insights"""
    st.markdown("### 📈 Predictive Insights")
    
    performance_data = get_recent_performance_data(hours=168)
    
    if not performance_data:
        st.info("Insufficient data for predictive analysis.")
        return
    
    df = pd.DataFrame(performance_data)
    
    if len(df) < 10:
        st.warning("Need at least 10 data points for reliable predictions.")
        return
    
    # Trend analysis
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Performance trends
        st.markdown("#### Performance Trend Analysis")
        
        # Calculate rolling averages
        df['accuracy_trend'] = df['accuracy_score'].rolling(window=5, min_periods=1).mean()
        df['cost_trend'] = df['cost'].rolling(window=5, min_periods=1).mean()
        df['latency_trend'] = df['latency'].rolling(window=5, min_periods=1).mean()
        
        # Create subplot for trends
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Accuracy Trend', 'Cost Trend', 'Latency Trend'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['accuracy_trend'], mode='lines', name='Accuracy'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['cost_trend'], mode='lines', name='Cost'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['latency_trend'], mode='lines', name='Latency'),
            row=3, col=1
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Performance Trends (5-point rolling average)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictive modeling
    st.markdown("#### Performance Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 24-Hour Forecast")
        
        # Simple linear prediction for next 24 hours
        predictions = generate_performance_predictions(df)
        
        for metric, prediction in predictions.items():
            current_value = df[metric].iloc[-1] if len(df) > 0 else 0
            change = prediction - current_value
            change_pct = (change / current_value * 100) if current_value != 0 else 0
            
            st.metric(
                f"Predicted {metric.replace('_', ' ').title()}",
                f"{prediction:.3f}",
                delta=f"{change:+.3f} ({change_pct:+.1f}%)"
            )
    
    with col2:
        st.markdown("##### Model Recommendations")
        
        recommendations = generate_predictive_recommendations(df)
        for rec in recommendations:
            st.markdown(f"• {rec}")
    
    # Anomaly detection
    st.markdown("#### Anomaly Detection")
    
    anomalies = detect_performance_anomalies(df)
    
    if anomalies:
        st.warning(f"⚠️ {len(anomalies)} potential anomalies detected:")
        
        for anomaly in anomalies:
            st.markdown(f"• **{anomaly['type']}**: {anomaly['description']}")
    else:
        st.success("✅ No performance anomalies detected")
    
    # Performance optimization suggestions
    st.markdown("#### 🚀 Optimization Opportunities")
    
    optimizations = generate_optimization_suggestions(df)
    for opt in optimizations:
        st.markdown(f"• {opt}")

def generate_performance_predictions(df: pd.DataFrame) -> Dict[str, float]:
    """Generate simple performance predictions"""
    predictions = {}
    
    metrics = ['accuracy_score', 'cost', 'latency']
    
    for metric in metrics:
        if metric in df.columns and len(df) > 5:
            # Simple linear trend prediction
            values = df[metric].dropna()
            if len(values) > 1:
                x = np.arange(len(values))
                y = values.values
                
                # Fit linear trend
                trend = np.polyfit(x, y, 1)[0]
                
                # Predict next value
                predictions[metric] = values.iloc[-1] + trend
            else:
                predictions[metric] = values.iloc[-1] if len(values) > 0 else 0
        else:
            predictions[metric] = 0
    
    return predictions

def generate_predictive_recommendations(df: pd.DataFrame) -> List[str]:
    """Generate recommendations based on trends"""
    recommendations = []
    
    if len(df) < 5:
        return ["Need more data for trend analysis"]
    
    # Accuracy trend
    if 'accuracy_score' in df.columns:
        recent_accuracy = df['accuracy_score'].tail(5).mean()
        older_accuracy = df['accuracy_score'].head(5).mean()
        
        if recent_accuracy < older_accuracy * 0.95:
            recommendations.append("📉 **Accuracy Declining**: Monitor model performance and consider retraining")
        elif recent_accuracy > older_accuracy * 1.05:
            recommendations.append("📈 **Accuracy Improving**: Current optimization strategies are working")
    
    # Cost trend
    if 'cost' in df.columns:
        recent_cost = df['cost'].tail(5).mean()
        older_cost = df['cost'].head(5).mean()
        
        if recent_cost > older_cost * 1.1:
            recommendations.append("💸 **Costs Rising**: Review model selection and usage patterns")
        elif recent_cost < older_cost * 0.9:
            recommendations.append("💰 **Costs Decreasing**: Cost optimization efforts are effective")
    
    return recommendations

def detect_performance_anomalies(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Detect performance anomalies"""
    anomalies = []
    
    if len(df) < 10:
        return anomalies
    
    # Statistical anomaly detection using IQR method
    for metric in ['accuracy_score', 'cost', 'latency']:
        if metric in df.columns:
            Q1 = df[metric].quantile(0.25)
            Q3 = df[metric].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[metric] < lower_bound) | (df[metric] > upper_bound)]
            
            if len(outliers) > 0:
                if metric == 'accuracy_score':
                    anomalies.append({
                        'type': 'Accuracy Anomaly',
                        'description': f"{len(outliers)} accuracy measurements outside normal range"
                    })
                elif metric == 'cost':
                    anomalies.append({
                        'type': 'Cost Anomaly',
                        'description': f"{len(outliers)} cost measurements unusually high/low"
                    })
                elif metric == 'latency':
                    anomalies.append({
                        'type': 'Latency Anomaly',
                        'description': f"{len(outliers)} processing times outside normal range"
                    })
    
    return anomalies

def generate_optimization_suggestions(df: pd.DataFrame) -> List[str]:
    """Generate optimization suggestions"""
    suggestions = []
    
    if len(df) == 0:
        return ["No data available for optimization analysis"]
    
    # Model usage optimization
    if 'model' in df.columns:
        model_performance = df.groupby('model').agg({
            'accuracy_score': 'mean',
            'cost': 'mean',
            'latency': 'mean'
        })
        
        # Find underutilized high-performance models
        high_accuracy_models = model_performance[model_performance['accuracy_score'] > model_performance['accuracy_score'].quantile(0.8)]
        
        if len(high_accuracy_models) > 0:
            best_model = high_accuracy_models['cost'].idxmin()
            suggestions.append(f"🎯 **Optimize Model Selection**: Consider using {best_model} more frequently for better cost-performance ratio")
    
    # Difficulty-based optimization
    if 'difficulty_level' in df.columns:
        easy_docs = df[df['difficulty_level'] == 'Easy']
        
        if len(easy_docs) > 0 and easy_docs['cost'].mean() > 0.005:
            suggestions.append("⚡ **Easy Document Optimization**: Use cheaper models for easy documents to reduce costs")
    
    # Performance consistency
    if 'accuracy_score' in df.columns:
        accuracy_variance = df['accuracy_score'].std()
        
        if accuracy_variance > 0.2:
            suggestions.append("📊 **Consistency Improvement**: High accuracy variance detected - implement ensemble methods for more consistent results")
    
    return suggestions

if __name__ == "__main__":
    show_page()