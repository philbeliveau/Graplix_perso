"""
Model Performance Analytics Dashboard

This page provides comprehensive analytics for multimodal LLM performance,
cost tracking, and model comparison across processing runs.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
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

from dashboard.utils import session_state, ui_components, auth, run_history
from llm.multimodal_llm_service import llm_service

def show_page():
    """Main model analytics dashboard"""
    st.markdown('<div class="section-header">📊 Model Performance Analytics</div>', 
                unsafe_allow_html=True)
    st.markdown("Comprehensive analytics for multimodal LLM performance, cost tracking, and model comparison.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Main layout
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview Dashboard",
        "🔍 Model Comparison", 
        "💰 Cost Analysis",
        "📊 Performance Trends"
    ])
    
    with tab1:
        show_overview_dashboard()
    
    with tab2:
        show_model_comparison()
    
    with tab3:
        show_cost_analysis()
    
    with tab4:
        show_performance_trends()

def show_overview_dashboard():
    """Show overview dashboard with key metrics"""
    st.markdown("### 📊 Performance Overview")
    
    # Time period selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        days_back = st.selectbox(
            "Time Period",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )
    
    with col2:
        st.info(f"📅 Showing data from the last {days_back} days")
    
    # Get recent runs
    recent_runs = run_history.run_history_manager.get_recent_runs(limit=100)
    
    if not recent_runs:
        st.info("No processing runs found. Start processing documents to see analytics.")
        return
    
    # Filter by time period
    cutoff_date = datetime.now() - timedelta(days=days_back)
    filtered_runs = [
        run for run in recent_runs 
        if datetime.fromisoformat(run['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None) >= cutoff_date
    ]
    
    if not filtered_runs:
        st.info(f"No runs found in the last {days_back} days.")
        return
    
    # Calculate overview metrics
    total_runs = len(filtered_runs)
    total_documents = sum(run['total_documents'] for run in filtered_runs)
    total_entities = sum(run['total_entities'] for run in filtered_runs)
    total_cost = sum(run['total_cost'] for run in filtered_runs)
    avg_processing_time = sum(run['total_processing_time'] for run in filtered_runs) / total_runs if total_runs > 0 else 0
    
    # Display key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Runs", total_runs)
    
    with col2:
        st.metric("Documents Processed", f"{total_documents:,}")
    
    with col3:
        st.metric("PII Entities Found", f"{total_entities:,}")
    
    with col4:
        st.metric("Total Cost", f"${total_cost:.4f}")
    
    with col5:
        st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
    
    # Create DataFrame for analysis
    df_runs = pd.DataFrame(filtered_runs)
    df_runs['timestamp'] = pd.to_datetime(df_runs['timestamp'])
    df_runs['date'] = df_runs['timestamp'].dt.date
    df_runs['cost_per_document'] = df_runs['total_cost'] / df_runs['total_documents']
    df_runs['entities_per_document'] = df_runs['total_entities'] / df_runs['total_documents']
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Daily Processing Volume")
        daily_stats = df_runs.groupby('date').agg({
            'total_documents': 'sum',
            'total_entities': 'sum'
        }).reset_index()
        
        fig = px.bar(daily_stats, x='date', y='total_documents',
                    title='Documents Processed per Day')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 Daily Cost Breakdown")
        daily_cost = df_runs.groupby('date')['total_cost'].sum().reset_index()
        
        fig = px.line(daily_cost, x='date', y='total_cost',
                     title='Daily Processing Costs', markers=True)
        fig.update_traces(line_color='#ff6b6b')
        st.plotly_chart(fig, use_container_width=True)
    
    # Model usage breakdown
    st.markdown("#### 🤖 Model Usage Distribution")
    model_usage = df_runs['model_used'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(values=model_usage.values, names=model_usage.index,
                    title='Runs by Model')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Model performance summary
        model_performance = df_runs.groupby('model_used').agg({
            'total_documents': 'sum',
            'total_entities': 'sum',
            'total_cost': 'sum',
            'cost_per_document': 'mean',
            'entities_per_document': 'mean'
        }).round(4)
        
        st.markdown("**Model Performance Summary:**")
        st.dataframe(model_performance, use_container_width=True)

def show_model_comparison():
    """Show detailed model comparison"""
    st.markdown("### 🔍 Model Comparison")
    
    # Time period selection
    days_back = st.slider("Days to analyze", 1, 90, 14)
    
    # Get model comparison data
    comparison_data = run_history.run_history_manager.get_model_comparison(days=days_back)
    
    if not comparison_data['models']:
        st.info("No model comparison data available for the selected period.")
        return
    
    # Display comparison table
    st.markdown(f"#### 📊 Model Performance (Last {days_back} days)")
    
    df_models = pd.DataFrame(comparison_data['models'])
    
    # Format the dataframe for display
    df_display = df_models.copy()
    df_display['total_cost'] = df_display['total_cost'].round(4)
    df_display['avg_processing_time'] = df_display['avg_processing_time'].round(2)
    df_display['avg_entities'] = df_display['avg_entities'].round(1)
    df_display['success_rate'] = (df_display['success_rate'] * 100).round(1)
    df_display['avg_confidence'] = (df_display['avg_confidence'] * 100).round(1)
    df_display['cost_per_document'] = df_display['cost_per_document'].round(4)
    
    # Rename columns for display
    df_display.columns = [
        'Model', 'Total Runs', 'Total Documents', 'Total Cost ($)', 
        'Avg Time (s)', 'Avg Entities', 'Success Rate (%)', 
        'Avg Confidence (%)', 'Cost per Doc ($)'
    ]
    
    st.dataframe(df_display, use_container_width=True)
    
    # Model comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚡ Processing Speed Comparison")
        fig = px.bar(df_models, x='model_used', y='avg_processing_time',
                    title='Average Processing Time by Model')
        fig.update_layout(xaxis_title="Model", yaxis_title="Time (seconds)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 Cost Efficiency Comparison")
        fig = px.bar(df_models, x='model_used', y='cost_per_document',
                    title='Cost per Document by Model')
        fig.update_layout(xaxis_title="Model", yaxis_title="Cost ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced comparison metrics
    st.markdown("#### 🎯 Quality vs Cost Analysis")
    
    # Create scatter plot: entities found vs cost
    fig = px.scatter(df_models, x='cost_per_document', y='avg_entities',
                    size='total_documents', hover_name='model_used',
                    title='Entities Found vs Cost per Document',
                    labels={'cost_per_document': 'Cost per Document ($)',
                           'avg_entities': 'Average Entities Found'})
    
    # Add trend line
    fig.add_traces(px.scatter(df_models, x='cost_per_document', y='avg_entities',
                             trendline='ols').data[1:])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model recommendation
    if len(df_models) > 1:
        st.markdown("#### 🏆 Model Recommendations")
        
        # Find best models for different criteria
        fastest_model = df_models.loc[df_models['avg_processing_time'].idxmin(), 'model_used']
        cheapest_model = df_models.loc[df_models['cost_per_document'].idxmin(), 'model_used']
        most_accurate = df_models.loc[df_models['avg_entities'].idxmax(), 'model_used']
        most_reliable = df_models.loc[df_models['success_rate'].idxmax(), 'model_used']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.success(f"⚡ **Fastest:** {fastest_model}")
        
        with col2:
            st.success(f"💰 **Most Cost-Effective:** {cheapest_model}")
        
        with col3:
            st.success(f"🎯 **Most Accurate:** {most_accurate}")
        
        with col4:
            st.success(f"🛡️ **Most Reliable:** {most_reliable}")

def show_cost_analysis():
    """Show detailed cost analysis"""
    st.markdown("### 💰 Cost Analysis")
    
    # Date range selection
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    # Get cost analysis data
    cost_data = run_history.run_history_manager.get_cost_analysis(
        start_date=datetime.combine(start_date, datetime.min.time()),
        end_date=datetime.combine(end_date, datetime.max.time())
    )
    
    if not cost_data['daily_breakdown']:
        st.info("No cost data available for the selected period.")
        return
    
    # Display summary metrics
    summary = cost_data['summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cost", f"${summary['total_cost']:.4f}")
    
    with col2:
        st.metric("Total API Calls", f"{summary['total_calls']:,}")
    
    with col3:
        st.metric("Avg Cost per Call", f"${summary['avg_cost_per_call']:.4f}")
    
    with col4:
        st.metric("Total Tokens", f"{summary['total_input_tokens'] + summary['total_output_tokens']:,}")
    
    # Cost breakdown charts
    df_costs = pd.DataFrame(cost_data['daily_breakdown'])
    
    if not df_costs.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Daily Cost Breakdown")
            daily_totals = df_costs.groupby('date')['total_cost'].sum().reset_index()
            daily_totals['date'] = pd.to_datetime(daily_totals['date'])
            
            fig = px.line(daily_totals, x='date', y='total_cost',
                         title='Daily Costs', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🤖 Cost by Model")
            model_costs = df_costs.groupby('model_name')['total_cost'].sum().reset_index()
            
            fig = px.pie(model_costs, values='total_cost', names='model_name',
                        title='Total Cost by Model')
            st.plotly_chart(fig, use_container_width=True)
        
        # Token usage analysis
        st.markdown("#### 🔤 Token Usage Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input vs Output tokens
            token_comparison = df_costs.groupby('model_name').agg({
                'total_input_tokens': 'sum',
                'total_output_tokens': 'sum'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Input Tokens', x=token_comparison['model_name'], 
                               y=token_comparison['total_input_tokens']))
            fig.add_trace(go.Bar(name='Output Tokens', x=token_comparison['model_name'], 
                               y=token_comparison['total_output_tokens']))
            
            fig.update_layout(barmode='stack', title='Token Usage by Model',
                            xaxis_title='Model', yaxis_title='Tokens')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cost efficiency
            df_costs['cost_per_token'] = df_costs['total_cost'] / (
                df_costs['total_input_tokens'] + df_costs['total_output_tokens']
            )
            
            model_efficiency = df_costs.groupby('model_name')['cost_per_token'].mean().reset_index()
            
            fig = px.bar(model_efficiency, x='model_name', y='cost_per_token',
                        title='Cost per Token by Model')
            fig.update_layout(xaxis_title='Model', yaxis_title='Cost per Token ($)')
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed cost table
        st.markdown("#### 📋 Detailed Cost Breakdown")
        
        # Aggregate by model and date
        cost_summary = df_costs.groupby(['model_name', 'date']).agg({
            'total_calls': 'sum',
            'total_cost': 'sum',
            'total_input_tokens': 'sum',
            'total_output_tokens': 'sum',
            'avg_cost_per_call': 'mean'
        }).round(4).reset_index()
        
        st.dataframe(cost_summary, use_container_width=True)

def show_performance_trends():
    """Show performance trends over time"""
    st.markdown("### 📊 Performance Trends")
    
    # Get performance history data
    performance_data = run_history.run_history_manager.get_model_performance_history(days=60)
    
    if not performance_data:
        st.info("No performance trend data available.")
        return
    
    df_performance = pd.DataFrame(performance_data)
    df_performance['date'] = pd.to_datetime(df_performance['date'])
    
    # Model selection for trend analysis
    available_models = df_performance['model_name'].unique()
    selected_models = st.multiselect(
        "Select models to compare",
        available_models,
        default=available_models[:3] if len(available_models) >= 3 else available_models
    )
    
    if not selected_models:
        st.warning("Please select at least one model to view trends.")
        return
    
    # Filter data
    df_filtered = df_performance[df_performance['model_name'].isin(selected_models)]
    
    # Create trend charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Processing Time Trends")
        fig = px.line(df_filtered, x='date', y='average_processing_time',
                     color='model_name', title='Average Processing Time Over Time',
                     markers=True)
        fig.update_layout(xaxis_title='Date', yaxis_title='Processing Time (s)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Entities Found Trends")
        fig = px.line(df_filtered, x='date', y='average_entities_per_doc',
                     color='model_name', title='Average Entities Found Over Time',
                     markers=True)
        fig.update_layout(xaxis_title='Date', yaxis_title='Entities per Document')
        st.plotly_chart(fig, use_container_width=True)
    
    # Success rate and cost trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✅ Success Rate Trends")
        fig = px.line(df_filtered, x='date', y='success_rate',
                     color='model_name', title='Success Rate Over Time',
                     markers=True)
        fig.update_layout(xaxis_title='Date', yaxis_title='Success Rate')
        fig.update_yaxis(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 💰 Cost Trends")
        fig = px.line(df_filtered, x='date', y='total_cost',
                     color='model_name', title='Daily Costs Over Time',
                     markers=True)
        fig.update_layout(xaxis_title='Date', yaxis_title='Total Cost ($)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance correlation analysis
    st.markdown("#### 🔗 Performance Correlations")
    
    # Calculate correlations between metrics
    correlation_metrics = ['total_documents', 'total_cost', 'average_processing_time', 
                          'average_entities_per_doc', 'success_rate']
    
    correlation_data = df_filtered[correlation_metrics].corr()
    
    # Create heatmap
    fig = px.imshow(correlation_data, text_auto=True, aspect="auto",
                   title="Performance Metrics Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance insights
    st.markdown("#### 💡 Performance Insights")
    
    if len(df_filtered) > 0:
        # Calculate recent vs historical performance
        recent_cutoff = df_filtered['date'].max() - timedelta(days=7)
        recent_data = df_filtered[df_filtered['date'] >= recent_cutoff]
        historical_data = df_filtered[df_filtered['date'] < recent_cutoff]
        
        if len(recent_data) > 0 and len(historical_data) > 0:
            for model in selected_models:
                model_recent = recent_data[recent_data['model_name'] == model]
                model_historical = historical_data[historical_data['model_name'] == model]
                
                if len(model_recent) > 0 and len(model_historical) > 0:
                    recent_avg_time = model_recent['average_processing_time'].mean()
                    historical_avg_time = model_historical['average_processing_time'].mean()
                    
                    recent_avg_entities = model_recent['average_entities_per_doc'].mean()
                    historical_avg_entities = model_historical['average_entities_per_doc'].mean()
                    
                    time_change = (recent_avg_time - historical_avg_time) / historical_avg_time * 100
                    entities_change = (recent_avg_entities - historical_avg_entities) / historical_avg_entities * 100
                    
                    if abs(time_change) > 10:  # Significant change
                        if time_change > 0:
                            st.warning(f"⚠️ {model}: Processing time increased by {time_change:.1f}% recently")
                        else:
                            st.success(f"✅ {model}: Processing time improved by {abs(time_change):.1f}% recently")
                    
                    if abs(entities_change) > 15:  # Significant change
                        if entities_change > 0:
                            st.success(f"📈 {model}: Entity detection improved by {entities_change:.1f}% recently")
                        else:
                            st.warning(f"📉 {model}: Entity detection decreased by {abs(entities_change):.1f}% recently")

if __name__ == "__main__":
    show_page()