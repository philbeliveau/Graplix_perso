"""
LLM API Status Dashboard

This module provides a comprehensive view of all available LLM APIs,
their status, costs, and configuration.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

try:
    from llm import llm_integration, MULTI_LLM_AVAILABLE
except ImportError:
    MULTI_LLM_AVAILABLE = False
    llm_integration = None


def show_page():
    """Display the LLM API Status page"""
    st.markdown("## 🤖 Multi-LLM API Integration Status")
    
    if not MULTI_LLM_AVAILABLE:
        st.error("Multi-LLM integration is not available. Please check the installation.")
        st.info("Make sure all required dependencies are installed and API keys are configured.")
        return
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "API Status", 
        "Available Models", 
        "Cost Analysis", 
        "Usage Limits",
        "System Health"
    ])
    
    with tab1:
        show_api_status()
    
    with tab2:
        show_available_models()
    
    with tab3:
        show_cost_analysis()
    
    with tab4:
        show_usage_limits()
    
    with tab5:
        show_system_health()


def show_api_status():
    """Show API key status and configuration"""
    st.markdown("### 🔑 API Key Status")
    
    # Get API key status
    api_status = llm_integration.get_api_key_status()
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Providers",
            api_status['total_providers'],
            help="Total number of supported LLM providers"
        )
    
    with col2:
        st.metric(
            "Available Providers",
            api_status['available_providers'],
            help="Number of providers with valid API keys"
        )
    
    with col3:
        availability_pct = api_status['availability_percentage']
        st.metric(
            "Availability",
            f"{availability_pct:.1f}%",
            delta=f"{availability_pct - 50:.1f}%" if availability_pct != 50 else None,
            help="Percentage of providers with configured API keys"
        )
    
    with col4:
        status_color = "🟢" if availability_pct > 70 else "🟡" if availability_pct > 30 else "🔴"
        st.metric(
            "Overall Status",
            f"{status_color} {'Good' if availability_pct > 70 else 'Limited' if availability_pct > 30 else 'Poor'}",
            help="Overall API configuration status"
        )
    
    # Provider details
    st.markdown("### Provider Details")
    
    provider_data = []
    for provider, details in api_status['provider_details'].items():
        provider_data.append({
            'Provider': provider.upper(),
            'Status': '✅' if details['available'] else '❌',
            'Configuration': details['status'],
            'Last Validated': details.get('last_validated', 'Never'),
            'Error': details.get('error', '-')
        })
    
    provider_df = pd.DataFrame(provider_data)
    
    # Use color coding for status
    def highlight_status(row):
        if row['Status'] == '✅':
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)
    
    styled_df = provider_df.style.apply(highlight_status, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Configuration instructions for missing providers
    if api_status['missing_list']:
        with st.expander("📝 Setup Instructions for Missing Providers"):
            st.markdown("Add the following to your `.env` file:")
            st.code("\n".join([
                line
                for provider in api_status['missing_list']
                for line in [
                    f"# {provider.upper()} API Key",
                    f"{llm_integration.api_key_manager.SUPPORTED_PROVIDERS[provider]['env_var']}=your-{provider}-api-key-here",
                    ""
                ]
            ]))
    
    # API Key validation
    st.markdown("### 🔍 API Key Validation")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Validate All Keys", type="primary"):
            with st.spinner("Validating API keys..."):
                validation_results = llm_integration.validate_all_api_keys(online=True)
                st.session_state['validation_results'] = validation_results
                st.session_state['validation_time'] = datetime.now()
    
    if 'validation_results' in st.session_state:
        st.info(f"Last validated: {st.session_state['validation_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        validation_data = []
        for provider, is_valid in st.session_state['validation_results'].items():
            validation_data.append({
                'Provider': provider.upper(),
                'Online Validation': '✅ Valid' if is_valid else '❌ Invalid',
                'Status': 'Working' if is_valid else 'Check API Key'
            })
        
        val_df = pd.DataFrame(validation_data)
        st.dataframe(val_df, use_container_width=True, hide_index=True)


def show_available_models():
    """Show available LLM models and their capabilities"""
    st.markdown("### 🤖 Available LLM Models")
    
    models = llm_integration.get_available_models()
    
    if not models:
        st.warning("No models available. Please configure API keys first.")
        return
    
    # Model comparison table
    model_data = []
    for model_key, model_info in models.items():
        model_data.append({
            'Provider': model_info.provider.upper(),
            'Model': model_info.display_name,
            'Vision': '✅' if model_info.supports_vision else '❌',
            'JSON': '✅' if model_info.supports_json else '⚠️',
            'Input Cost': f"${model_info.cost_per_1k_input_tokens:.4f}",
            'Output Cost': f"${model_info.cost_per_1k_output_tokens:.4f}",
            'Context Length': f"{model_info.max_context_length:,}",
            'Recommended For': ', '.join(model_info.recommended_for[:2])
        })
    
    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, use_container_width=True, hide_index=True)
    
    # Cost comparison chart
    st.markdown("### 💰 Cost Comparison")
    
    # Prepare data for cost comparison
    cost_data = []
    for model_key, model_info in models.items():
        cost_data.append({
            'Model': model_info.display_name,
            'Input Cost (per 1K tokens)': model_info.cost_per_1k_input_tokens,
            'Output Cost (per 1K tokens)': model_info.cost_per_1k_output_tokens,
            'Provider': model_info.provider.upper()
        })
    
    cost_df = pd.DataFrame(cost_data)
    
    # Create cost comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Input Cost',
        x=cost_df['Model'],
        y=cost_df['Input Cost (per 1K tokens)'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Output Cost',
        x=cost_df['Model'],
        y=cost_df['Output Cost (per 1K tokens)'],
        marker_color='darkblue'
    ))
    
    fig.update_layout(
        title='LLM Cost Comparison (per 1K tokens)',
        xaxis_title='Model',
        yaxis_title='Cost (USD)',
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model recommendations
    st.markdown("### 🎯 Model Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Best for Accuracy")
        accuracy_models = llm_integration.get_models_by_capability('high_accuracy')
        for model in accuracy_models[:3]:
            st.write(f"• {model.display_name}")
    
    with col2:
        st.markdown("#### Best for Cost")
        cost_models = llm_integration.get_models_by_capability('cost_effective')
        for model in cost_models[:3]:
            st.write(f"• {model.display_name}")


def show_cost_analysis():
    """Show cost tracking and analysis"""
    st.markdown("### 💵 Cost Analysis & Tracking")
    
    # Time period selector
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        time_period = st.selectbox(
            "Time Period",
            ["session", "daily", "monthly", "all"],
            format_func=lambda x: {
                "session": "Current Session",
                "daily": "Today",
                "monthly": "This Month",
                "all": "Last 30 Days"
            }[x]
        )
    
    # Get cost summary
    cost_summary = llm_integration.get_cost_summary(time_period)
    
    if 'error' in cost_summary:
        st.error(cost_summary['error'])
        return
    
    # Display metrics based on time period
    if time_period == "session":
        show_session_costs(cost_summary)
    elif time_period == "daily":
        show_daily_costs(cost_summary)
    elif time_period == "monthly":
        show_monthly_costs(cost_summary)
    else:
        show_detailed_analysis(cost_summary)


def show_session_costs(summary):
    """Show session-specific costs"""
    st.markdown(f"#### Session: {summary.get('session_id', 'Unknown')}")
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    stats = summary.get('summary', {})
    
    with col1:
        st.metric(
            "Total Requests",
            stats.get('total_requests') or 0
        )
    
    with col2:
        st.metric(
            "Total Cost",
            f"${(stats.get('total_estimated_cost') or 0):.4f}"
        )
    
    with col3:
        st.metric(
            "Total Tokens",
            f"{(stats.get('total_tokens') or 0):,}"
        )
    
    with col4:
        st.metric(
            "Success Rate",
            f"{(summary.get('success_rate') or 0):.1f}%"
        )
    
    # Provider breakdown
    if summary.get('by_provider'):
        st.markdown("#### Cost by Provider")
        provider_df = pd.DataFrame(summary['by_provider'])
        
        fig = px.pie(
            provider_df,
            values='cost',
            names='provider',
            title='Cost Distribution by Provider'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model breakdown
    if summary.get('by_model'):
        st.markdown("#### Usage by Model")
        model_df = pd.DataFrame(summary['by_model'])
        st.dataframe(model_df, use_container_width=True, hide_index=True)


def show_daily_costs(summary):
    """Show daily cost breakdown"""
    st.markdown(f"#### Daily Costs for {summary.get('date', 'Today')}")
    
    total_cost = summary.get('total_cost', 0)
    st.metric("Total Daily Cost", f"${total_cost:.4f}")
    
    if summary.get('by_provider'):
        provider_df = pd.DataFrame(summary['by_provider'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=provider_df['provider'],
                y=provider_df['provider_cost'],
                text=[f"${cost:.4f}" for cost in provider_df['provider_cost']],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Daily Cost by Provider',
            xaxis_title='Provider',
            yaxis_title='Cost (USD)'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_monthly_costs(summary):
    """Show monthly cost breakdown"""
    st.markdown(f"#### Monthly Costs for {summary.get('month')}/{summary.get('year')}")
    
    total_cost = summary.get('total_cost', 0)
    st.metric("Total Monthly Cost", f"${total_cost:.2f}")
    
    if summary.get('daily_breakdown'):
        daily_df = pd.DataFrame(summary['daily_breakdown'])
        
        fig = go.Figure(data=[
            go.Scatter(
                x=daily_df['date'],
                y=daily_df['daily_cost'],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            )
        ])
        
        fig.update_layout(
            title='Daily Cost Trend',
            xaxis_title='Date',
            yaxis_title='Cost (USD)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def show_detailed_analysis(analysis):
    """Show detailed cost analysis"""
    st.markdown(f"#### Analysis Period: {analysis.get('period', 'Last 30 Days')}")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Cost",
            f"${(analysis.get('total_cost') or 0):.2f}"
        )
    
    with col2:
        st.metric(
            "Total Requests",
            f"{(analysis.get('total_requests') or 0):,}"
        )
    
    with col3:
        st.metric(
            "Avg Cost per Request",
            f"${(analysis.get('avg_cost_per_request') or 0):.4f}"
        )
    
    # Daily trends
    if analysis.get('daily_trends'):
        st.markdown("#### Daily Trends")
        trends_df = pd.DataFrame(analysis['daily_trends'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trends_df['date'],
            y=trends_df['cost'],
            name='Cost',
            yaxis='y',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Bar(
            x=trends_df['date'],
            y=trends_df['requests'],
            name='Requests',
            yaxis='y2',
            opacity=0.3
        ))
        
        fig.update_layout(
            title='Daily Cost and Request Trends',
            xaxis_title='Date',
            yaxis=dict(title='Cost (USD)', side='left'),
            yaxis2=dict(title='Requests', side='right', overlaying='y'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Most expensive models
    if analysis.get('expensive_models'):
        st.markdown("#### Most Expensive Models")
        expensive_df = pd.DataFrame(analysis['expensive_models'])
        st.dataframe(expensive_df, use_container_width=True, hide_index=True)


def show_usage_limits():
    """Show and configure usage limits"""
    st.markdown("### 🚦 Usage Limits & Monitoring")
    
    # Current limits display
    st.markdown("#### Current Limits")
    
    providers = llm_integration.api_key_manager.get_available_providers()
    
    if not providers:
        st.warning("No providers available to set limits for.")
        return
    
    # Create a form for setting limits
    with st.form("usage_limits_form"):
        st.markdown("Set daily and monthly cost limits for each provider:")
        
        limits_data = []
        for provider in providers:
            col1, col2, col3 = st.columns([2, 2, 2])
            
            with col1:
                st.markdown(f"**{provider.upper()}**")
            
            with col2:
                daily_limit = st.number_input(
                    f"Daily Limit ($)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=10.0,
                    step=1.0,
                    key=f"daily_{provider}"
                )
            
            with col3:
                monthly_limit = st.number_input(
                    f"Monthly Limit ($)",
                    min_value=0.0,
                    max_value=10000.0,
                    value=100.0,
                    step=10.0,
                    key=f"monthly_{provider}"
                )
            
            limits_data.append({
                'provider': provider,
                'daily': daily_limit,
                'monthly': monthly_limit
            })
        
        if st.form_submit_button("Apply Limits", type="primary"):
            for limit in limits_data:
                llm_integration.set_usage_limits(
                    limit['provider'],
                    limit['daily'],
                    limit['monthly']
                )
            st.success("Usage limits updated successfully!")
    
    # Current usage vs limits
    st.markdown("#### Current Usage vs Limits")
    
    usage_data = []
    for provider in providers:
        limit_check = llm_integration.token_monitor.check_limits(provider) if llm_integration.token_monitor else {}
        
        usage_data.append({
            'Provider': provider.upper(),
            'Daily Usage': f"${(limit_check.get('daily_usage') or 0):.2f}",
            'Daily Limit': f"${limit_check.get('daily_limit', 'Not Set')}" if limit_check.get('daily_limit') else 'Not Set',
            'Monthly Usage': f"${(limit_check.get('monthly_usage') or 0):.2f}",
            'Monthly Limit': f"${limit_check.get('monthly_limit', 'Not Set')}" if limit_check.get('monthly_limit') else 'Not Set',
            'Alerts': len(limit_check.get('alerts', []))
        })
    
    usage_df = pd.DataFrame(usage_data)
    st.dataframe(usage_df, use_container_width=True, hide_index=True)
    
    # Display any alerts
    for provider in providers:
        limit_check = llm_integration.token_monitor.check_limits(provider) if llm_integration.token_monitor else {}
        alerts = limit_check.get('alerts', [])
        
        for alert in alerts:
            if alert['type'].endswith('_exceeded'):
                st.error(f"⚠️ {alert['message']}")
            else:
                st.warning(f"⚡ {alert['message']}")


def show_system_health():
    """Show overall system health and diagnostics"""
    st.markdown("### 🏥 System Health & Diagnostics")
    
    health = llm_integration.get_system_health()
    
    # Overall status
    status_color = {
        'healthy': '🟢',
        'warning': '🟡',
        'critical': '🔴'
    }.get(health['overall_status'], '⚪')
    
    st.markdown(f"### {status_color} Overall Status: {health['status_message']}")
    
    # System metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "API Keys",
            f"{health['api_keys']['available']}/{health['api_keys']['total']}",
            help="Available / Total API keys"
        )
    
    with col2:
        st.metric(
            "Available Models",
            health['models']['total_available'],
            help="Total number of available LLM models"
        )
    
    with col3:
        cost_status = "Enabled" if health['cost_tracking']['enabled'] else "Disabled"
        st.metric(
            "Cost Tracking",
            cost_status,
            help="Cost tracking system status"
        )
    
    # Models by provider
    st.markdown("#### Models by Provider")
    
    provider_model_data = [
        {'Provider': provider.upper(), 'Available Models': count}
        for provider, count in health['models']['by_provider'].items()
    ]
    
    if provider_model_data:
        provider_model_df = pd.DataFrame(provider_model_data)
        
        fig = px.bar(
            provider_model_df,
            x='Provider',
            y='Available Models',
            title='Available Models by Provider',
            color='Available Models',
            color_continuous_scale='Blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Real-time session stats
    if health['cost_tracking']['enabled'] and health['cost_tracking']['session_stats']:
        st.markdown("#### Current Session Statistics")
        
        session_stats = health['cost_tracking']['session_stats']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Requests", session_stats.get('total_requests') or 0)
        
        with col2:
            st.metric("Total Cost", f"${(session_stats.get('total_cost') or 0):.4f}")
        
        with col3:
            st.metric("Total Tokens", f"{(session_stats.get('total_tokens') or 0):,}")
        
        with col4:
            st.metric("Success Rate", f"{(session_stats.get('success_rate') or 0):.1f}%")
        
        # Recent activity
        if session_stats.get('recent_activity'):
            st.markdown("#### Recent Activity")
            
            activity_data = []
            for activity in session_stats['recent_activity']:
                activity_data.append({
                    'Time': activity.get('timestamp', 'Unknown'),
                    'Provider': (activity.get('provider') or 'Unknown').upper(),
                    'Model': activity.get('model', 'Unknown'),
                    'Cost': f"${(activity.get('cost') or 0):.4f}",
                    'Tokens': f"{(activity.get('tokens') or 0):,}",
                    'Status': '✅' if activity.get('success') else '❌'
                })
            
            activity_df = pd.DataFrame(activity_data)
            st.dataframe(activity_df, use_container_width=True, hide_index=True)
    
    # Export options
    st.markdown("#### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Usage Report (JSON)"):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"llm_usage_report_{timestamp}.json"
            if llm_integration.export_usage_report(output_path, format='json'):
                st.success(f"Report exported to {output_path}")
            else:
                st.error("Failed to export report")
    
    with col2:
        if st.button("Export API Status Report"):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report = llm_integration.api_key_manager.export_status_report()
            st.download_button(
                label="Download Report",
                data=report,
                file_name=f"api_status_report_{timestamp}.txt",
                mime="text/plain"
            )


# Export the show_page function
__all__ = ['show_page']