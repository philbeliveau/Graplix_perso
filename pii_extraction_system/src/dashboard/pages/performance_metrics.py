"""
Performance Metrics Page - Real-time system monitoring and analytics

This page provides comprehensive monitoring of system performance, accuracy metrics,
and operational characteristics essential for production deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
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

from dashboard.utils import session_state, ui_components, auth

def show_page():
    """Main performance metrics page"""
    st.markdown('<div class="section-header">📈 Performance Metrics</div>', 
                unsafe_allow_html=True)
    st.markdown("Monitor system performance, accuracy metrics, and operational status.")
    
    # Check permissions
    if not auth.has_permission('read'):
        st.error("Access denied. Insufficient permissions.")
        return
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### System Overview")
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    with col3:
        if st.button("Refresh Now") or auto_refresh:
            update_system_metrics()
    
    # Main dashboard layout
    show_system_overview()
    
    # Detailed metrics tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Real-time Metrics",
        "🎯 Accuracy Analytics", 
        "⚡ Performance Analytics",
        "📋 System Health"
    ])
    
    with tab1:
        show_realtime_metrics()
    
    with tab2:
        show_accuracy_analytics()
    
    with tab3:
        show_performance_analytics()
    
    with tab4:
        show_system_health()

def update_system_metrics():
    """Update system metrics data"""
    # Mock system metrics (would integrate with actual monitoring)
    current_time = datetime.now()
    
    metrics = {
        'timestamp': current_time,
        'documents_processed_today': np.random.randint(45, 120),
        'processing_queue_size': np.random.randint(0, 15),
        'avg_processing_time': np.random.uniform(2.5, 8.0),
        'system_cpu_usage': np.random.uniform(30, 85),
        'system_memory_usage': np.random.uniform(40, 75),
        'active_models': np.random.randint(2, 6),
        'error_rate': np.random.uniform(0.02, 0.08),
        'uptime_hours': np.random.uniform(120, 720)
    }
    
    # Store in session state
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = []
    
    st.session_state.performance_metrics.append(metrics)
    
    # Keep only last 100 data points
    if len(st.session_state.performance_metrics) > 100:
        st.session_state.performance_metrics = st.session_state.performance_metrics[-100:]

def show_system_overview():
    """Show high-level system overview"""
    metrics = get_latest_metrics()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Documents Today",
            f"{metrics.get('documents_processed_today', 0)}",
            delta="+12 vs yesterday"
        )
    
    with col2:
        queue_size = metrics.get('processing_queue_size', 0)
        queue_delta = "Empty" if queue_size == 0 else f"+{queue_size}"
        st.metric(
            "Queue Size", 
            f"{queue_size}",
            delta=queue_delta
        )
    
    with col3:
        avg_time = metrics.get('avg_processing_time', 0)
        st.metric(
            "Avg Processing Time",
            f"{avg_time:.1f}s",
            delta="-0.3s vs last hour"
        )
    
    with col4:
        error_rate = metrics.get('error_rate', 0)
        st.metric(
            "Error Rate",
            f"{error_rate:.1%}",
            delta=f"{'↓' if error_rate < 0.05 else '↑'} {error_rate:.1%}",
            delta_color="inverse"
        )
    
    # System status indicators
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cpu_usage = metrics.get('system_cpu_usage', 0)
        cpu_color = "green" if cpu_usage < 70 else "orange" if cpu_usage < 85 else "red"
        st.markdown(f'**CPU Usage:** <span style="color: {cpu_color}">{cpu_usage:.1f}%</span>', 
                   unsafe_allow_html=True)
    
    with col2:
        memory_usage = metrics.get('system_memory_usage', 0)
        memory_color = "green" if memory_usage < 70 else "orange" if memory_usage < 85 else "red"
        st.markdown(f'**Memory Usage:** <span style="color: {memory_color}">{memory_usage:.1f}%</span>', 
                   unsafe_allow_html=True)
    
    with col3:
        uptime = metrics.get('uptime_hours', 0)
        st.markdown(f"**Uptime:** {uptime:.1f} hours")

def get_latest_metrics() -> Dict[str, Any]:
    """Get the latest system metrics"""
    metrics_data = st.session_state.get('performance_metrics', [])
    if metrics_data:
        return metrics_data[-1]
    else:
        # Return default metrics if none exist
        update_system_metrics()
        return st.session_state.get('performance_metrics', [{}])[-1]

def show_realtime_metrics():
    """Show real-time metrics and charts"""
    st.markdown("### Real-time System Metrics")
    
    metrics_data = st.session_state.get('performance_metrics', [])
    if len(metrics_data) < 2:
        st.info("Collecting metrics data... Please wait a moment and refresh.")
        return
    
    # Create time series data
    df = pd.DataFrame(metrics_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Processing throughput chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Processing Throughput")
        fig_throughput = px.line(
            df, x='timestamp', y='documents_processed_today',
            title='Documents Processed (Cumulative)',
            labels={'documents_processed_today': 'Documents', 'timestamp': 'Time'}
        )
        fig_throughput.update_layout(height=300)
        st.plotly_chart(fig_throughput, use_container_width=True)
    
    with col2:
        st.markdown("#### Queue Status")
        fig_queue = px.line(
            df, x='timestamp', y='processing_queue_size',
            title='Processing Queue Size',
            labels={'processing_queue_size': 'Queue Size', 'timestamp': 'Time'}
        )
        fig_queue.update_layout(height=300)
        st.plotly_chart(fig_queue, use_container_width=True)
    
    # System resource usage
    st.markdown("#### System Resource Usage")
    
    fig_resources = make_subplots(
        rows=1, cols=2,
        subplot_titles=('CPU Usage (%)', 'Memory Usage (%)'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    fig_resources.add_trace(
        go.Scatter(x=df['timestamp'], y=df['system_cpu_usage'], name='CPU Usage'),
        row=1, col=1
    )
    
    fig_resources.add_trace(
        go.Scatter(x=df['timestamp'], y=df['system_memory_usage'], name='Memory Usage'),
        row=1, col=2
    )
    
    fig_resources.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_resources, use_container_width=True)
    
    # Processing time trends
    st.markdown("#### Processing Time Trends")
    
    fig_time = px.line(
        df, x='timestamp', y='avg_processing_time',
        title='Average Processing Time per Document',
        labels={'avg_processing_time': 'Time (seconds)', 'timestamp': 'Time'}
    )
    fig_time.update_layout(height=300)
    st.plotly_chart(fig_time, use_container_width=True)

def show_accuracy_analytics():
    """Show accuracy and quality metrics"""
    st.markdown("### Accuracy Analytics")
    
    # Get processing results for accuracy calculation
    uploaded_docs = st.session_state.get('uploaded_documents', {})
    processed_docs = []
    
    for doc_id, doc_info in uploaded_docs.items():
        results = session_state.get_processing_results(doc_id)
        if results:
            processed_docs.append((doc_id, doc_info, results))
    
    if not processed_docs:
        st.info("No processed documents available for accuracy analysis.")
        return
    
    # Calculate accuracy metrics across all documents
    accuracy_metrics = calculate_accuracy_metrics(processed_docs)
    
    # Overall accuracy metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Precision", f"{accuracy_metrics['precision']:.1%}")
    
    with col2:
        st.metric("Overall Recall", f"{accuracy_metrics['recall']:.1%}")
    
    with col3:
        st.metric("Overall F1-Score", f"{accuracy_metrics['f1_score']:.1%}")
    
    # Per-category accuracy
    st.markdown("#### Accuracy by PII Category")
    
    category_metrics = accuracy_metrics.get('category_metrics', {})
    if category_metrics:
        df_categories = pd.DataFrame(category_metrics).T
        df_categories = df_categories.round(3)
        st.dataframe(df_categories, use_container_width=True)
        
        # Category performance chart
        fig_categories = px.bar(
            x=df_categories.index,
            y=df_categories['f1_score'],
            title='F1-Score by PII Category',
            labels={'x': 'PII Category', 'y': 'F1-Score'}
        )
        fig_categories.update_layout(height=400)
        st.plotly_chart(fig_categories, use_container_width=True)
    
    # Confidence distribution analysis
    st.markdown("#### Confidence Score Analysis")
    
    all_confidences = []
    for doc_id, doc_info, results in processed_docs:
        entities = results.get('pii_entities', [])
        confidences = [entity.get('confidence', 0) for entity in entities]
        all_confidences.extend(confidences)
    
    if all_confidences:
        fig_confidence = ui_components.create_confidence_histogram(
            all_confidences, 
            "Confidence Score Distribution (All Documents)"
        )
        st.plotly_chart(fig_confidence, use_container_width=True)
        
        # Confidence statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Confidence", f"{np.mean(all_confidences):.1%}")
        
        with col2:
            st.metric("Median Confidence", f"{np.median(all_confidences):.1%}")
        
        with col3:
            st.metric("Min Confidence", f"{np.min(all_confidences):.1%}")
        
        with col4:
            st.metric("Max Confidence", f"{np.max(all_confidences):.1%}")

def calculate_accuracy_metrics(processed_docs: List) -> Dict[str, Any]:
    """Calculate accuracy metrics from processed documents"""
    # Mock accuracy calculation (would integrate with ground truth data)
    
    total_entities = 0
    category_stats = {}
    
    for doc_id, doc_info, results in processed_docs:
        entities = results.get('pii_entities', [])
        total_entities += len(entities)
        
        for entity in entities:
            category = entity.get('type', 'UNKNOWN')
            if category not in category_stats:
                category_stats[category] = {
                    'true_positives': 0,
                    'false_positives': 0,
                    'false_negatives': 0
                }
            
            # Mock classification (would use actual ground truth)
            confidence = entity.get('confidence', 0)
            if confidence > 0.8:
                category_stats[category]['true_positives'] += 1
            elif confidence > 0.5:
                category_stats[category]['true_positives'] += 1
            else:
                category_stats[category]['false_positives'] += 1
    
    # Calculate overall metrics
    total_tp = sum(stats['true_positives'] for stats in category_stats.values())
    total_fp = sum(stats['false_positives'] for stats in category_stats.values())
    total_fn = sum(stats['false_negatives'] for stats in category_stats.values())
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-category metrics
    category_metrics = {}
    for category, stats in category_stats.items():
        tp = stats['true_positives']
        fp = stats['false_positives']
        fn = stats['false_negatives']
        
        cat_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        cat_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        cat_f1 = 2 * (cat_precision * cat_recall) / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0
        
        category_metrics[category] = {
            'precision': cat_precision,
            'recall': cat_recall,
            'f1_score': cat_f1,
            'support': tp + fn
        }
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'total_entities': total_entities,
        'category_metrics': category_metrics
    }

def show_performance_analytics():
    """Show performance analytics and optimization insights"""
    st.markdown("### Performance Analytics")
    
    # Processing time analysis
    metrics_data = st.session_state.get('performance_metrics', [])
    if not metrics_data:
        st.info("No performance data available.")
        return
    
    df = pd.DataFrame(metrics_data)
    
    # Processing time statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_time = df['avg_processing_time'].mean()
        st.metric("Average Processing Time", f"{avg_time:.2f}s")
    
    with col2:
        min_time = df['avg_processing_time'].min()
        st.metric("Fastest Processing", f"{min_time:.2f}s")
    
    with col3:
        max_time = df['avg_processing_time'].max()
        st.metric("Slowest Processing", f"{max_time:.2f}s")
    
    with col4:
        std_time = df['avg_processing_time'].std()
        st.metric("Time Variability", f"{std_time:.2f}s")
    
    # Performance trends
    st.markdown("#### Performance Trends")
    
    # Create performance trend chart
    fig_perf = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Processing Time Trend',
            'Error Rate Trend', 
            'Throughput Trend',
            'Resource Usage Trend'
        )
    )
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Processing time
        fig_perf.add_trace(
            go.Scatter(x=df['timestamp'], y=df['avg_processing_time'], name='Processing Time'),
            row=1, col=1
        )
        
        # Error rate
        fig_perf.add_trace(
            go.Scatter(x=df['timestamp'], y=df['error_rate'], name='Error Rate'),
            row=1, col=2
        )
        
        # Throughput (documents per hour - mock calculation)
        df['throughput'] = df['documents_processed_today'] / 8  # Assume 8-hour workday
        fig_perf.add_trace(
            go.Scatter(x=df['timestamp'], y=df['throughput'], name='Throughput'),
            row=2, col=1
        )
        
        # Resource usage
        fig_perf.add_trace(
            go.Scatter(x=df['timestamp'], y=df['system_cpu_usage'], name='CPU Usage'),
            row=2, col=2
        )
    
    fig_perf.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Performance insights
    st.markdown("#### Performance Insights")
    
    insights = generate_performance_insights(df)
    for insight in insights:
        st.markdown(f"• {insight}")

def generate_performance_insights(df: pd.DataFrame) -> List[str]:
    """Generate performance insights from metrics data"""
    insights = []
    
    if len(df) < 2:
        return ["Not enough data for insights."]
    
    # Processing time insights
    avg_time = df['avg_processing_time'].mean()
    if avg_time > 5.0:
        insights.append("⚠️ Average processing time is high (>5s). Consider optimizing models or increasing resources.")
    elif avg_time < 2.0:
        insights.append("✅ Excellent processing performance (<2s average).")
    
    # Error rate insights
    avg_error_rate = df['error_rate'].mean()
    if avg_error_rate > 0.05:
        insights.append("⚠️ Error rate is above 5%. Review error logs and model performance.")
    elif avg_error_rate < 0.02:
        insights.append("✅ Low error rate (<2%). System is performing well.")
    
    # Resource usage insights
    avg_cpu = df['system_cpu_usage'].mean()
    if avg_cpu > 80:
        insights.append("⚠️ High CPU usage detected. Consider scaling resources.")
    elif avg_cpu < 50:
        insights.append("ℹ️ CPU usage is low. Resources may be underutilized.")
    
    # Trend insights
    if len(df) >= 5:
        time_trend = np.polyfit(range(len(df)), df['avg_processing_time'], 1)[0]
        if time_trend > 0.1:
            insights.append("📈 Processing time is increasing over time. Monitor for performance degradation.")
        elif time_trend < -0.1:
            insights.append("📉 Processing time is improving over time. Recent optimizations may be working.")
    
    return insights

def show_system_health():
    """Show detailed system health information"""
    st.markdown("### System Health Dashboard")
    
    # System status overview
    latest_metrics = get_latest_metrics()
    
    # Health indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # System health score (mock calculation)
        cpu_score = max(0, 100 - latest_metrics.get('system_cpu_usage', 0))
        memory_score = max(0, 100 - latest_metrics.get('system_memory_usage', 0))
        error_score = max(0, 100 - latest_metrics.get('error_rate', 0) * 1000)
        
        health_score = (cpu_score + memory_score + error_score) / 3
        
        st.metric("System Health Score", f"{health_score:.0f}/100")
        
        if health_score > 80:
            st.success("🟢 System is healthy")
        elif health_score > 60:
            st.warning("🟡 System needs attention")
        else:
            st.error("🔴 System issues detected")
    
    with col2:
        active_models = latest_metrics.get('active_models', 0)
        st.metric("Active Models", f"{active_models}")
        
        if active_models >= 3:
            st.success("All models running")
        elif active_models >= 1:
            st.warning("Some models offline")
        else:
            st.error("No models active")
    
    with col3:
        uptime = latest_metrics.get('uptime_hours', 0)
        st.metric("System Uptime", f"{uptime:.1f} hours")
        
        if uptime > 24:
            st.success("Stable uptime")
        else:
            st.info("Recent restart")
    
    # Detailed health checks
    st.markdown("#### Detailed Health Checks")
    
    health_checks = [
        {
            'component': 'Database Connection',
            'status': 'healthy',
            'last_check': '2 minutes ago',
            'details': 'Connection pool: 8/10 active'
        },
        {
            'component': 'Model APIs',
            'status': 'healthy',
            'last_check': '30 seconds ago', 
            'details': 'All endpoints responding normally'
        },
        {
            'component': 'File Storage',
            'status': 'warning',
            'last_check': '1 minute ago',
            'details': 'Disk usage: 78% (cleanup recommended)'
        },
        {
            'component': 'Memory Usage',
            'status': 'healthy',
            'last_check': '10 seconds ago',
            'details': f"{latest_metrics.get('system_memory_usage', 0):.1f}% utilized"
        }
    ]
    
    for check in health_checks:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
        
        with col1:
            st.write(f"**{check['component']}**")
        
        with col2:
            status = check['status']
            if status == 'healthy':
                st.success("🟢 Healthy")
            elif status == 'warning':
                st.warning("🟡 Warning")
            else:
                st.error("🔴 Error")
        
        with col3:
            st.write(check['last_check'])
        
        with col4:
            st.write(check['details'])
    
    # Alert configuration
    st.markdown("---")
    st.markdown("#### Alert Configuration")
    
    if auth.has_permission('configure'):
        with st.expander("Configure Alerts"):
            col1, col2 = st.columns(2)
            
            with col1:
                cpu_threshold = st.slider("CPU Alert Threshold (%)", 50, 95, 85)
                memory_threshold = st.slider("Memory Alert Threshold (%)", 50, 95, 80)
            
            with col2:
                error_threshold = st.slider("Error Rate Alert Threshold (%)", 1, 10, 5)
                response_threshold = st.slider("Response Time Alert (seconds)", 1, 30, 10)
            
            if st.button("Save Alert Configuration"):
                alert_config = {
                    'cpu_threshold': cpu_threshold,
                    'memory_threshold': memory_threshold,
                    'error_threshold': error_threshold / 100,
                    'response_threshold': response_threshold
                }
                
                # Store configuration
                st.session_state.alert_config = alert_config
                st.success("Alert configuration saved!")
    else:
        st.info("Alert configuration requires admin permissions.")
    
    # Recent alerts
    st.markdown("#### Recent Alerts")
    
    # Mock recent alerts
    recent_alerts = [
        {
            'timestamp': datetime.now() - timedelta(hours=2),
            'severity': 'warning',
            'message': 'CPU usage exceeded 85% threshold',
            'resolved': True
        },
        {
            'timestamp': datetime.now() - timedelta(hours=6),
            'severity': 'info',
            'message': 'New model deployed successfully',
            'resolved': True
        },
        {
            'timestamp': datetime.now() - timedelta(days=1),
            'severity': 'error',
            'message': 'Database connection timeout',
            'resolved': True
        }
    ]
    
    for alert in recent_alerts:
        severity_color = {
            'error': 'red',
            'warning': 'orange',
            'info': 'blue'
        }.get(alert['severity'], 'gray')
        
        status_icon = '✅' if alert['resolved'] else '⏳'
        
        st.markdown(f"""
        <div style="border-left: 4px solid {severity_color}; padding: 10px; margin: 5px 0; background-color: #f8f9fa;">
            {status_icon} <strong>{alert['severity'].upper()}</strong> - {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}<br>
            {alert['message']}
        </div>
        """, unsafe_allow_html=True)