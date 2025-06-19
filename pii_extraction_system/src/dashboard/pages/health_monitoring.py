"""Health monitoring dashboard page."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

from src.core.health_checks import get_health_monitor, HealthStatus
from src.core.environment_manager import get_env_manager


def show_health_monitoring():
    """Show health monitoring dashboard."""
    st.title("🏥 System Health Monitoring")
    
    # Get health monitor
    health_monitor = get_health_monitor()
    env_manager = get_env_manager()
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.rerun(30)
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    try:
        # Get current system health
        system_health = health_monitor.get_system_health()
        
        # Overall status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = {
                HealthStatus.HEALTHY: "🟢",
                HealthStatus.DEGRADED: "🟡", 
                HealthStatus.UNHEALTHY: "🔴",
                HealthStatus.UNKNOWN: "⚪"
            }.get(system_health.status, "⚪")
            
            st.metric(
                "Overall Status",
                f"{status_color} {system_health.status.value.title()}",
                delta=None
            )
        
        with col2:
            healthy_count = len([c for c in system_health.checks if c.status == HealthStatus.HEALTHY])
            st.metric(
                "Healthy Checks",
                f"{healthy_count}/{len(system_health.checks)}",
                delta=None
            )
        
        with col3:
            avg_response_time = system_health.summary.get("average_response_time_ms", 0)
            st.metric(
                "Avg Response Time",
                f"{avg_response_time:.1f}ms",
                delta=None
            )
        
        with col4:
            st.metric(
                "Environment",
                env_manager.get_current_environment().title(),
                delta=None
            )
        
        # Health checks detail
        st.subheader("Health Checks Detail")
        
        # Create DataFrame for health checks
        health_data = []
        for check in system_health.checks:
            health_data.append({
                "Component": check.name.replace("_", " ").title(),
                "Status": check.status.value.title(),
                "Message": check.message,
                "Response Time (ms)": f"{check.response_time_ms:.1f}",
                "Last Check": check.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        df = pd.DataFrame(health_data)
        
        # Status filter
        status_filter = st.multiselect(
            "Filter by Status",
            options=["Healthy", "Degraded", "Unhealthy", "Unknown"],
            default=["Healthy", "Degraded", "Unhealthy", "Unknown"]
        )
        
        # Filter DataFrame
        if status_filter:
            df_filtered = df[df["Status"].isin(status_filter)]
        else:
            df_filtered = df
        
        # Color-code the status column
        def color_status(val):
            colors = {
                "Healthy": "background-color: #d4edda; color: #155724",
                "Degraded": "background-color: #fff3cd; color: #856404",
                "Unhealthy": "background-color: #f8d7da; color: #721c24",
                "Unknown": "background-color: #e2e3e5; color: #383d41"
            }
            return colors.get(val, "")
        
        # Display table
        st.dataframe(
            df_filtered.style.applymap(color_status, subset=["Status"]),
            use_container_width=True
        )
        
        # Health status distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Status Distribution")
            status_counts = system_health.summary.get("status_counts", {})
            
            if status_counts:
                fig_pie = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    title="Health Check Status Distribution",
                    color_discrete_map={
                        "healthy": "#28a745",
                        "degraded": "#ffc107",
                        "unhealthy": "#dc3545",
                        "unknown": "#6c757d"
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("Response Times")
            response_times = [check.response_time_ms for check in system_health.checks]
            component_names = [check.name.replace("_", " ").title() for check in system_health.checks]
            
            fig_bar = px.bar(
                x=component_names,
                y=response_times,
                title="Response Time by Component",
                labels={"x": "Component", "y": "Response Time (ms)"}
            )
            fig_bar.update_xaxis(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Availability metrics
        st.subheader("📊 Availability Metrics (24h)")
        
        availability_data = []
        for name, checker in health_monitor.checkers.items():
            availability = checker.get_availability(hours=24)
            availability_data.append({
                "Component": name.replace("_", " ").title(),
                "Availability": f"{availability:.2f}%",
                "Availability_Numeric": availability
            })
        
        if availability_data:
            avail_df = pd.DataFrame(availability_data)
            
            # Color-code availability
            def color_availability(val):
                if val >= 99:
                    return "background-color: #d4edda; color: #155724"
                elif val >= 95:
                    return "background-color: #fff3cd; color: #856404"
                else:
                    return "background-color: #f8d7da; color: #721c24"
            
            st.dataframe(
                avail_df[["Component", "Availability"]].style.applymap(
                    color_availability, 
                    subset=["Availability_Numeric"]
                ),
                use_container_width=True
            )
        
        # Failed checks details
        failed_checks = [c for c in system_health.checks if c.status == HealthStatus.UNHEALTHY]
        if failed_checks:
            st.error("🚨 Failed Health Checks")
            
            for check in failed_checks:
                with st.expander(f"❌ {check.name.replace('_', ' ').title()}", expanded=True):
                    st.write(f"**Message:** {check.message}")
                    st.write(f"**Timestamp:** {check.timestamp}")
                    st.write(f"**Response Time:** {check.response_time_ms:.1f}ms")
                    
                    if check.error:
                        st.code(check.error, language="text")
                    
                    if check.details:
                        st.json(check.details)
        
        # System information
        with st.expander("🔧 System Information", expanded=False):
            system_info = {
                "Environment": env_manager.get_current_environment(),
                "Is Production": env_manager.is_production(),
                "Debug Mode": env_manager.get_environment_info().debug_enabled,
                "Monitoring Enabled": env_manager.get_environment_info().monitoring_enabled,
                "Last Updated": system_health.timestamp.isoformat()
            }
            
            st.json(system_info)
        
        # Export options
        st.subheader("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Health Report"):
                report_file = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                health_monitor.export_health_report(report_file)
                st.success(f"Health report exported to {report_file}")
        
        with col2:
            if st.button("Export CSV"):
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"health_checks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("Export JSON"):
                json_data = json.dumps(system_health.summary, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"health_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Health check history (if available)
        st.subheader("📈 Health Trends")
        
        # Create sample trend data for demonstration
        trend_data = []
        for name, checker in health_monitor.checkers.items():
            if checker.history:
                for result in checker.history[-10:]:  # Last 10 results
                    trend_data.append({
                        "Component": name.replace("_", " ").title(),
                        "Timestamp": result.timestamp,
                        "Status": result.status.value,
                        "Response Time": result.response_time_ms
                    })
        
        if trend_data:
            trend_df = pd.DataFrame(trend_data)
            
            # Response time trend
            fig_trend = px.line(
                trend_df,
                x="Timestamp",
                y="Response Time",
                color="Component",
                title="Response Time Trends"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No historical data available yet. Run health checks multiple times to see trends.")
    
    except Exception as e:
        st.error(f"Error loading health monitoring data: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    show_health_monitoring()