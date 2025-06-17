"""
Main Streamlit Dashboard for PII Extraction System

This module provides the primary entry point for the comprehensive PII extraction
dashboard, featuring 7 main sections for complete system interaction and monitoring.
"""

import streamlit as st
from streamlit_option_menu import option_menu
import sys
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from dashboard.pages import (
    document_processing,
    batch_analysis,
    model_comparison,
    error_analysis,
    performance_metrics,
    data_management,
    configuration
)
from dashboard.utils import session_state, auth, ui_components

def main():
    """Main dashboard application"""
    # Configure page
    st.set_page_config(
        page_title="PII Extraction System Dashboard",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional appearance
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5em;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 30px;
        }
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 20px;
            margin-bottom: 15px;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #1f77b4;
            margin: 10px 0;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active { background-color: #28a745; }
        .status-warning { background-color: #ffc107; }
        .status-error { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    session_state.initialize_session_state()
    
    # Authentication check
    if not auth.check_authentication():
        auth.show_login_page()
        return
    
    # Main header
    st.markdown('<div class="main-header">🔒 PII Extraction System Dashboard</div>', 
                unsafe_allow_html=True)
    
    # Navigation menu
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60/1f77b4/ffffff?text=PII+System", 
                width=200)
        
        selected_page = option_menu(
            menu_title="Navigation",
            options=[
                "Document Processing",
                "Batch Analysis", 
                "Model Comparison",
                "Error Analysis",
                "Performance Metrics",
                "Data Management",
                "Configuration"
            ],
            icons=[
                "file-earmark-text",
                "files", 
                "diagram-3",
                "bug",
                "graph-up",
                "database",
                "gear"
            ],
            menu_icon="list",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1f77b4", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "--hover-color": "#eee"
                },
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )
        
        # System status indicator
        st.markdown("---")
        st.markdown("### System Status")
        ui_components.show_system_status()
        
        # User info
        st.markdown("---")
        st.markdown(f"**User:** {st.session_state.get('username', 'Anonymous')}")
        if st.button("Logout"):
            auth.logout()
            st.rerun()
    
    # Route to selected page
    if selected_page == "Document Processing":
        document_processing.show_page()
    elif selected_page == "Batch Analysis":
        batch_analysis.show_page()
    elif selected_page == "Model Comparison":
        model_comparison.show_page()
    elif selected_page == "Error Analysis":
        error_analysis.show_page()
    elif selected_page == "Performance Metrics":
        performance_metrics.show_page()
    elif selected_page == "Data Management":
        data_management.show_page()
    elif selected_page == "Configuration":
        configuration.show_page()

if __name__ == "__main__":
    main()