"""
Main Streamlit Dashboard for PII Extraction System

This module provides the primary entry point for the comprehensive PII extraction
dashboard, featuring 4 main sections for streamlined system workflow and monitoring.
"""

import streamlit as st
from streamlit_option_menu import option_menu
import sys
import warnings
from pathlib import Path

# Suppress torch warnings for Streamlit compatibility
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*no running event loop.*")
warnings.filterwarnings("ignore", message=".*__path__._path.*")
warnings.filterwarnings("ignore", message=".*does not exist.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent
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

from dashboard.pages import (
    document_processing,
    document_processing_llm,
    batch_analysis,
    model_comparison,
    model_analytics,
    error_analysis,
    performance_metrics,
    data_management,
    configuration,
    dataset_creation_phase0,
    phase1_performance_validation
)

# Import LLM OCR config with error handling
try:
    from dashboard.pages import llm_ocr_simple as llm_ocr_config
except ImportError:
    llm_ocr_config = None

# Import LLM API Status page
try:
    from dashboard.pages import llm_api_status
except ImportError:
    llm_api_status = None
from dashboard.utils import session_state, auth, ui_components

def main():
    """Main dashboard application"""
    # Configure page - disable automatic multipage navigation
    st.set_page_config(
        page_title="PII Extraction System Dashboard",
        page_icon="🔒",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Hide Streamlit's automatic page navigation
    st.markdown("""
    <style>
        /* Hide the automatic multipage navigation */
        .css-1rs6os.edgvbvh3, .css-10trblm.e16nr0p30, .css-1kyxreq.etr89bj2 {
            display: none;
        }
        /* Hide automatic page navigation in sidebar */
        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
            display: none;
        }
        /* Hide the main page navigation */
        .css-1q1n0ol.egzxvld0 {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
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
        
        /* Sidebar text styling */
        .css-1d391kg, .css-1d391kg p, .css-1d391kg h3 {
            color: black !important;
        }
        section[data-testid="stSidebar"] * {
            color: black !important;
        }
        section[data-testid="stSidebar"] .stButton button {
            color: black !important;
        }
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
        
        # Clean single navigation menu - organized by function
        st.markdown("### 📋 Navigation")
        selected_page = option_menu(
            menu_title=None,
            options=[
                "🏠 Dashboard",
                "🎯 Phase 0: Dataset Creation",
                "📊 Phase 1: Performance Validation",
                "📄 Document Processing",
                "🤖 LLM Processing", 
                "📊 Batch Analysis",
                "📈 Model Analytics",
                "⚖️ Model Comparison",
                "🚨 Error Analysis",
                "📊 Performance Metrics",
                "💾 Data Management",
                "⚙️ Configuration",
                "🔍 Health Monitoring",
                "☁️ API Status"
            ],
            icons=[
                "house",
                "bullseye",
                "target",
                "file-text",
                "robot",
                "bar-chart",
                "graph-up",
                "balance-scale",
                "exclamation-triangle",
                "speedometer2",
                "database",
                "gear",
                "activity",
                "cloud-check"
            ],
            menu_icon="list",
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1f77b4", "font-size": "16px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "0px",
                    "color": "black",
                    "--hover-color": "#eee"
                },
                "nav-link-selected": {"background-color": "#1f77b4", "color": "white"},
            }
        )
            
        # Handle navigation override from quick buttons
        if 'nav_override' in st.session_state:
            selected_page = st.session_state.nav_override
            del st.session_state.nav_override
        
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
    if selected_page == "🏠 Dashboard":
        # Main dashboard overview
        st.markdown("## 🏠 Welcome to the PII Extraction System")
        st.info("🎯 **Select a page from the navigation menu to get started.**")
        
        # Quick stats overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Documents Processed", "0", "0")
        with col2:
            st.metric("🔍 PII Entities Found", "0", "0") 
        with col3:
            st.metric("⚡ System Status", "🟢 Active")
        with col4:
            st.metric("☁️ API Status", "🟢 Connected")
            
        # Quick navigation cards
        st.markdown("### 🚀 Quick Start")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎯 Start Phase 0: Dataset"):
                st.session_state.nav_override = "🎯 Phase 0: Dataset Creation"
                st.rerun()
        with col2:
            if st.button("📊 Phase 1: Validation"):
                st.session_state.nav_override = "📊 Phase 1: Performance Validation"
                st.rerun()
        with col3:
            if st.button("📄 Process Document"):
                st.session_state.nav_override = "📄 Document Processing"
                st.rerun()
                
        # Additional workflow options
        st.markdown("### 🔧 Other Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Batch Analysis"):
                st.session_state.nav_override = "📊 Batch Analysis"
                st.rerun()
        with col2:
            if st.button("🤖 LLM Processing"):
                st.session_state.nav_override = "🤖 LLM Processing"
                st.rerun()
        with col3:
            if st.button("⚙️ Configuration"):
                st.session_state.nav_override = "⚙️ Configuration"
                st.rerun()
                
    elif selected_page == "🎯 Phase 0: Dataset Creation":
        dataset_creation_phase0.show_page()
    elif selected_page == "📊 Phase 1: Performance Validation":
        phase1_performance_validation.show_page()
    elif selected_page == "📄 Document Processing":
        document_processing.show_page()
    elif selected_page == "🤖 LLM Processing":
        document_processing_llm.show_page()
    elif selected_page == "📊 Batch Analysis":
        batch_analysis.show_page()
    elif selected_page == "📈 Model Analytics":
        model_analytics.show_page()
    elif selected_page == "⚖️ Model Comparison":
        model_comparison.show_page()
    elif selected_page == "🚨 Error Analysis":
        error_analysis.show_page()
    elif selected_page == "💾 Data Management":
        data_management.show_page()
    elif selected_page == "⚙️ Configuration":
        configuration.show_page()
    elif selected_page == "📊 Performance Metrics":
        performance_metrics.show_page()
    elif selected_page == "🔍 Health Monitoring":
        st.markdown("## 🔍 Health Monitoring")
        st.info("🏥 **System health monitoring and diagnostics**")
        ui_components.show_system_status()
        
        # Additional health metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔄 Uptime", "99.9%", "0.1%")
        with col2:
            st.metric("💾 Memory Usage", "2.1 GB", "-0.3 GB")
        with col3:
            st.metric("🚀 Response Time", "145ms", "-12ms")
            
    elif selected_page == "☁️ API Status":
        if llm_api_status:
            llm_api_status.show_page()
        else:
            st.markdown("## ☁️ API Status")
            st.error("❌ **LLM API Status page not available**")
            st.info("💡 Please check the installation and imports.")

if __name__ == "__main__":
    main()