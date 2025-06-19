"""LLM OCR Configuration Page for the PII Extraction Dashboard."""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

try:
    from llm import (
        LLMModelRegistry, CostCalculator, cost_tracker, 
        OCRTaskType, LLMProvider, llm_config
    )
    from core.config import settings
    LLM_IMPORTS_AVAILABLE = True
    LLM_IMPORT_ERROR = None
except ImportError as e:
    # Store error for later display
    LLM_IMPORTS_AVAILABLE = False
    LLM_IMPORT_ERROR = str(e)


def show_llm_ocr_config():
    """Show LLM OCR configuration interface."""
    st.title("🤖 LLM OCR Configuration")
    
    if not LLM_IMPORTS_AVAILABLE:
        st.error("⚠️ LLM modules are not available. Please install the required packages:")
        st.code("pip install openai anthropic google-generativeai")
        if LLM_IMPORT_ERROR:
            with st.expander("Show Error Details"):
                st.code(LLM_IMPORT_ERROR)
        st.info("You can still configure basic OCR settings in the main Configuration page.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Selection", 
        "Cost Analysis", 
        "Performance Monitoring", 
        "Advanced Settings"
    ])
    
    with tab1:
        show_model_selection()
    
    with tab2:
        show_cost_analysis()
    
    with tab3:
        show_performance_monitoring()
    
    with tab4:
        show_advanced_settings()


def show_model_selection():
    """Show model selection interface."""
    st.markdown("### Model Selection & Configuration")
    
    # Enable/disable LLM OCR
    col1, col2 = st.columns([2, 1])
    
    with col1:
        enable_llm_ocr = st.checkbox(
            "Enable LLM-based OCR",
            value=getattr(settings.processing, 'enable_llm_ocr', False),
            help="Use AI models for enhanced OCR accuracy"
        )
    
    with col2:
        if enable_llm_ocr:
            st.success("✅ LLM OCR Enabled")
        else:
            st.info("❌ LLM OCR Disabled")
    
    if not enable_llm_ocr:
        st.info("Enable LLM OCR to access advanced model selection features.")
        return
    
    # Model selection
    st.markdown("#### Available Models")
    
    # Get available models
    all_models = LLMModelRegistry.MODELS
    vision_models = LLMModelRegistry.get_vision_models()
    
    # Create model comparison table
    model_data = []
    for model in vision_models:
        model_data.append({
            "Model": model.display_name,
            "Provider": model.provider.value.title(),
            "Input Cost ($/1K tokens)": f"${model.input_cost_per_1k_tokens:.6f}",
            "Output Cost ($/1K tokens)": f"${model.output_cost_per_1k_tokens:.6f}",
            "Quality Score": f"{model.quality_score:.2f}/1.0",
            "Speed Score": f"{model.speed_score:.2f}/1.0",
            "Max Tokens": f"{model.max_tokens:,}",
            "Description": model.description
        })
    
    df_models = pd.DataFrame(model_data)
    st.dataframe(df_models, use_container_width=True)
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Primary Model Selection")
        
        available_model_names = [model.model_name for model in vision_models]
        available_display_names = [model.display_name for model in vision_models]
        
        primary_model_display = st.selectbox(
            "Primary OCR Model",
            available_display_names,
            index=0,
            help="Main model used for OCR processing"
        )
        
        # Get actual model name
        primary_model_idx = available_display_names.index(primary_model_display)
        primary_model = available_model_names[primary_model_idx]
        
        # Show model details
        selected_model = LLMModelRegistry.get_model(primary_model)
        if selected_model:
            st.info(f"**Cost**: ${selected_model.input_cost_per_1k_tokens:.6f} input + ${selected_model.output_cost_per_1k_tokens:.6f} output per 1K tokens")
            st.info(f"**Quality**: {selected_model.quality_score:.1%} | **Speed**: {selected_model.speed_score:.1%}")
    
    with col2:
        st.markdown("#### Fallback Model Selection")
        
        fallback_options = ["None"] + [name for name in available_display_names if name != primary_model_display]
        fallback_model_display = st.selectbox(
            "Fallback OCR Model",
            fallback_options,
            index=1 if len(fallback_options) > 1 else 0,
            help="Model used if primary model fails"
        )
        
        if fallback_model_display != "None":
            fallback_idx = available_display_names.index(fallback_model_display)
            fallback_model = available_model_names[fallback_idx]
            fallback_model_obj = LLMModelRegistry.get_model(fallback_model)
            if fallback_model_obj:
                st.info(f"**Cost**: ${fallback_model_obj.input_cost_per_1k_tokens:.6f} input + ${fallback_model_obj.output_cost_per_1k_tokens:.6f} output per 1K tokens")
    
    # Task-specific model mapping
    st.markdown("#### Task-Specific Model Mapping")
    st.info("Configure different models for specific OCR tasks to optimize cost and quality.")
    
    task_models = {}
    for task_type in OCRTaskType:
        task_models[task_type.value] = st.selectbox(
            f"{task_type.value.replace('_', ' ').title()}",
            ["Use Primary"] + available_display_names,
            key=f"task_{task_type.value}",
            help=f"Model for {task_type.value.replace('_', ' ')} tasks"
        )
    
    # Cost estimation
    st.markdown("#### Cost Estimation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        image_width = st.number_input("Image Width (px)", 100, 4000, 1024)
        image_height = st.number_input("Image Height (px)", 100, 4000, 1024)
    
    with col2:
        task_type = st.selectbox(
            "Task Type",
            [task.value.replace('_', ' ').title() for task in OCRTaskType]
        )
        complex_text = st.checkbox("Complex Text/Layout", value=True)
    
    with col3:
        # Calculate cost estimate
        task_enum = OCRTaskType(task_type.lower().replace(' ', '_'))
        cost_estimate = CostCalculator.estimate_document_cost(
            primary_model,
            task_enum,
            image_width,
            image_height,
            complex_text
        )
        
        st.metric("Estimated Cost", f"${cost_estimate['total_cost']:.6f}")
        st.metric("Input Tokens", f"{cost_estimate['input_tokens']:,}")
        st.metric("Output Tokens", f"{cost_estimate['output_tokens']:,}")
    
    # Save configuration
    if st.button("Save Model Configuration", type="primary"):
        save_llm_ocr_config({
            'enable_llm_ocr': enable_llm_ocr,
            'primary_model': primary_model,
            'fallback_model': fallback_model if fallback_model_display != "None" else None,
            'task_models': task_models
        })


def show_cost_analysis():
    """Show cost analysis and usage statistics."""
    st.markdown("### Cost Analysis & Usage Statistics")
    
    # Cost controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Cost Controls")
        
        max_cost_per_doc = st.slider(
            "Max Cost per Document ($)",
            0.01, 1.00, 
            getattr(settings.processing, 'max_llm_cost_per_document', 0.10),
            0.01,
            help="Maximum allowed cost per document"
        )
        
        enable_cost_optimization = st.checkbox(
            "Enable Cost Optimization",
            value=True,
            help="Automatically switch to cheaper models when cost limits are exceeded"
        )
        
        prefer_cheaper_models = st.checkbox(
            "Prefer Cheaper Models",
            value=True,
            help="Prioritize cost over quality when selecting models"
        )
    
    with col2:
        st.markdown("#### Cost Limits & Alerts")
        
        daily_limit = st.number_input(
            "Daily Cost Limit ($)",
            0.0, 100.0, 10.0,
            help="Alert when daily costs exceed this amount"
        )
        
        monthly_limit = st.number_input(
            "Monthly Cost Limit ($)",
            0.0, 1000.0, 200.0,
            help="Alert when monthly costs exceed this amount"
        )
    
    # Usage statistics
    st.markdown("#### Usage Statistics")
    
    # Get usage stats
    model_stats = cost_tracker.get_model_usage_stats()
    daily_cost = cost_tracker.get_daily_costs()
    monthly_cost = cost_tracker.get_monthly_costs()
    
    # Display current costs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Today's Cost", f"${daily_cost:.4f}")
    with col2:
        st.metric("Monthly Cost", f"${monthly_cost:.4f}")
    with col3:
        remaining_daily = max(0, daily_limit - daily_cost)
        st.metric("Daily Remaining", f"${remaining_daily:.4f}")
    with col4:
        remaining_monthly = max(0, monthly_limit - monthly_cost)
        st.metric("Monthly Remaining", f"${remaining_monthly:.4f}")
    
    # Model usage breakdown
    if model_stats:
        st.markdown("#### Model Usage Breakdown")
        
        usage_data = []
        for model_name, stats in model_stats.items():
            model_obj = LLMModelRegistry.get_model(model_name)
            usage_data.append({
                "Model": model_obj.display_name if model_obj else model_name,
                "Requests": stats['total_requests'],
                "Input Tokens": f"{stats['total_input_tokens']:,}",
                "Output Tokens": f"{stats['total_output_tokens']:,}",
                "Total Cost": f"${stats['total_cost']:.6f}",
                "Avg Cost/Request": f"${stats['total_cost']/max(stats['total_requests'], 1):.6f}"
            })
        
        df_usage = pd.DataFrame(usage_data)
        st.dataframe(df_usage, use_container_width=True)
    else:
        st.info("No usage data available yet. Process some documents to see statistics.")
    
    # Cost comparison
    st.markdown("#### Model Cost Comparison")
    
    # Get cheapest models for comparison
    cheapest_models = LLMModelRegistry.get_cheapest_models(vision_only=True)
    best_value_models = LLMModelRegistry.get_best_value_models(vision_only=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Cost-Effective Models:**")
        for i, model in enumerate(cheapest_models[:5], 1):
            st.write(f"{i}. {model.display_name}: ${model.input_cost_per_1k_tokens:.6f}/1K tokens")
    
    with col2:
        st.markdown("**Best Value Models (Quality/Cost):**")
        for i, model in enumerate(best_value_models[:5], 1):
            avg_cost = (model.input_cost_per_1k_tokens + model.output_cost_per_1k_tokens) / 2
            value_score = model.quality_score / max(avg_cost, 0.00001)
            st.write(f"{i}. {model.display_name}: Score {value_score:.0f}")


def show_performance_monitoring():
    """Show performance monitoring dashboard."""
    st.markdown("### Performance Monitoring")
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Response Time", "2.3s", "-0.4s")
        st.metric("Success Rate", "98.5%", "+1.2%")
    
    with col2:
        st.metric("Avg Confidence", "87.3%", "+2.1%")
        st.metric("Fallback Rate", "3.2%", "-0.8%")
    
    with col3:
        st.metric("Tokens/Second", "1,247", "+156")
        st.metric("Cost/Page", "$0.023", "-$0.004")
    
    # Performance by model
    st.markdown("#### Performance by Model")
    
    performance_data = [
        {"Model": "GPT-4o Mini", "Avg Time": "1.8s", "Confidence": "89%", "Cost/Page": "$0.015", "Accuracy": "94%"},
        {"Model": "Claude 3 Haiku", "Avg Time": "1.5s", "Confidence": "85%", "Cost/Page": "$0.012", "Accuracy": "91%"},
        {"Model": "Gemini 1.5 Flash", "Avg Time": "2.1s", "Confidence": "87%", "Cost/Page": "$0.008", "Accuracy": "92%"},
        {"Model": "GPT-3.5 Turbo", "Avg Time": "1.2s", "Confidence": "82%", "Cost/Page": "$0.021", "Accuracy": "88%"}
    ]
    
    df_performance = pd.DataFrame(performance_data)
    st.dataframe(df_performance, use_container_width=True)
    
    # Quality vs Cost scatter plot
    st.markdown("#### Quality vs Cost Analysis")
    
    # Create scatter plot data
    models = LLMModelRegistry.get_vision_models()
    quality_scores = [model.quality_score for model in models]
    avg_costs = [(model.input_cost_per_1k_tokens + model.output_cost_per_1k_tokens) / 2 for model in models]
    model_names = [model.display_name for model in models]
    
    scatter_data = pd.DataFrame({
        'Quality Score': quality_scores,
        'Average Cost ($/1K tokens)': avg_costs,
        'Model': model_names
    })
    
    st.scatter_chart(scatter_data, x='Average Cost ($/1K tokens)', y='Quality Score', color='Model')


def show_advanced_settings():
    """Show advanced LLM OCR settings."""
    st.markdown("### Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Quality & Confidence")
        
        confidence_threshold = st.slider(
            "Minimum Confidence Threshold",
            0.0, 1.0, 
            getattr(settings.processing, 'llm_confidence_threshold', 0.8),
            0.05,
            help="Minimum confidence to use LLM result over traditional OCR"
        )
        
        use_ensemble_critical = st.checkbox(
            "Use Ensemble for Critical Documents",
            value=True,
            help="Use multiple models for important documents"
        )
        
        enable_quality_filtering = st.checkbox(
            "Enable Quality Filtering",
            value=True,
            help="Filter out low-quality results automatically"
        )
    
    with col2:
        st.markdown("#### Retry & Timeout")
        
        max_retry_attempts = st.number_input(
            "Max Retry Attempts",
            1, 10, 3,
            help="Number of retry attempts for failed requests"
        )
        
        timeout_seconds = st.number_input(
            "Request Timeout (seconds)",
            10, 300, 60,
            help="Timeout for LLM API requests"
        )
        
        enable_batch_processing = st.checkbox(
            "Enable Batch Processing",
            value=True,
            help="Process multiple images in batches when possible"
        )
    
    # Provider-specific settings
    st.markdown("#### Provider-Specific Settings")
    
    # OpenAI settings
    with st.expander("OpenAI Settings"):
        openai_temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        openai_max_tokens = st.number_input("Max Tokens", 100, 4000, 2000)
    
    # Anthropic settings  
    with st.expander("Anthropic Settings"):
        anthropic_temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, key="anthropic_temp")
        anthropic_max_tokens = st.number_input("Max Tokens", 100, 4000, 2000, key="anthropic_max")
    
    # Google settings
    with st.expander("Google Gemini Settings"):
        gemini_temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1, key="gemini_temp")
        gemini_candidate_count = st.number_input("Candidate Count", 1, 4, 1)
    
    # Custom prompts
    st.markdown("#### Custom Prompts")
    
    for task_type in OCRTaskType:
        with st.expander(f"{task_type.value.replace('_', ' ').title()} Prompt"):
            custom_prompt = st.text_area(
                f"Custom prompt for {task_type.value}",
                placeholder=f"Enter custom prompt for {task_type.value} tasks...",
                key=f"prompt_{task_type.value}",
                height=100
            )
    
    # Save advanced settings
    if st.button("Save Advanced Settings", type="primary"):
        save_advanced_llm_settings({
            'confidence_threshold': confidence_threshold,
            'max_retry_attempts': max_retry_attempts,
            'timeout_seconds': timeout_seconds,
            'use_ensemble_critical': use_ensemble_critical,
            'enable_batch_processing': enable_batch_processing,
            'provider_settings': {
                'openai': {'temperature': openai_temperature, 'max_tokens': openai_max_tokens},
                'anthropic': {'temperature': anthropic_temperature, 'max_tokens': anthropic_max_tokens},
                'google': {'temperature': gemini_temperature, 'candidate_count': gemini_candidate_count}
            }
        })


def save_llm_ocr_config(config: Dict[str, Any]):
    """Save LLM OCR configuration."""
    st.session_state.llm_ocr_config = config
    st.success("LLM OCR configuration saved successfully!")
    
    # Update global config
    llm_config.default_model = config.get('primary_model', 'gpt-4o-mini')
    llm_config.fallback_model = config.get('fallback_model', 'gpt-3.5-turbo')
    llm_config.enabled_models = [config['primary_model']]
    if config.get('fallback_model'):
        llm_config.enabled_models.append(config['fallback_model'])


def save_advanced_llm_settings(config: Dict[str, Any]):
    """Save advanced LLM settings."""
    st.session_state.llm_advanced_config = config
    st.success("Advanced LLM settings saved successfully!")
    
    # Update global config
    llm_config.min_confidence_threshold = config.get('confidence_threshold', 0.8)
    llm_config.max_retry_attempts = config.get('max_retry_attempts', 3)
    llm_config.timeout_seconds = config.get('timeout_seconds', 60)
    llm_config.use_ensemble_for_critical = config.get('use_ensemble_critical', True)
    llm_config.batch_processing = config.get('enable_batch_processing', True)