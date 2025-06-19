"""Simplified LLM OCR Configuration Page that handles missing dependencies gracefully."""

import streamlit as st
import os
import sys
from pathlib import Path

# Add path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from load_env import load_env_file
    load_env_file()
except ImportError:
    # If load_env is not available, try to find .env manually
    env_paths = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env", 
        Path.cwd().parent.parent / ".env",
        project_root / ".env"
    ]
    for env_path in env_paths:
        if env_path.exists():
            st.info(f"Found .env file at: {env_path}")
            break


def show_llm_ocr_config():
    """Show LLM OCR configuration interface with graceful error handling."""
    st.title("🤖 LLM OCR Configuration")
    
    # Check for API keys first
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
        "DeepSeek": os.getenv("DEEPSEEK_API"),
        "NVIDIA": os.getenv("NVIDIA_KEY")
    }
    
    # Show API key status
    st.markdown("### 🔑 API Key Status")
    
    available_providers = []
    col1, col2 = st.columns(2)
    
    for i, (provider, key) in enumerate(api_keys.items()):
        column = col1 if i % 2 == 0 else col2
        with column:
            if key:
                available_providers.append(provider)
                masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
                st.success(f"✅ {provider}: Configured ({masked_key})")
            else:
                st.error(f"❌ {provider}: Not configured")
    
    if not available_providers:
        st.warning("⚠️ No API keys configured. Please add API keys to your .env file to use LLM OCR.")
        
        st.markdown("### 📝 Required Environment Variables")
        st.code("""
# Add these to your .env file:
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here  
GOOGLE_API_KEY=your_google_key_here
DEEPSEEK_API=your_deepseek_key_here
NVIDIA_KEY=your_nvidia_key_here
        """)
        return
    
    # Basic LLM OCR Configuration
    st.markdown("### ⚙️ LLM OCR Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        enable_llm_ocr = st.checkbox(
            "Enable LLM-based OCR",
            value=False,
            help="Use AI models for enhanced OCR accuracy"
        )
        
        if enable_llm_ocr:
            st.info("✅ LLM OCR will be used for better text extraction quality")
        else:
            st.info("❌ Traditional OCR (Tesseract/EasyOCR) will be used")
    
    with col2:
        if enable_llm_ocr:
            # Model selection based on available providers
            available_models = []
            
            if "OpenAI" in available_providers:
                available_models.extend(["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-vision-preview"])
            if "Anthropic" in available_providers:
                available_models.extend(["claude-3-haiku", "claude-3-sonnet"])
            if "Google" in available_providers:
                available_models.extend(["gemini-1.5-flash", "gemini-1.5-pro"])
            if "DeepSeek" in available_providers:
                available_models.extend(["deepseek-chat"])
            
            if available_models:
                primary_model = st.selectbox(
                    "Primary LLM Model",
                    available_models,
                    help="Main model for LLM-based OCR"
                )
            else:
                st.error("No LLM models available. Please configure API keys.")
    
    # Cost settings
    if enable_llm_ocr:
        st.markdown("### 💰 Cost Control")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_cost_per_doc = st.slider(
                "Max Cost per Document ($)",
                0.01, 1.00, 0.10, 0.01,
                help="Maximum cost allowed per document"
            )
            
            enable_cost_optimization = st.checkbox(
                "Enable Cost Optimization",
                value=True,
                help="Automatically switch to cheaper models when needed"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 0.8, 0.05,
                help="Minimum confidence to use LLM result over traditional OCR"
            )
            
            use_fallback = st.checkbox(
                "Use Traditional OCR Fallback",
                value=True,
                help="Fall back to Tesseract/EasyOCR if LLM fails"
            )
    
    # Model comparison table
    st.markdown("### 📊 Model Information")
    
    # Static model information (doesn't require imports)
    model_info = [
        {"Model": "GPT-4o Mini", "Provider": "OpenAI", "Cost": "$0.000150/1K", "Quality": "High", "Speed": "Fast", "Vision": "✅"},
        {"Model": "GPT-3.5 Turbo", "Provider": "OpenAI", "Cost": "$0.000500/1K", "Quality": "Good", "Speed": "Very Fast", "Vision": "❌"},
        {"Model": "Claude 3 Haiku", "Provider": "Anthropic", "Cost": "$0.000250/1K", "Quality": "Good", "Speed": "Very Fast", "Vision": "✅"},
        {"Model": "Claude 3 Sonnet", "Provider": "Anthropic", "Cost": "$0.003000/1K", "Quality": "Very High", "Speed": "Medium", "Vision": "✅"},
        {"Model": "Gemini 1.5 Flash", "Provider": "Google", "Cost": "$0.000075/1K", "Quality": "Good", "Speed": "Fast", "Vision": "✅"},
        {"Model": "Gemini 1.5 Pro", "Provider": "Google", "Cost": "$0.001250/1K", "Quality": "High", "Speed": "Medium", "Vision": "✅"},
        {"Model": "DeepSeek Chat", "Provider": "DeepSeek", "Cost": "$0.000140/1K", "Quality": "Good", "Speed": "Fast", "Vision": "❌"},
    ]
    
    # Filter models based on available providers
    available_model_info = [
        model for model in model_info 
        if model["Provider"] in available_providers
    ]
    
    if available_model_info:
        st.table(available_model_info)
    else:
        st.info("Configure API keys to see available models.")
    
    # Cost estimation
    if enable_llm_ocr and available_models:
        st.markdown("### 🧮 Cost Estimation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            doc_pages = st.number_input("Document Pages", 1, 100, 5)
            image_complexity = st.selectbox("Image Complexity", ["Low", "Medium", "High"])
        
        with col2:
            text_density = st.selectbox("Text Density", ["Light", "Medium", "Dense"])
            processing_type = st.selectbox("Processing Type", ["Basic OCR", "Structured Data", "Complex Analysis"])
        
        with col3:
            # Simple cost estimation
            base_cost = {
                "gpt-4o-mini": 0.0002,
                "gpt-3.5-turbo": 0.0008,
                "claude-3-haiku": 0.0004,
                "claude-3-sonnet": 0.005,
                "gemini-1.5-flash": 0.0001,
                "gemini-1.5-pro": 0.002,
                "deepseek-chat": 0.0002
            }
            
            if enable_llm_ocr and 'primary_model' in locals():
                cost_per_page = base_cost.get(primary_model, 0.001)
                
                # Adjust for complexity
                complexity_multiplier = {"Low": 0.8, "Medium": 1.0, "High": 1.5}[image_complexity]
                density_multiplier = {"Light": 0.7, "Medium": 1.0, "Dense": 1.3}[text_density]
                type_multiplier = {"Basic OCR": 1.0, "Structured Data": 1.5, "Complex Analysis": 2.0}[processing_type]
                
                estimated_cost = (cost_per_page * doc_pages * 
                                complexity_multiplier * density_multiplier * type_multiplier)
                
                st.metric("Estimated Cost", f"${estimated_cost:.4f}")
                st.metric("Cost per Page", f"${estimated_cost/doc_pages:.4f}")
                
                if estimated_cost > max_cost_per_doc:
                    st.warning(f"⚠️ Estimated cost exceeds limit (${max_cost_per_doc:.2f})")
                else:
                    st.success("✅ Within cost limit")
    
    # Save configuration
    if st.button("Save LLM OCR Configuration", type="primary"):
        config = {
            "enable_llm_ocr": enable_llm_ocr,
            "available_providers": available_providers
        }
        
        if enable_llm_ocr and 'primary_model' in locals():
            config.update({
                "primary_model": primary_model,
                "max_cost_per_document": max_cost_per_doc,
                "confidence_threshold": confidence_threshold,
                "enable_cost_optimization": enable_cost_optimization,
                "use_fallback": use_fallback
            })
        
        st.session_state.llm_ocr_config = config
        st.success("✅ LLM OCR configuration saved successfully!")
        
        if enable_llm_ocr:
            st.info("🚀 LLM OCR is now enabled. You may need to install additional packages:")
            st.code("pip install openai anthropic google-generativeai")
    
    # Installation instructions
    with st.expander("📦 Installation Instructions"):
        st.markdown("""
        To use LLM OCR functionality, install the required packages:
        
        ```bash
        # Install all LLM providers
        pip install openai anthropic google-generativeai
        
        # Or install specific providers
        pip install openai                    # For OpenAI models
        pip install anthropic                 # For Claude models  
        pip install google-generativeai       # For Gemini models
        ```
        
        Then configure your API keys in the .env file:
        ```
        OPENAI_API_KEY=your_key_here
        ANTHROPIC_API_KEY=your_key_here
        GOOGLE_API_KEY=your_key_here
        ```
        """)
    
    # Help section
    with st.expander("❓ Help & FAQ"):
        st.markdown("""
        **Q: Which model should I choose?**
        A: For most use cases, start with Gemini 1.5 Flash (cheapest) or GPT-4o Mini (good balance of cost/quality).
        
        **Q: How much does LLM OCR cost?**
        A: Costs range from $0.000075 to $0.005 per 1K tokens. A typical page costs $0.0001-0.01.
        
        **Q: What if LLM OCR fails?**
        A: Enable fallback to automatically use traditional OCR (Tesseract/EasyOCR) as backup.
        
        **Q: How can I control costs?**
        A: Set max cost per document, enable cost optimization, and use confidence thresholds.
        """)


if __name__ == "__main__":
    # For testing
    show_llm_ocr_config()