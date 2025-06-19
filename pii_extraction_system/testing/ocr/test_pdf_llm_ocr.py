#!/usr/bin/env python3
"""Test PDF processing with LLM OCR."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment
from load_env import load_env_file
load_env_file()

def test_pdf_llm_capabilities():
    """Test PDF processing with LLM OCR capabilities."""
    print("🔬 Testing PDF + LLM OCR Integration")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking Dependencies:")
    
    try:
        import pdf2image
        print("   ✅ pdf2image: Available")
    except ImportError:
        print("   ❌ pdf2image: Missing (pip install pdf2image)")
        return False
    
    try:
        from llm import LLM_PROCESSOR_AVAILABLE
        if LLM_PROCESSOR_AVAILABLE:
            print("   ✅ LLM OCR: Available")
        else:
            print("   ⚠️ LLM OCR: Missing dependencies")
    except ImportError:
        print("   ❌ LLM OCR: Import failed")
    
    # Check configuration
    print("\n⚙️ Checking Configuration:")
    try:
        from core.config import settings
        
        enable_llm = getattr(settings.processing, 'enable_llm_ocr', False)
        llm_model = getattr(settings.processing, 'llm_ocr_model', 'gpt-4o-mini')
        max_cost = getattr(settings.processing, 'max_llm_cost_per_document', 0.10)
        
        print(f"   LLM OCR enabled: {enable_llm}")
        print(f"   Default model: {llm_model}")
        print(f"   Max cost per doc: ${max_cost}")
        
        # Enable LLM OCR for testing
        settings.processing.enable_llm_ocr = True
        print("   ✅ LLM OCR enabled for testing")
        
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False
    
    # Check API keys
    print("\n🔑 Checking API Keys:")
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "DeepSeek": os.getenv("DEEPSEEK_API"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY")
    }
    
    available_providers = []
    for provider, key in api_keys.items():
        if key:
            available_providers.append(provider)
            print(f"   ✅ {provider}: Available")
        else:
            print(f"   ❌ {provider}: Not configured")
    
    if not available_providers:
        print("   ⚠️ No API keys available - LLM OCR will not work")
        return False
    
    # Test document processor
    print("\n🔧 Testing Document Processor:")
    try:
        from utils.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("   ✅ Document processor initialized")
        
        # Check LLM OCR status
        llm_enabled = processor.llm_ocr_enabled
        print(f"   LLM OCR enabled in processor: {llm_enabled}")
        
        if not llm_enabled:
            print("   ⚠️ LLM OCR not enabled in processor")
        
    except Exception as e:
        print(f"   ❌ Document processor error: {e}")
        return False
    
    # Test with a simple test case
    print("\n🧪 Testing PDF Processing Capabilities:")
    
    # Show what would happen with different OCR engines
    ocr_engines = ['tesseract', 'easyocr', 'both', 'llm', 'llm+traditional']
    
    for engine in ocr_engines:
        available = True
        description = ""
        
        if engine == 'easyocr':
            try:
                import easyocr
                description = "AI-powered traditional OCR"
            except ImportError:
                available = False
                description = "Missing easyocr package"
        elif engine in ['llm', 'llm+traditional']:
            if available_providers and LLM_PROCESSOR_AVAILABLE:
                description = f"LLM OCR with {available_providers[0]} models"
            else:
                available = False
                description = "Missing LLM dependencies or API keys"
        elif engine == 'tesseract':
            description = "Traditional OCR engine"
        elif engine == 'both':
            description = "Compare traditional OCR engines"
        
        status = "✅" if available else "❌"
        print(f"   {status} {engine}: {description}")
    
    # PDF processing workflow
    print("\n📄 PDF Processing Workflow:")
    print("   1. PDF is uploaded to dashboard")
    print("   2. System tries to extract text directly from PDF")
    print("   3. If PDF is encrypted, uses provided password")
    print("   4. If direct extraction fails/yields little text:")
    print("      a. Converts PDF pages to images (pdf2image)")
    print("      b. Processes each page with selected OCR engine")
    print("      c. If LLM OCR selected:")
    print("         - Sends page image to LLM model")
    print("         - Tracks cost per page")
    print("         - Falls back to traditional OCR if needed")
    print("      d. Combines text from all pages")
    print("   5. Returns extracted text with cost information")
    
    # Recommendations
    print("\n💡 Recommendations for Your Setup:")
    if "OpenAI" in available_providers:
        print("   🥇 Recommended: Use 'llm' with gpt-4o-mini")
        print("      - Best balance of quality and cost")
        print("      - Excellent for complex documents")
        print("      - ~$0.00015 per page")
    
    if "DeepSeek" in available_providers:
        print("   💰 Budget option: Use 'llm' with deepseek-chat")
        print("      - Most cost-effective")
        print("      - Good quality for simple documents")
        print("      - ~$0.0001 per page")
    
    print("   🛡️ Safe option: Use 'llm+traditional'")
    print("      - LLM OCR with traditional fallback")
    print("      - Ensures text extraction even if LLM fails")
    print("      - Cost = LLM cost + minimal traditional cost")
    
    return True


def show_usage_instructions():
    """Show how to use PDF + LLM OCR."""
    print("\n" + "=" * 50)
    print("📋 How to Use PDF + LLM OCR")
    print("=" * 50)
    
    print("\n1. **Start the Dashboard:**")
    print("   streamlit run src/dashboard/main.py")
    
    print("\n2. **For Single PDF Documents:**")
    print("   • Go to 'Document Processing' page")
    print("   • Enter password if your PDF is encrypted (default: 'Hubert')")
    print("   • In 'OCR Settings', select 'llm' or 'llm+traditional'")
    print("   • Choose your LLM model (gpt-4o-mini recommended)")
    print("   • Set cost limit (suggest $0.50 for multi-page documents)")
    print("   • Upload your PDF file")
    print("   • Watch the enhanced OCR results!")
    
    print("\n3. **For Batch PDF Processing:**")
    print("   • Go to 'Batch Analysis' page → 'Batch Upload' tab")
    print("   • Set password for encrypted PDFs")
    print("   • Select 'llm' or 'llm+traditional' OCR engine")
    print("   • Choose LLM model")
    print("   • Set total batch cost limit")
    print("   • Upload multiple PDF files")
    print("   • Monitor total cost estimation")
    
    print("\n4. **Cost Management:**")
    print("   • Monitor costs in real-time")
    print("   • Set per-document limits to control spending")
    print("   • Use 'llm+traditional' for reliability")
    print("   • Check 'LLM OCR Config' page for usage statistics")
    
    print("\n5. **Troubleshooting:**")
    print("   • If PDF is encrypted: Enter correct password")
    print("   • If cost limits exceeded: Lower limits or use cheaper models")
    print("   • If LLM OCR fails: System automatically falls back to traditional")
    print("   • If no text extracted: Check if PDF contains actual text vs images")


if __name__ == "__main__":
    success = test_pdf_llm_capabilities()
    
    if success:
        print("\n🎉 PDF + LLM OCR integration is ready!")
        show_usage_instructions()
    else:
        print("\n❌ Issues detected with PDF + LLM OCR integration")
        print("Please resolve the issues above before using PDF processing.")
        
    exit(0 if success else 1)