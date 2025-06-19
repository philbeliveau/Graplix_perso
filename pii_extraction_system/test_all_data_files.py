#!/usr/bin/env python3
"""Test LLM OCR with all files in the data directory."""

import os
import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load environment
from load_env import load_env_file
load_env_file()

def analyze_text_quality(text: str) -> dict:
    """Analyze extracted text quality."""
    if not text:
        return {'quality': 0, 'issues': ['No text extracted'], 'word_count': 0, 'char_count': 0, 'line_count': 0, 'strengths': []}
    
    analysis = {
        'char_count': len(text),
        'word_count': len(text.split()),
        'line_count': len([l for l in text.split('\n') if l.strip()]),
        'quality': 0,
        'issues': [],
        'strengths': []
    }
    
    # Check for French accents
    french_accents = ['é', 'è', 'à', 'ç', 'ô', 'û', 'ê', 'ù', 'â', 'î']
    found_accents = [c for c in french_accents if c in text]
    if found_accents:
        analysis['strengths'].append(f"French accents: {found_accents[:3]}")
        analysis['quality'] += 20
    
    # Check for garbled text
    garbled_patterns = ['ÿ', 'rs bu à me', 'Lo mue']
    if any(pattern in text for pattern in garbled_patterns):
        analysis['issues'].append("Garbled text detected")
    else:
        analysis['quality'] += 30
    
    # Check word count
    if analysis['word_count'] > 50:
        analysis['quality'] += 25
        analysis['strengths'].append("Good word count")
    elif analysis['word_count'] > 10:
        analysis['quality'] += 15
    
    # Check structure
    if analysis['line_count'] > 5:
        analysis['quality'] += 15
        analysis['strengths'].append("Good structure")
    elif analysis['line_count'] > 2:
        analysis['quality'] += 10
    
    # Check for meaningful content
    meaningful_words = ['de', 'du', 'le', 'la', 'et', 'pour', 'dans', 'avec']
    if any(word in text.lower() for word in meaningful_words):
        analysis['quality'] += 10
        analysis['strengths'].append("Meaningful content")
    
    return analysis

def test_all_data_files():
    """Test all files in the data directory."""
    data_dir = Path('/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/data')
    
    print("🚀 Testing All Files in Data Directory")
    print("=" * 70)
    print(f"📁 Directory: {data_dir}")
    
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return
    
    # Get all files
    files = [f for f in data_dir.iterdir() if f.is_file()]
    supported_extensions = {'.jpg', '.jpeg', '.png', '.pdf', '.docx', '.doc', '.txt', '.xlsx'}
    processable_files = [f for f in files if f.suffix.lower() in supported_extensions]
    
    print(f"📊 Found {len(files)} total files, {len(processable_files)} processable")
    
    if not processable_files:
        print("❌ No supported files found!")
        return
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OpenAI API key found!")
        return
    
    try:
        # Import and setup
        from utils.document_processor import DocumentProcessor
        from llm import LLM_PROCESSOR_AVAILABLE
        
        if not LLM_PROCESSOR_AVAILABLE:
            print("❌ LLM processor not available!")
            return
        
        processor = DocumentProcessor()
        if not processor.llm_ocr_enabled:
            print("❌ LLM OCR not enabled!")
            return
        
        print("✅ LLM OCR ready - starting processing...")
        
        # Process each file
        results = []
        total_cost = 0
        
        for i, file_path in enumerate(processable_files, 1):
            print(f"\n{'='*70}")
            print(f"🔄 Processing {i}/{len(processable_files)}: {file_path.name}")
            print(f"📁 Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
            
            start_time = time.time()
            
            try:
                # Special handling for encrypted PDFs
                password = None
                if file_path.name == "specimen cheque - Souad Touati.pdf":
                    password = "Hubert"  # Known password for this file
                    print("🔐 Using password 'Hubert' for encrypted PDF")
                
                # Process the file
                result = processor.process_document(file_path, password=password)
                processing_time = time.time() - start_time
                
                # Extract results
                text = result.get('ocr_text', result.get('raw_text', ''))
                confidence = result.get('confidence', 0)
                engine = result.get('ocr_engine', 'unknown')
                
                # Get cost info
                cost = 0
                model = 'N/A'
                llm_metadata = result.get('llm_metadata', {})
                if llm_metadata:
                    cost_info = llm_metadata.get('cost_info', {})
                    cost = cost_info.get('actual_cost', 0)
                    model = llm_metadata.get('model_used', 'N/A')
                    total_cost += cost
                
                # Analyze quality
                analysis = analyze_text_quality(text)
                
                # Show results
                print(f"✅ SUCCESS")
                print(f"   Engine: {engine}")
                print(f"   Model: {model}")
                print(f"   Confidence: {confidence:.2f}")
                print(f"   Time: {processing_time:.1f}s")
                print(f"   Cost: ${cost:.6f}")
                print(f"   Quality: {analysis['quality']}/100")
                print(f"   Text: {analysis['word_count']} words, {analysis['char_count']} chars")
                
                if analysis['strengths']:
                    print(f"   ✅ Strengths: {', '.join(analysis['strengths'])}")
                if analysis['issues']:
                    print(f"   ⚠️  Issues: {', '.join(analysis['issues'])}")
                
                # Show text preview
                if text:
                    preview = text[:300].replace('\n', ' ').strip()
                    if len(text) > 300:
                        preview += "..."
                    print(f"   📄 Preview: {preview}")
                else:
                    print("   📄 No text extracted")
                
                results.append({
                    'file': file_path.name,
                    'success': True,
                    'engine': engine,
                    'model': model,
                    'cost': cost,
                    'quality': analysis['quality'],
                    'words': analysis['word_count'],
                    'text_length': len(text),
                    'time': processing_time,
                    'issues': analysis['issues'],
                    'strengths': analysis['strengths']
                })
                
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"❌ FAILED: {str(e)}")
                results.append({
                    'file': file_path.name,
                    'success': False,
                    'error': str(e),
                    'time': processing_time
                })
        
        # Summary
        print(f"\n{'='*70}")
        print("📋 FINAL SUMMARY")
        print(f"{'='*70}")
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print(f"📊 Results:")
        print(f"   ✅ Successful: {len(successful)}/{len(results)} files")
        print(f"   ❌ Failed: {len(failed)} files")
        print(f"   💰 Total cost: ${total_cost:.6f}")
        
        if successful:
            avg_quality = sum(r['quality'] for r in successful) / len(successful)
            total_words = sum(r['words'] for r in successful)
            print(f"   🏆 Average quality: {avg_quality:.1f}/100")
            print(f"   📝 Total words: {total_words}")
        
        # File-by-file breakdown
        print(f"\n📄 File Results:")
        print("-" * 70)
        for r in results:
            if r['success']:
                status = "✅"
                details = f"Quality: {r['quality']:3d}/100 | Words: {r['words']:4d} | ${r['cost']:.6f}"
            else:
                status = "❌"
                details = f"Error: {r['error'][:40]}..."
            
            print(f"{status} {r['file']:<35} | {details}")
        
        # Issues summary
        all_issues = []
        for r in successful:
            all_issues.extend(r['issues'])
        
        if all_issues:
            print(f"\n⚠️  Common Issues:")
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            for issue, count in sorted(issue_counts.items()):
                print(f"   • {issue}: {count} files")
        
        print(f"\n🎉 Processing complete! LLM OCR processed {len(successful)}/{len(results)} files successfully.")
        
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all_data_files()