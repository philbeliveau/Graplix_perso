#!/usr/bin/env python3
"""
Launch script for PII Extraction Dashboard

This script starts the Streamlit dashboard application for the PII extraction system.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    # Get the dashboard main.py path
    dashboard_path = Path(__file__).parent / "src" / "dashboard" / "main.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard main.py not found at {dashboard_path}")
        sys.exit(1)
    
    # Launch Streamlit
    try:
        print("🚀 Starting PII Extraction Dashboard...")
        print(f"📁 Dashboard location: {dashboard_path}")
        print("🌐 Opening browser at http://localhost:8501")
        print("🔒 Use these demo credentials:")
        print("   Admin: admin/admin")
        print("   Analyst: analyst/hello") 
        print("   Viewer: viewer/viewer")
        print("=" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()