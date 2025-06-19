#!/usr/bin/env python3
"""
Install Tesseract and dependencies for fallback OCR
"""

import subprocess
import sys
import platform

def install_tesseract():
    """Install Tesseract OCR based on the operating system."""
    system = platform.system().lower()
    
    print(f"🔧 Installing Tesseract OCR for {system}...")
    
    try:
        if system == "darwin":  # macOS
            print("📦 Installing Tesseract via Homebrew...")
            subprocess.run(["brew", "install", "tesseract"], check=True)
            subprocess.run(["brew", "install", "tesseract-lang"], check=True)
            
        elif system == "linux":
            print("📦 Installing Tesseract via apt...")
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(["sudo", "apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-fra"], check=True)
            
        elif system == "windows":
            print("📦 For Windows, please download Tesseract from:")
            print("https://github.com/UB-Mannheim/tesseract/wiki")
            return False
            
        print("✅ Tesseract installed successfully!")
        
        # Install Python packages
        print("📦 Installing Python dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytesseract", "opencv-python"], check=True)
        
        print("✅ All dependencies installed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Package manager not found. Please install manually:")
        if system == "darwin":
            print("- Install Homebrew: https://brew.sh/")
        elif system == "linux":
            print("- Use your distribution's package manager")
        return False

if __name__ == "__main__":
    install_tesseract()