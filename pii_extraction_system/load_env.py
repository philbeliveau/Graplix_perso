"""Load environment variables from .env file."""

import os
from pathlib import Path


def load_env_file():
    """Load environment variables from .env file."""
    # Look for .env file in multiple locations
    possible_env_paths = [
        Path.cwd() / ".env",  # Current directory
        Path.cwd().parent / ".env",  # Parent directory
        Path.cwd().parent.parent / ".env",  # Grandparent directory
        Path("/Users/philippebeliveau/Desktop/Notebook/EZBI/GRAPLIX_GIT/.env")  # Known location
    ]
    
    env_file = None
    for path in possible_env_paths:
        if path.exists():
            env_file = path
            break
    
    if not env_file:
        print("⚠️ No .env file found in expected locations")
        return False
    
    print(f"📄 Loading environment from: {env_file}")
    
    # Parse and load .env file
    try:
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    # Only set if not already in environment
                    if key not in os.environ and value:
                        os.environ[key] = value
        
        print(f"✅ Environment variables loaded from {env_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading .env file: {e}")
        return False


def check_api_keys():
    """Check which API keys are available."""
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
        "DeepSeek": os.getenv("DEEPSEEK_API"),
        "NVIDIA": os.getenv("NVIDIA_KEY")
    }
    
    available_keys = []
    for provider, key in api_keys.items():
        if key:
            available_keys.append(provider)
            masked_key = key[:8] + "..." + key[-4:] if len(key) > 12 else "***"
            print(f"✅ {provider}: {masked_key}")
        else:
            print(f"❌ {provider}: Not configured")
    
    return available_keys


if __name__ == "__main__":
    load_env_file()
    check_api_keys()