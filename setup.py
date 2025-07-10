#!/usr/bin/env python3
"""
Setup script for RAG Application
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required. You have {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} is compatible")
    return True


def create_virtual_environment():
    """Create a virtual environment if it doesn't exist."""
    if Path("venv").exists():
        print("✅ Virtual environment already exists")
        return True
    
    return run_command("python3 -m venv venv", "Creating virtual environment")


def activate_and_install():
    """Activate virtual environment and install dependencies."""
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:  # Unix/Linux/MacOS
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    # Install requirements
    success = run_command(f"{pip_command} install --upgrade pip", "Upgrading pip")
    if not success:
        return False
    
    success = run_command(f"{pip_command} install -r requirements.txt", "Installing dependencies")
    return success


def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom API endpoint (for OpenAI-compatible APIs)
# OPENAI_BASE_URL=https://api.openai.com/v1

# Application Settings
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_MAX_RETRIEVED_DOCS=5

# Model Settings
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_CHAT_MODEL=gpt-4o-mini
RAG_TEMPERATURE=0.7
"""
    
    if not Path(".env").exists():
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file - please edit it with your API key")
    else:
        print("✅ .env file already exists")


def main():
    """Main setup function."""
    print("🚀 RAG Application Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not activate_and_install():
        sys.exit(1)
    
    # Create sample environment file
    create_sample_env()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit the .env file with your OpenAI API key")
    print("2. Activate the virtual environment:")
    
    if os.name == 'nt':  # Windows
        print("   venv\\Scripts\\activate")
    else:  # Unix/Linux/MacOS
        print("   source venv/bin/activate")
    
    print("3. Run the application:")
    print("   streamlit run streamlit_app.py")
    print("   OR")
    print("   jupyter lab  # then open rag_demo.ipynb")
    
    print("\n📖 For more information, see README.md")


if __name__ == "__main__":
    main() 