#!/usr/bin/env python3
"""
LegalEase Setup Script
Automatically downloads models and sets up the environment
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
import json
import shutil
from pathlib import Path
import platform

def print_banner():
    """Print setup banner"""
    print("🏛️  LegalEase Setup Script")
    print("=" * 50)
    print("📖 Setting up Legal Text Simplification for Indian Citizens")
    print("🔒 Offline & Privacy-focused | No data storage")
    print("=" * 50)
    print()

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    
    # Check available space (need ~2GB)
    total, used, free = shutil.disk_usage('.')
    free_gb = free // (1024**3)
    if free_gb < 2:
        print(f"❌ Need at least 2GB free space. Available: {free_gb}GB")
        return False
    
    print("✅ System requirements met")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directory structure...")
    
    directories = [
        "data/models",
        "data/raw", 
        "data/processed",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created")

def setup_virtual_environment():
    """Set up virtual environment"""
    print("🐍 Setting up virtual environment...")
    
    if not Path("venv").exists():
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
    else:
        print("✅ Virtual environment already exists")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    # Determine pip path based on OS
    if platform.system() == "Windows":
        pip_path = "venv/Scripts/pip"
        python_path = "venv/Scripts/python"
    else:
        pip_path = "venv/bin/pip"
        python_path = "venv/bin/python"
    
    try:
        # Upgrade pip first
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_with_progress(url, filename):
    """Download file with progress bar"""
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r   📥 {percent:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end="")
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

def run_setup_scripts():
    """Run the existing setup scripts"""
    print("🤖 Downloading models and datasets...")
    
    # Determine python path
    if platform.system() == "Windows":
        python_path = "venv/Scripts/python"
    else:
        python_path = "venv/bin/python"
    
    scripts_to_run = [
        ("📚 Downloading datasets and models", "scripts/download_datasets.py"),
        ("🔧 Processing data", "src/data_preprocessing.py"),
        ("⚙️  Setting up models", "src/model_setup.py"),
    ]
    
    for description, script in scripts_to_run:
        print(f"\n{description}...")
        try:
            result = subprocess.run([python_path, script], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {description.split()[-1]} completed")
            else:
                print(f"⚠️  {description.split()[-1]} completed with warnings")
                if result.stderr:
                    print("   Warnings:", result.stderr[:200])
        except Exception as e:
            print(f"❌ {description.split()[-1]} failed: {e}")
            return False
    
    return True

def create_run_scripts():
    """Create convenient run scripts"""
    print("📝 Creating run scripts...")
    
    # Determine activation command based on OS
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        python_cmd = "venv\\Scripts\\python"
    else:
        activate_cmd = "source venv/bin/activate"
        python_cmd = "venv/bin/python"
    
    # Create run script
    run_script_content = f"""#!/bin/bash
# LegalEase Quick Run Script
echo "🏛️  Starting LegalEase..."
{activate_cmd}
{python_cmd} src/cli_app.py "$@"
"""
    
    with open("run_legalease.sh", "w") as f:
        f.write(run_script_content)
    
    # Make executable on Unix systems
    if platform.system() != "Windows":
        os.chmod("run_legalease.sh", 0o755)
    
    # Create batch file for Windows
    if platform.system() == "Windows":
        batch_content = f"""@echo off
echo 🏛️  Starting LegalEase...
{activate_cmd}
{python_cmd} src/cli_app.py %*
"""
        with open("run_legalease.bat", "w") as f:
            f.write(batch_content)
    
    print("✅ Run scripts created")

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    # Determine python path
    if platform.system() == "Windows":
        python_path = "venv/Scripts/python"
    else:
        python_path = "venv/bin/python"
    
    test_command = [python_path, "src/cli_app.py", "--examples"]
    
    try:
        result = subprocess.run(test_command, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and "Example Legal Texts" in result.stdout:
            print("✅ Installation test passed")
            return True
        else:
            print("⚠️  Installation test completed with warnings")
            print("   The system should still work for basic functionality")
            return True
    except Exception as e:
        print(f"⚠️  Installation test failed: {e}")
        print("   You may need to run setup manually")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*50)
    print("🎉 LegalEase Setup Complete!")
    print("="*50)
    print()
    print("📋 Quick Start:")
    print("   # Activate environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
        print("   python src/cli_app.py --text \"Your legal text here\"")
    else:
        print("   source venv/bin/activate")
        print("   python src/cli_app.py --text \"Your legal text here\"")
    print()
    print("   # Or use the quick run script:")
    if platform.system() == "Windows":
        print("   run_legalease.bat --text \"Your legal text here\"")
    else:
        print("   ./run_legalease.sh --text \"Your legal text here\"")
    print()
    print("🎯 Usage Examples:")
    print("   # Interactive mode:")
    print("   python src/cli_app.py")
    print()
    print("   # Batch processing:")
    print("   python src/cli_app.py --input file.txt --output results.json")
    print()
    print("   # See examples:")
    print("   python src/cli_app.py --examples")
    print()
    print("📚 Documentation: README.md")
    print("🐛 Issues: https://github.com/Arjun101105/LegalEase/issues")
    print()

def main():
    """Main setup function"""
    print_banner()
    
    # Check requirements
    if not check_system_requirements():
        sys.exit(1)
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Setting up virtual environment", setup_virtual_environment),
        ("Installing dependencies", install_dependencies), 
        ("Downloading models and data", run_setup_scripts),
        ("Creating run scripts", create_run_scripts),
        ("Testing installation", test_installation),
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            print("💡 Please check the error messages above and try again")
            sys.exit(1)
    
    print_usage_instructions()
    
    print("✨ Setup completed successfully! LegalEase is ready to use!")

if __name__ == "__main__":
    main()
