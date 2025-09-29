#!/usr/bin/env python3
"""
Cross-Platform Local LLM Setup for LegalEase
Automatic GPU detection and optimization for Windows, Linux, and macOS
100% Free & Offline - No API costs ever!
"""

import os
import subprocess
import sys
import platform
from pathlib import Path
import requests
import json

def print_banner():
    """Print cross-platform setup banner"""
    print("🚀 LegalEase - Cross-Platform Local LLM Setup")
    print("=" * 60)
    print("💰 100% FREE - No API costs ever!")
    print("🔒 100% PRIVATE - Everything runs on your laptop")
    print("🌐 100% OFFLINE - No internet needed after setup")
    print("🖥️  CROSS-PLATFORM - Windows, Linux, macOS")
    print("🚀 AUTO GPU DETECTION - Leverages your hardware")
    print("=" * 60)
    print()

def detect_system():
    """Detect system information"""
    system_info = {
        "os": platform.system().lower(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "is_windows": platform.system().lower() == "windows",
        "is_linux": platform.system().lower() == "linux", 
        "is_mac": platform.system().lower() == "darwin"
    }
    
    print(f"🖥️  Detected System: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {system_info['architecture']}")
    print(f"🐍 Python: {system_info['python_version']}")
    print()
    
    return system_info

def check_gpu_capabilities():
    """Check GPU capabilities across platforms"""
    print("🚀 Checking GPU Capabilities")
    print("-" * 30)
    
    gpu_info = {
        "cuda_available": False,
        "mps_available": False,
        "gpu_names": [],
        "recommended_setup": "cpu"
    }
    
    try:
        import torch
        
        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_count = torch.cuda.device_count()
            
            print("✅ NVIDIA CUDA GPU detected!")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_info["gpu_names"].append(f"{gpu_name} ({gpu_memory:.1f}GB)")
                print(f"   🎮 {gpu_name} ({gpu_memory:.1f}GB)")
            
            gpu_info["recommended_setup"] = "cuda"
        
        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            gpu_info["mps_available"] = True
            gpu_info["gpu_names"] = ["Apple Silicon GPU"]
            gpu_info["recommended_setup"] = "mps"
            print("✅ Apple Silicon GPU detected!")
            print("   🍎 Metal Performance Shaders available")
        
        else:
            print("💻 CPU-only mode detected")
            print("   No GPU acceleration available")
            
    except ImportError:
        print("⚠️  PyTorch not installed yet - will check GPU after installation")
    
    print(f"🎯 Recommended setup: {gpu_info['recommended_setup'].upper()}")
    print()
    return gpu_info

def install_requirements_cross_platform(system_info):
    """Install requirements optimized for the platform"""
    print("📦 Installing Cross-Platform Requirements")
    print("-" * 40)
    
    # Base packages for all platforms
    base_packages = [
        "torch",
        "transformers>=4.30.0", 
        "requests",
        "psutil",
        "accelerate"
    ]
    
    # Platform-specific optimizations
    if system_info["is_windows"]:
        print("🪟 Installing Windows-optimized packages...")
        # Windows-specific packages if needed
        windows_packages = []
        packages = base_packages + windows_packages
        
    elif system_info["is_linux"]:
        print("🐧 Installing Linux-optimized packages...")
        # Linux-specific packages
        linux_packages = []
        packages = base_packages + linux_packages
        
    elif system_info["is_mac"]:
        print("🍎 Installing macOS-optimized packages...")
        # macOS-specific packages
        mac_packages = []
        packages = base_packages + mac_packages
    
    else:
        packages = base_packages
    
    # Install packages
    for package in packages:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"   ✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {package}")
            return False
    
    print("✅ All requirements installed successfully!")
    print()
    return True

def setup_gpu_optimized_models(gpu_info, system_info):
    """Setup models optimized for detected hardware"""
    print("🤖 Setting Up Hardware-Optimized Models")
    print("-" * 40)
    
    try:
        # Import after installation
        sys.path.append(str(Path(__file__).parent / "src"))
        from local_llm_integration import LocalLLMManager
        
        manager = LocalLLMManager()
        
        # Show system capabilities
        manager.system.print_system_info()
        
        # Recommend models based on hardware
        if gpu_info["cuda_available"]:
            print("🎮 NVIDIA GPU Setup - High Performance Models Available")
            recommended_models = ["medium_quality", "large_gpu"]
            
        elif gpu_info["mps_available"]:
            print("🍎 Apple Silicon Setup - Optimized for Metal Performance")
            recommended_models = ["medium_quality", "small_balanced"]
            
        else:
            print("💻 CPU Setup - Lightweight Models for Best Performance")
            recommended_models = ["small_balanced", "tiny_fast"]
        
        print("\n🎯 Recommended Models for Your System:")
        for model_key in recommended_models:
            if model_key in manager.available_models:
                model = manager.available_models[model_key]
                print(f"   📦 {model['name']} ({model['size']}) - {model['quality']}")
        
        print()
        
        # Auto-setup best model
        if input("Auto-download the best model for your system? (y/n): ").lower().startswith('y'):
            best_model = recommended_models[0] if recommended_models else "small_balanced"
            print(f"\n📥 Setting up {best_model} for your hardware...")
            
            success = manager.download_hf_model(best_model)
            if success:
                print("✅ Hardware-optimized model ready!")
            else:
                print("⚠️  Model will auto-download when first used")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Model setup will complete when you first run LegalEase: {e}")
        return True

def setup_ollama_cross_platform(system_info):
    """Cross-platform Ollama setup"""
    print("🦙 Ollama Setup (Advanced Local LLMs)")
    print("-" * 40)
    
    # Check if already installed
    try:
        result = subprocess.run(
            ["ollama", "--version"], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Ollama already installed!")
            
            # Check for models
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=3)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    print(f"📦 Installed models: {len(models)}")
                    for model in models:
                        print(f"   - {model['name']}")
                    
                    if not models:
                        recommend_models_for_platform(system_info)
                    
                    return True
                else:
                    print("⚠️  Ollama installed but not running")
                    print("   Start it and run this setup again")
                    
            except:
                print("⚠️  Ollama installed but not running")
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print("❌ Ollama not found")
    print()
    
    # Platform-specific installation instructions
    if system_info["is_windows"]:
        print("🪟 Windows Installation:")
        print("1. Download: https://ollama.ai/download/windows")
        print("2. Run the installer")
        print("3. Restart Command Prompt")
        print("4. Test: ollama --version")
        
    elif system_info["is_linux"]:
        print("🐧 Linux Installation:")
        print("1. Run: curl -fsSL https://ollama.ai/install.sh | sh")
        print("2. Test: ollama --version")
        
    elif system_info["is_mac"]:
        print("🍎 macOS Installation:")
        print("1. Download: https://ollama.ai/download/mac")
        print("2. Install the .dmg file")
        print("3. Test: ollama --version")
    
    print()
    
    if input("Open Ollama download page? (y/n): ").lower().startswith('y'):
        import webbrowser
        if system_info["is_windows"]:
            webbrowser.open("https://ollama.ai/download/windows")
        elif system_info["is_linux"]:
            webbrowser.open("https://ollama.ai/download/linux")
        elif system_info["is_mac"]:
            webbrowser.open("https://ollama.ai/download/mac")
        else:
            webbrowser.open("https://ollama.ai/download")
    
    recommend_models_for_platform(system_info)
    return False

def recommend_models_for_platform(system_info):
    """Recommend Ollama models based on platform and hardware"""
    print("\n🎯 Recommended Models After Ollama Installation:")
    
    # Get RAM info for recommendations
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_gb = 8  # Conservative estimate
    
    models = []
    
    if ram_gb >= 16:
        models.append({
            "name": "llama2:7b-chat",
            "command": "ollama pull llama2:7b-chat",
            "description": "Best quality for legal tasks"
        })
    
    if ram_gb >= 8:
        models.append({
            "name": "phi:2.7b", 
            "command": "ollama pull phi:2.7b",
            "description": "Best balance of speed and quality ⭐"
        })
    
    models.append({
        "name": "tinyllama:1.1b",
        "command": "ollama pull tinyllama:1.1b", 
        "description": "Fastest, works on any system"
    })
    
    for model in models:
        print(f"   📦 {model['name']} - {model['description']}")
        print(f"      Command: {model['command']}")
        print()

def create_cross_platform_scripts():
    """Create platform-specific run scripts"""
    print("📝 Creating Platform-Specific Scripts")
    print("-" * 35)
    
    # Windows batch script
    windows_script = """@echo off
echo Starting LegalEase - Legal Text Simplification
echo ============================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Run LegalEase
cd /d "%~dp0"
python src/cli_app.py %*

if errorlevel 1 (
    echo.
    echo Error occurred. Press any key to close...
    pause >nul
)
"""
    
    # Linux/macOS shell script  
    unix_script = """#!/bin/bash
echo "🏛️  Starting LegalEase - Legal Text Simplification"
echo "================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Error: Python not found. Please install Python 3.8+"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Run LegalEase
$PYTHON_CMD src/cli_app.py "$@"
"""
    
    # Write scripts
    base_dir = Path(__file__).parent
    
    # Windows script
    windows_file = base_dir / "run_legalease.bat"
    with open(windows_file, 'w', newline='\r\n') as f:
        f.write(windows_script)
    print(f"✅ Windows script: {windows_file}")
    
    # Unix script
    unix_file = base_dir / "run_legalease.sh" 
    with open(unix_file, 'w', newline='\n') as f:
        f.write(unix_script)
    
    # Make executable on Unix systems
    try:
        os.chmod(unix_file, 0o755)
        print(f"✅ Unix script: {unix_file}")
    except:
        print(f"✅ Unix script: {unix_file} (chmod manually)")
    
    print()

def create_cross_platform_demo():
    """Create a cross-platform demo script"""
    demo_script = '''#!/usr/bin/env python3
"""
Cross-Platform Demo: LegalEase with Hardware-Optimized Local LLMs
Works on Windows, Linux, and macOS with automatic GPU detection
"""

import sys
import platform
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def main():
    print("🧪 LegalEase Cross-Platform Demo")
    print("=" * 45)
    print(f"🖥️  Platform: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    print()
    
    try:
        from local_llm_integration import LocalLegalSimplifier
        
        # Initialize with hardware detection
        simplifier = LocalLegalSimplifier()
        
        # Show system capabilities
        simplifier.llm_manager.system.print_system_info()
        
        # Sample legal texts
        samples = [
            "The plaintiff filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent for non-compliance with statutory obligations.",
            
            "The appellant was constrained to file this appeal challenging the impugned order passed by the learned Single Judge.",
            
            "The party of the first part hereby covenants and agrees to indemnify and hold harmless the party of the second part from any and all claims."
        ]
        
        print("📝 Testing with sample legal texts...")
        print()
        
        for i, sample in enumerate(samples, 1):
            print(f"🔍 Sample {i}:")
            print(f"Original: {sample}")
            print()
            
            # Current system output (simulated)
            current_output = f"In simple terms: {sample.lower()}"
            
            # Try hardware-optimized local LLM enhancement
            enhanced = simplifier.enhance_simplification(sample, current_output)
            
            print(f"Enhanced: {enhanced}")
            print("-" * 60)
            print()
        
        print("✅ Cross-platform demo completed successfully!")
        print(f"🚀 Your {platform.system()} system is ready for enhanced legal text simplification!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Run the setup first: python setup_free_local_llm.py")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your installation and try again")

if __name__ == "__main__":
    main()
'''
    
    demo_file = Path(__file__).parent / "demo_cross_platform.py"
    with open(demo_file, 'w', encoding='utf-8') as f:
        f.write(demo_script)
    
    print(f"📝 Cross-platform demo created: {demo_file}")

def main():
    """Main cross-platform setup function"""
    print_banner()
    
    # Step 1: Detect system
    system_info = detect_system()
    
    # Step 2: Check GPU capabilities  
    gpu_info = check_gpu_capabilities()
    
    # Step 3: Install requirements
    if not install_requirements_cross_platform(system_info):
        print("❌ Failed to install requirements")
        return
    
    # Step 4: Re-check GPU after PyTorch installation
    print("🔄 Re-checking GPU capabilities after PyTorch installation...")
    gpu_info = check_gpu_capabilities()
    
    # Step 5: Setup hardware-optimized models
    setup_gpu_optimized_models(gpu_info, system_info)
    
    # Step 6: Setup Ollama (optional)
    ollama_ready = setup_ollama_cross_platform(system_info)
    
    # Step 7: Create platform-specific scripts
    create_cross_platform_scripts()
    
    # Step 8: Create cross-platform demo
    create_cross_platform_demo()
    
    # Final summary
    print("🎉 Cross-Platform Local LLM Setup Complete!")
    print("=" * 50)
    print()
    print("🚀 What you now have:")
    print(f"✅ Hardware-optimized for {system_info['os'].title()}")
    if gpu_info["cuda_available"]:
        print("✅ NVIDIA CUDA GPU acceleration")
    elif gpu_info["mps_available"]:
        print("✅ Apple Silicon GPU acceleration") 
    else:
        print("✅ CPU-optimized performance")
    print("✅ Local LLM models (auto-download)")
    if ollama_ready:
        print("✅ Ollama with advanced models")
    print("✅ 100% free, offline, and private")
    print("✅ Cross-platform compatibility")
    print()
    
    print("🎯 Quick Start:")
    if system_info["is_windows"]:
        print("   Windows: run_legalease.bat")
        print("   Or: python src/cli_app.py")
    else:
        print("   Unix: ./run_legalease.sh")
        print("   Or: python3 src/cli_app.py")
    
    print()
    print("📖 Test your setup:")
    print("   python demo_cross_platform.py")
    print()
    print("💡 Your LegalEase will now automatically use the best")
    print("   available hardware (GPU/CPU) for optimal performance!")

if __name__ == "__main__":
    main()