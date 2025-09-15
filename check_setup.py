#!/usr/bin/env python3
"""
Check if models are downloaded and provide guidance
"""

import os
from pathlib import Path

def check_setup():
    """Check if LegalEase is properly set up"""
    project_root = Path(__file__).parent
    
    print("🔍 Checking LegalEase Setup...")
    print("=" * 40)
    
    checks = [
        ("Virtual environment", project_root / "venv"),
        ("Requirements file", project_root / "requirements.txt"),
        ("Data directory", project_root / "data"),
        ("Models directory", project_root / "data" / "models"),
        ("Source code", project_root / "src" / "cli_app.py"),
        ("Download script", project_root / "scripts" / "download_datasets.py"),
    ]
    
    all_good = True
    
    for name, path in checks:
        if path.exists():
            print(f"✅ {name}: Found")
        else:
            print(f"❌ {name}: Missing")
            all_good = False
    
    print("\n" + "=" * 40)
    
    if all_good:
        print("🎉 LegalEase setup looks good!")
        print("\n💻 To use LegalEase:")
        print("   source venv/bin/activate")
        print("   python src/cli_app.py --examples")
    else:
        print("⚠️  Setup incomplete. Please run:")
        print("   python setup.py")
    
    return all_good

if __name__ == "__main__":
    check_setup()
