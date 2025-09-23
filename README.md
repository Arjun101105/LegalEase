# 🏛️ LegalEase - Legal Text Simplification for Indian Citizens

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Offline](https://img.shields.io/badge/mode-offline-green.svg)]()
[![Privacy First](https://img.shields.io/badge/privacy-first-brightgreen.svg)]()

> 🎯 **Mission**: Convert complex Indian legal texts into simple, understandable English for common citizens

## 🌟 **What is LegalEase?**

LegalEase is an AI-powered tool that transforms complex Indian legal language into clear, everyday English. Whether you're reading a court judgment, understanding a contract, or deciphering a legal notice, LegalEase makes it accessible to everyone.

### 🎪 **Live Example**
```
📜 Legal Text:
"The plaintiff filed a writ petition under Article 32 of the Constitution seeking mandamus against the respondent for non-compliance with statutory obligations."

✨ LegalEase Simplified:
"In simple terms: The person filing the case submitted a formal legal request according to constitutional right to justice asking for court order to do something versus the person being sued for not following the rules with legal duties."
```

## 🚀 **Quick Start** (Cross-Platform)

### **🖥️ For Windows Users**
```cmd
# Clone the repository
git clone https://github.com/Arjun101105/LegalEase.git
cd LegalEase

# One-command setup (Windows)
setup_windows.bat

# Start using
run_legalease.bat --text "Your legal text here"
```

### **🐧 For Linux/Mac Users**
```bash
# Clone the repository
git clone https://github.com/Arjun101105/LegalEase.git
cd LegalEase

# One-command setup
python setup.py

# Start using
./run_legalease.sh --text "Your legal text here"
```

### **🐍 Manual Setup (All Platforms)**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run initial setup
python setup.py

# Start LegalEase
python src/cli_app.py --text "The appellant seeks relief under Article 226"
```

## 🛠️ **Tech Stack**
- **Model**: FLAN-T5 + InLegalBERT (Enhanced with instruction-tuning)
- **Language**: Python 3.8+
- **Libraries**: PyTorch, Transformers, Accelerate, SentencePiece, NLTK
- **Interface**: CLI + Tkinter GUI
- **Platform**: Cross-platform (Windows, Linux, macOS)
- **Dataset**: Enhanced training with 50+ legal→simple pairs

## 📁 **Project Structure**
```
LegalEase/
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed datasets
│   └── models/             # AI models (auto-downloaded)
├── src/
│   ├── cli_app.py          # Command-line interface
│   ├── gui_app.py          # Graphical interface
│   ├── training.py         # Model training
│   └── *.py               # Core functionality
├── improvements/           # Enhancement scripts
│   ├── upgrade_to_flan_t5.py
│   ├── simplified_retraining.py
│   └── enhanced_*.py      # Quality improvements
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
├── setup.py              # Cross-platform setup
├── run_legalease.sh      # Linux/Mac launcher
├── run_legalease.bat     # Windows launcher
└── setup_windows.bat     # Windows setup
```

## 🎮 **Usage Examples**

### **Command Line Interface**
```bash
# Windows
run_legalease.bat --text "The plaintiff filed a writ petition seeking mandamus"

# Linux/Mac
./run_legalease.sh --text "The plaintiff filed a writ petition seeking mandamus"

# Direct Python (all platforms)
python src/cli_app.py --text "Your legal text here"
```

### **Graphical Interface**
```bash
# Windows: Double-click run_legalease.bat or:
python src/gui_app.py

# Linux/Mac:
python src/gui_app.py
```

### **Batch Processing**
```bash
python src/cli_app.py --batch input_folder/ output_folder/
```

## 📊 Dataset Information

### InLegalBERT
- **Size**: ~400-500 MB (110M parameters)
- **Training Data**: 5.4M Indian legal documents (1950-2019)
- **Source**: Supreme Court and High Court judgments
- **Usage**: Base model for legal text understanding

### MILDSum_Samples
- **Size**: Sample subset of 3,122 judgment-summary pairs
- **Languages**: English judgments + Hindi summaries
- **Usage**: Training data for simplification (20-30 pairs for MVP)
- **Target**: Scale to 50-100 pairs by mid-September

## 🎯 Development Timeline
- **Step 1**: Environment setup and dataset download
- **Step 2**: Data preprocessing and simplification pairs creation
- **Step 3**: Model optimization for CPU-only inference
- **Step 4**: Tkinter GUI development
- **Step 5**: Testing and optimization for low-spec hardware

**Target**: MVP by early September 2025, full version by mid-September 2025

## 🔧 Hardware Requirements
- **RAM**: 4-8 GB (with model quantization)
- **CPU**: 4-core minimum
- **Storage**: 2-3 GB for models and data
- **GPU**: Not required (CPU-optimized)

## 🚀 Future Scope
- Hindi text simplification support
- Additional Indian legal datasets integration
- Q&A feature for clause explanations
- Web interface option
