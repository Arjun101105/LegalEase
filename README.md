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

## 🚀 **Quick Start**

### **One-Command Setup**
```bash
# Clone and setup everything automatically
git clone https://github.com/Arjun101105/LegalEase.git
cd LegalEase
python setup.py
```

### **Start Using Immediately**
```bash
# Quick run
./run_legalease.sh --text "Your legal text here"

# Or activate environment manually
source venv/bin/activate
python src/cli_app.py --text "The appellant seeks relief under Article 226"
```

## 🛠️ Tech Stack
- **Model**: InLegalBERT (Hugging Face Transformers)
- **Language**: Python 3.8+
- **Libraries**: PyTorch (CPU), NLTK, IndicNLP, NumPy, Pandas
- **Interface**: Tkinter GUI
- **Dataset**: MILDSum_Samples (20-30 pairs for MVP)

## 📁 Project Structure
```
LegalEase-2/
├── data/
│   ├── raw/                 # Original datasets
│   ├── processed/           # Preprocessed datasets
│   └── models/             # Downloaded models
├── src/
│   ├── data_preprocessing.py
│   ├── model_setup.py
│   ├── training.py
│   └── gui_app.py
├── scripts/
│   └── download_datasets.py
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Datasets**
   ```bash
   python scripts/download_datasets.py
   ```

3. **Preprocess Data**
   ```bash
   python src/data_preprocessing.py
   ```

4. **Setup Model**
   ```bash
   python src/model_setup.py
   ```

5. **Run Training** (Optional - for fine-tuning)
   ```bash
   python src/training.py
   ```

6. **Launch GUI**
   ```bash
   python src/gui_app.py
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
