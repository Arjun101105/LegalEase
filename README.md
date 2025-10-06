# 🏛️ LegalEase - AI-Powered Legal Text Simplification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)](https://github.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com)

**Transform complex Indian legal documents into simple, understandable English**

---

## 🌟 **What is LegalEase?**

LegalEase is an AI-powered tool that transforms complex Indian legal language into clear, everyday English. Whether you're reading court judgments, contracts, or legal notices, LegalEase makes it accessible to everyone.

### ✨ **Live Example**
```
📜 Complex Legal Text:
"The plaintiff filed a writ petition under Article 32 of the Constitution seeking enforcement of fundamental rights guaranteed under Articles 14, 19, and 21, alleging violation thereof by the respondent state authorities."

🎯 LegalEase Simplified:
"A person went to court asking for their basic rights to be protected. They felt that the government was not respecting their rights to equality, freedom of speech, and justice."
```

---

## 🚀 **Key Features**

- ✅ **Multi-Layer AI Processing**: T5 + Rule-based + LLM Enhancement
- ✅ **Real Legal Training**: Fine-tuned on actual court cases (MILDSum dataset)
- ✅ **Multiple Interfaces**: Web UI, CLI, GUI, and REST API
- ✅ **OCR Support**: Process PDFs and images of legal documents
- ✅ **Privacy-Focused**: Works completely offline (optional LLM enhancement)
- ✅ **Production Ready**: FastAPI backend with comprehensive error handling

---

## 📋 **Quick Start**

### **Prerequisites**
- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- Internet connection for initial setup

### **1. Clone Repository**
```bash
git clone https://github.com/Arjun101105/LegalEase.git
cd LegalEase
```

### **2. Setup Environment**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

### **3. Download Pre-trained Models**
```bash
# Download required models (this will take a few minutes)
python scripts/download_datasets.py
```

### **4. Start the Application**

#### **Option A: Web Interface (Recommended)**
```bash
# Start backend server
python backend/app.py &

# Start frontend server  
python3 -m http.server 3000 &

# Open in browser: http://localhost:3000/web_interface.html
```

#### **Option B: Command Line Interface**
```bash
python src/cli_app.py --text "Your legal text here" --enhanced
```

#### **Option C: Desktop GUI**
```bash
python src/gui_app.py
```

---

## 🏗️ **Project Structure**

```
LegalEase/
├── 📁 backend/              # FastAPI backend server
│   ├── app.py               # Main FastAPI application
│   ├── requirements.txt     # Backend dependencies
│   └── config.py            # Configuration settings
├── 📁 src/                  # Core application logic
│   ├── cli_app.py           # Command line interface
│   ├── gui_app.py           # Desktop GUI application
│   ├── training.py          # Model training utilities
│   ├── llm_integration.py   # LLM enhancement module
│   └── ocr_processor.py     # OCR processing for documents
├── 📁 scripts/              # Utility scripts
│   ├── download_datasets.py # Download required models/data
│   └── evaluate_*.py        # Performance evaluation tools
├── 📁 data/                 # Data directory (created after setup)
│   ├── models/              # Pre-trained models (downloaded)
│   └── processed/           # Processed datasets
├── web_interface.html       # Web UI (single file)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🔧 **API Documentation**

### **REST API Endpoints**

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation.

#### **Text Simplification**
```bash
curl -X POST "http://localhost:8000/api/simplify-text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The respondent state authorities are hereby directed to implement the mandamus",
    "use_llm": true
  }'
```

#### **Health Check**
```bash
curl -X GET "http://localhost:8000/health"
```

#### **OCR Processing**
```bash
curl -X POST "http://localhost:8000/api/ocr" \
  -F "file=@legal_document.pdf" \
  -F "simplify_text=true"
```

---

## 🧠 **How It Works**

LegalEase uses a sophisticated multi-layer approach:

1. **📚 InLegalBERT**: Understands legal context and terminology
2. **🎯 Fine-tuned T5**: Trained on real legal cases (MILDSum dataset)
3. **🔧 Rule-based Processing**: Handles common legal phrases
4. **🤖 LLM Enhancement** (Optional): Provides contextual explanations

### **Training Data**
- **MILDSum Dataset**: 10 real court cases with 3,079 words average
- **Legal Terminology Database**: Common Indian legal terms and phrases
- **Constitutional Articles**: Rights and procedures references

---

## 📊 **Performance**

- **Speed**: ~1-2 seconds for rule-based, ~30-60 seconds with LLM
- **Accuracy**: Maintains legal precision while improving readability
- **Coverage**: Handles constitutional law, property law, criminal procedure, contracts
- **Reliability**: Fallback mechanisms ensure consistent results

---

## 🛠️ **Development**

### **Training Your Own Model**
```bash
# Train on new legal data
python src/training.py --dataset your_dataset.csv --epochs 3
```

### **Running Tests**
```bash
# Evaluate performance
python scripts/evaluate_text_simplification.py
```

### **Contributing**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📁 **File Downloads**

The following files will be automatically downloaded during setup:

- **InLegalBERT** (~419MB): Legal context understanding
- **T5 Legal Model** (~232MB): Core simplification model
- **MILDSum Dataset** (~10MB): Training and evaluation data

*Note: Models are cached locally and only downloaded once.*

---

## 🔒 **Privacy & Security**

- ✅ **Offline Capable**: Core functionality works without internet
- ✅ **No Data Storage**: Texts are processed and discarded
- ✅ **Local Processing**: No data sent to external servers
- ✅ **Optional LLM**: Enhanced features can use local Ollama

---

## 🐛 **Troubleshooting**

### **Common Issues**

**1. ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
# or
venv\\Scripts\\activate   # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**2. CUDA/GPU Issues**
```bash
# For CPU-only usage
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**3. Model Download Errors**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python scripts/download_datasets.py
```

**4. Port Already in Use**
```bash
# Kill existing processes
pkill -f "python backend/app.py"
pkill -f "python3 -m http.server"
```

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Hugging Face**: For transformer models and datasets
- **MILDSum**: For legal simplification dataset
- **FastAPI**: For the robust API framework
- **Indian Legal Community**: For domain expertise and feedback

---

## 📧 **Contact**

- **GitHub Issues**: [Report bugs or request features](https://github.com/Arjun101105/LegalEase/issues)
- **Email**: [Your contact email]
- **Documentation**: [Project Wiki](https://github.com/Arjun101105/LegalEase/wiki)

---

## 🔮 **Future Plans**

- [ ] Support for regional Indian languages
- [ ] Mobile app development
- [ ] Integration with legal databases
- [ ] Real-time document collaboration
- [ ] Advanced legal analysis features

---

**Made with ❤️ for the Indian legal community**