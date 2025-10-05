# LegalEase Models Directory

This directory contains AI models used by LegalEase. The models are automatically downloaded when you run the setup scripts.

## 🤖 Model Structure

```
models/
├── InLegalBERT/              # Legal domain understanding (110MB)
├── flan_t5_enhanced/         # Enhanced T5 model (308MB)
├── local_llms/               # Local LLM models (varies)
│   ├── small_fast/           # GPT-2 Small (117MB)
│   ├── medium_balanced/      # DistilGPT2 (319MB)
│   └── large_quality/        # GPT-2 (548MB)
└── simplification_config.json
```

## 📥 How to Download Models

Models are automatically downloaded when you run:

```bash
# Windows
python setup_free_local_llm.py

# Linux/macOS
python3 setup_free_local_llm.py
```

## 🚀 Manual Download (if needed)

If automatic download fails, you can manually download:

1. **InLegalBERT**: `python scripts/download_datasets.py`
2. **Local LLMs**: Models download automatically on first use
3. **Ollama Models**: Install Ollama and run `ollama pull phi:2.7b`

## 💾 Storage Requirements

- **Minimum**: 1GB (basic models only)
- **Recommended**: 5GB (includes enhanced models)
- **Full Setup**: 10GB (includes all optional models)

## ⚠️ Important Notes

- Models are not included in git repository due to size
- First-time setup requires internet connection
- Models are cached locally for offline use
- GPU-optimized models selected automatically based on hardware