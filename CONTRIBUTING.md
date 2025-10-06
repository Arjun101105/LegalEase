# Contributing to LegalEase

Thank you for your interest in contributing to LegalEase! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LegalEase.git
   cd LegalEase
   ```
3. **Set up the development environment**:
   ```bash
   ./setup.sh
   ```

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- Git
- 4GB+ RAM (8GB+ recommended)

### Local Development
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## 📝 Making Changes

### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Comment complex logic

### Before Submitting
```bash
# Format code
black src/ backend/ scripts/

# Check style
flake8 src/ backend/ scripts/

# Run tests
pytest tests/
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_simplification.py

# Run with coverage
pytest --cov=src tests/
```

### Adding Tests
- Write tests for new features
- Ensure existing tests pass
- Aim for good test coverage

## 📋 Submitting Changes

### Pull Request Process
1. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** and commit:
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

3. **Push to your fork**:
   ```bash
   git push origin feature/amazing-feature
   ```

4. **Create a Pull Request** on GitHub

### PR Guidelines
- **Clear title**: Describe what the PR does
- **Detailed description**: Explain the changes and why
- **Link issues**: Reference related issues
- **Test coverage**: Include tests for new features
- **Documentation**: Update docs if needed

## 🐛 Reporting Issues

### Bug Reports
Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Error messages and logs

### Feature Requests
Include:
- Clear description of the feature
- Use case and benefits
- Possible implementation approaches

## 📚 Areas for Contribution

### High Priority
- **Performance optimization**: Improve model inference speed
- **Language support**: Add regional Indian languages
- **UI/UX improvements**: Enhance web interface
- **Documentation**: Improve guides and examples

### Medium Priority
- **Test coverage**: Add more comprehensive tests
- **Error handling**: Improve error messages and recovery
- **Mobile support**: Responsive design improvements
- **Integration**: API clients and plugins

### Ideas Welcome
- **New features**: OCR improvements, batch processing
- **Tools**: Development and deployment utilities
- **Examples**: Legal use case demonstrations

## 🏗️ Project Structure

```
LegalEase/
├── backend/          # FastAPI backend
├── src/             # Core application logic
├── scripts/         # Utility scripts
├── tests/           # Test files
├── docs/            # Documentation
└── frontend/        # Web interface
```

## 📖 Documentation

### Adding Documentation
- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Create examples for new features
- Update API documentation

### Building Docs Locally
```bash
# Install docs dependencies
pip install mkdocs mkdocs-material

# Serve docs locally
mkdocs serve
```

## 🔒 Security

### Reporting Security Issues
- **Do not** create public issues for security vulnerabilities
- Email security concerns to [security email]
- Include detailed description and steps to reproduce

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in the CONTRIBUTORS.md file
- Mentioned in release notes
- Given credit in the project documentation

## 💬 Community

### Communication
- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Discord**: [If you have a Discord server]

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's code of conduct

## 🔄 Development Workflow

### Typical Workflow
1. Check existing issues and PRs
2. Create or comment on an issue
3. Fork and create a feature branch
4. Develop and test your changes
5. Submit a pull request
6. Respond to review feedback
7. Celebrate when merged! 🎉

Thank you for contributing to LegalEase! Your contributions help make legal information more accessible to everyone.