# Contributing to UCAN

Thank you for your interest in contributing to UCAN! This document provides guidelines for contributing to the project.

## 🚀 Quick Start for Contributors

1. **Fork the repository** and clone your fork
2. **Create a virtual environment** and install dependencies
3. **Create a new branch** for your feature/fix
4. **Make your changes** following our guidelines
5. **Test your changes** thoroughly
6. **Submit a pull request** with a clear description

## 🛠️ Development Setup

### Environment Setup
```bash
# Clone your fork
git clone [YOUR_FORK_URL]
cd UCAN

# Create conda environment
conda env create -f environment.yml
conda activate ucan

# Or use pip
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Development Dependencies
```bash
# Install additional development tools
pip install pytest black flake8 jupyter
```

## 📝 Code Style Guidelines

### Python Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add type hints where appropriate
- Maximum line length: 88 characters (Black formatter default)

### Code Formatting
We use Black for code formatting:
```bash
# Format all Python files
black .

# Check formatting without making changes
black --check .
```

### Linting
We use flake8 for linting:
```bash
flake8 --max-line-length=88 --ignore=E203,W503 .
```

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_noises.py
```

### Writing Tests
- Write tests for all new functionality
- Follow the existing test structure
- Use descriptive test names
- Include edge cases and error conditions

Example test structure:
```python
def test_gaussian_noise_generation():
    """Test that Gaussian noise is generated correctly."""
    # Arrange
    noise_generator = GaussianNoise(sigma=1.0)
    
    # Act
    noise = noise_generator.sample((32, 32, 3))
    
    # Assert
    assert noise.shape == (32, 32, 3)
    assert abs(noise.std() - 1.0) < 0.1
```

## 📋 Types of Contributions

### 🐛 Bug Reports
When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, etc.)
- Relevant code snippets or error messages

### ✨ Feature Requests
For new features, please provide:
- Clear description of the feature
- Use case and motivation
- Proposed API or interface
- Any relevant research papers or references

### 🔧 Code Contributions

#### New Noise Generators
If adding a new NPG method:
1. Create a new class inheriting from appropriate base class
2. Implement required methods
3. Add comprehensive tests
4. Update documentation
5. Add example usage

#### New Architectures
For new model architectures:
1. Add to `archs/` directory
2. Register in `architectures.py`
3. Include proper documentation
4. Test with existing NPG methods

#### Experimental Features
When adding experimental features:
1. Mark clearly as experimental
2. Include thorough documentation
3. Add configuration options
4. Provide usage examples

## 📚 Documentation Guidelines

### Code Documentation
- Use docstrings for all public functions and classes
- Follow Google or NumPy docstring format
- Include parameter types and return values
- Provide usage examples for complex functions

Example:
```python
def generate_anisotropic_noise(x: torch.Tensor, sigma: torch.Tensor, 
                             mu: torch.Tensor) -> torch.Tensor:
    """Generate anisotropic noise for given input.
    
    Args:
        x: Input tensor of shape (batch_size, channels, height, width)
        sigma: Standard deviation tensor of same shape as x
        mu: Mean tensor of same shape as x
        
    Returns:
        torch.Tensor: Anisotropic noise tensor of same shape as x
        
    Example:
        >>> x = torch.randn(1, 3, 32, 32)
        >>> sigma = torch.ones_like(x) * 0.5
        >>> mu = torch.zeros_like(x)
        >>> noise = generate_anisotropic_noise(x, sigma, mu)
    """
```

### README and Markdown Files
- Use clear, concise language
- Include code examples
- Add relevant badges and links
- Keep table of contents updated for long documents

## 🏃‍♂️ Performance Guidelines

### Code Performance
- Profile code for performance bottlenecks
- Use vectorized operations when possible
- Avoid unnecessary loops or computations
- Consider memory usage for large datasets

### Experimental Performance
- Include timing benchmarks for new methods
- Compare against existing baselines
- Document computational complexity
- Provide memory usage estimates

## 🔄 Pull Request Process

### Before Submitting
1. **Test thoroughly**: Run all tests and ensure they pass
2. **Update documentation**: Add/update docstrings and README
3. **Check code style**: Run Black and flake8
4. **Write clear commit messages**: Use conventional commit format

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added new tests
- [ ] All tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process
1. **Automated checks**: All CI checks must pass
2. **Code review**: At least one maintainer review required
3. **Testing**: Ensure all tests pass on multiple environments
4. **Documentation**: Verify documentation is updated

## 🚨 Issue Guidelines

### Issue Labels
We use the following labels:
- `bug`: Something isn't working
- `enhancement`: New feature or request  
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `question`: Further information requested

### Issue Templates
Use appropriate issue templates when available:
- Bug report template for bugs
- Feature request template for enhancements
- Question template for questions

## 🌟 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Paper acknowledgments (for significant contributions)
- Release notes for major contributions

## 📞 Getting Help

If you need help contributing:
- Open an issue with the `question` label
- Join discussions in existing issues
- Contact maintainers directly

## 📄 License

By contributing to UCAN, you agree that your contributions will be licensed under the same MIT License that covers the project.