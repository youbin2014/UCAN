# UCAN: Towards Strong Certified Defense with Asymmetric Randomization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

This repository contains the official implementation of **UCAN: Towards Strong Certified Defense with Asymmetric Randomization**, providing code for reproducible certified adversarial robustness experiments.

## 📖 Paper Information

**Abstract**: This work presents UCAN, a unified framework for customizing anisotropic noise in randomized smoothing to achieve stronger certified adversarial robustness. We propose three novel Noise Parameter Generators (NPGs) with different optimality levels and provide theoretical guarantees for anisotropic randomized smoothing.

**Key Contributions**:
- Universal theory for anisotropic randomized smoothing based on linear transformations
- Three NPG methods with different optimality-efficiency trade-offs
- Certification-wise approach ensuring soundness without memory overhead
- Significant improvements in certified accuracy across multiple datasets

## 🚀 Quick Start

### Environment Setup

#### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone [ANONYMOUS_REPO_URL]
cd UCAN

# Create conda environment
conda env create -f environment.yml
conda activate ucan
```

#### Option 2: Using pip
```bash
# Clone the repository
git clone [ANONYMOUS_REPO_URL]
cd UCAN

# Install dependencies
pip install -r requirements.txt
```

### Quick Demo

```bash
# Train a certification-wise model on CIFAR-10
python train_certification_noise.py cifar10 cifar_resnet110 ./model_saved/ \
    --method="PersNoise_isoR" --lr=0.01 --batch=100 --sigma=1.0 \
    --epochs=200 --gpu="0" --noise_name="Gaussian"

# Certify the test set
python certification_certification_noise.py cifar10 cifar_resnet110 \
    --method="PersNoise_isoR" --batch=1000 --sigma=1.0 --gpu="0" \
    --norm=2 --noise_name="Gaussian"
```

## 📁 Project Structure

```
UCAN/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                   # Conda environment
├── examples/                         # Example scripts and notebooks
│   ├── quick_start.py               # Minimal working example
│   └── demo.ipynb                   # Interactive demo
├── archs/                           # Neural network architectures
│   └── cifar_resnet.py             # ResNet for CIFAR
├── utils/                           # Utility functions
│   ├── model_prepare.py            # Model preparation utilities
│   ├── plot_examples.py            # Visualization utilities
│   └── plot_runtime.py             # Runtime analysis
├── model_saved/                     # Pre-trained models directory
├── results/                         # Experimental results
└── Core Implementation Files:
    ├── architectures.py             # NPG architectures
    ├── noisegenerator.py           # Noise parameter generators
    ├── noises.py                   # Noise distribution definitions
    ├── datasets.py                 # Dataset loading and preprocessing
    ├── core.py                     # Core UCAN certification
    ├── core_baseline.py            # Baseline certification (Cohen et al.)
    └── Training & Certification Scripts:
        ├── train_*.py              # Training scripts for each NPG method
        └── certification_*.py      # Certification scripts for each method
```

## 🧪 Experiments & Methods

### Three NPG Methods

1. **Pattern-wise Anisotropic Noise** (Low optimality)
   - Fixed hand-crafted spatial patterns
   - No training required, inference-free
   - Basic but computationally efficient

2. **Dataset-wise Anisotropic Noise** (Moderate optimality)
   - Learned parameters optimized for entire dataset
   - Pre-training required, one-time inference
   - Balanced performance-efficiency trade-off

3. **Certification-wise Anisotropic Noise** (High optimality)
   - Input-specific parameter optimization
   - Per-input inference required
   - Maximum adaptation capability

### Supported Datasets & Models

- **Datasets**: MNIST, CIFAR-10, ImageNet
- **Architectures**: ResNet (various depths), CNN architectures
- **Threat Models**: ℓ₁, ℓ₂, ℓ∞ perturbations

## 🔧 Detailed Usage

### Training Models

#### 1. Certification-wise NPG Training
```bash
python train_certification_noise.py cifar10 cifar_resnet110 ./model_saved/ \
    --method="PersNoise_isoR" \
    --lr=0.01 \
    --batch=100 \
    --sigma=1.0 \
    --epochs=200 \
    --workers=16 \
    --lr_step_size=50 \
    --gpu="0" \
    --noise_name="Gaussian" \
    --IsoMeasure=True
```

#### 2. Dataset-wise NPG Training  
```bash
python train_dataset_noise.py cifar10 cifar_resnet110 ./model_saved/ \
    --method="UniversalNoise" \
    --lr=0.01 \
    --batch=100 \
    --sigma=1.0 \
    --epochs=200 \
    --gpu="0"
```

#### 3. Pattern-wise NPG Training
```bash
python train_pattern_noise.py cifar10 cifar_resnet110 ./model_saved/ \
    --method="PreassignedNoise" \
    --pattern_type="center_focus" \
    --lr=0.01 \
    --batch=100 \
    --epochs=200 \
    --gpu="0"
```

### Certification (Testing)

#### Certification-wise Method
```bash
python certification_certification_noise.py cifar10 cifar_resnet110 \
    --method="PersNoise_isoR" \
    --batch=1000 \
    --sigma=1.0 \
    --workers=16 \
    --gpu="0" \
    --norm=2 \
    --noise_name="Gaussian" \
    --IsoMeasure=True
```

#### Baseline Comparison
```bash
python certification_baseline.py cifar10 cifar_resnet110 \
    --sigma=1.0 \
    --batch=1000 \
    --gpu="0" \
    --norm=2
```

## 📊 Results Reproduction

Our method achieves significant improvements in certified accuracy:

- **MNIST**: Up to 142.5% improvement over best baseline
- **CIFAR-10**: Up to 182.6% improvement over best baseline  
- **ImageNet**: Up to 121.1% improvement over best baseline

To reproduce paper results:

```bash
# Download pre-trained models (if available)
# Run full experimental pipeline
bash scripts/reproduce_paper_results.sh
```

## 🔬 Key Features

### Theoretical Contributions
- **Linear Transformation Theory**: Direct mapping between isotropic and anisotropic noise
- **Soundness Guarantees**: Certification-wise approach avoids memory-based certification
- **Universal Framework**: Works with any existing randomized smoothing method

### Practical Advantages
- **No Memory Overhead**: Unlike ANCER/RANCER, no parameter caching required
- **Flexible Trade-offs**: Choose NPG method based on efficiency requirements
- **Strong Performance**: Consistent improvements across datasets and threat models

## 🛠️ Advanced Configuration

### Custom Noise Patterns
```python
from noises import GaussianNoise
from noisegenerator import NoiseGenerator

# Create custom pattern-wise noise
custom_pattern = lambda x, y: 0.1 + 0.9 * (x**2 + y**2) / (32**2)
noise_gen = NoiseGenerator(pattern=custom_pattern)
```

### Multi-GPU Training
```bash
# Use multiple GPUs
python train_certification_noise.py cifar10 cifar_resnet110 ./model_saved/ \
    --gpu="0,1,2,3" \
    --batch=400  # Scale batch size accordingly
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{anonymous2024ucan,
  title={UCAN: Towards Strong Certified Defense with Asymmetric Randomization},
  author={Anonymous Authors},
  journal={Under Review},
  year={2024}
}
```

## 🔗 Related Work

- Cohen et al. - Certified Adversarial Robustness via Randomized Smoothing
- ANCER - Anisotropic Certified Robustness
- RANCER - Randomized Anisotropic Noise

## 📞 Contact

For questions about the code or paper, please:
- Open an issue on GitHub
- Contact: Anonymous submission - contact information will be provided upon acceptance

## 🙏 Acknowledgments

- Built on top of the certified robustness framework by Cohen et al.
- Neural network architectures adapted from pytorch-classification
- Thanks to the randomized smoothing community for foundational work

---

**Note**: This implementation is provided for research purposes. For production use, additional testing and validation may be required.