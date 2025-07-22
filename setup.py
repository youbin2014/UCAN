#!/usr/bin/env python3
"""
Setup script for UCAN: Towards Strong Certified Defense with Asymmetric Randomization
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements():
    requirements = []
    with open("requirements.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                requirements.append(line)
    return requirements

setup(
    name="ucan",
    version="1.0.0",
    description="UCAN: Towards Strong Certified Defense with Asymmetric Randomization",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Anonymous Authors",
    author_email="anonymous@review.submission",
    url="[ANONYMOUS_REPO_URL]",
    
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    
    keywords="adversarial robustness, certified defense, randomized smoothing, deep learning",
    
    entry_points={
        "console_scripts": [
            "ucan-train=train_certification_noise:main",
            "ucan-certify=certification_certification_noise:main",
        ],
    },
    
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0", 
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",
            "torchvision>=0.10.0+cu111",
        ],
    },
    
    include_package_data=True,
    zip_safe=False,
    
    project_urls={
        "Bug Reports": "[ANONYMOUS_REPO_URL]/issues",
        "Source": "[ANONYMOUS_REPO_URL]",
        "Documentation": "[ANONYMOUS_REPO_URL]#readme",
    },
)