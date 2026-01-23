from setuptools import setup, find_packages
import os

# Read README for long description
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fca-torch",
    version="0.1.0",
    description="Functional Component Analysis: Find functionally sufficient subspaces in neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Satchel Grant",
    author_email="grantsrb@stanford.edu",
    url="https://github.com/grantsrb/fca",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "scikit-learn>=0.24.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "vision": [
            "torchvision>=0.10.0",
        ],
        "transformers": [
            "transformers>=4.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "all": [
            "torchvision>=0.10.0",
            "transformers>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords=[
        "deep learning",
        "neural networks",
        "interpretability",
        "pytorch",
        "functional components",
        "dimensionality reduction",
        "circuit analysis",
        "representation learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/grantsrb/fca/issues",
        "Source": "https://github.com/grantsrb/fca",
    },
)
