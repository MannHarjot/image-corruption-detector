"""Package setup for image-corruption-detector."""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="image-corruption-detector",
    version="1.0.0",
    description="CNN-based image corruption detection using PyTorch and ResNet-18 transfer learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Harjot Singh Mann",
    author_email="mannh12@mcmaster.ca",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.26.0",
        "opencv-python>=4.9.0",
        "Pillow>=10.2.0",
        "scikit-learn>=1.4.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pandas>=2.2.0",
        "PyYAML>=6.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.1.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "icd-train=scripts.train:main",
            "icd-evaluate=scripts.evaluate:main",
            "icd-predict=scripts.predict:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
