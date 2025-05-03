"""
Setup script for gloss_translator package.
"""

import os
from setuptools import setup, find_packages

# Read the contents of README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read version from package
with open("gloss_translator/__init__.py", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"\'')
            break

setup(
    name="gloss_translator",
    version=version,
    description="A package for translating English text to gloss notation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/gloss_translator",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.18.0",
        "nltk>=3.7",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "jiwer>=2.3.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "translate-gloss=gloss_translator.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gloss_translator": ["data/*.json"],
    },
)