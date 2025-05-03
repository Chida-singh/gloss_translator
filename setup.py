from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gloss_translator",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="English to Sign Language Gloss Translation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gloss_translator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.11.0",
        "pandas>=1.3.0",
        "nltk>=3.6.0",
        "jiwer>=2.3.0",
        "scikit-learn>=0.24.0",
    ],
    entry_points={
        'console_scripts': [
            'gloss-translate=gloss_translator.cli:translate_cli',
            'gloss-evaluate=gloss_translator.cli:evaluate_cli',
            'gloss-train=gloss_translator.cli:train_cli',
        ],
    },
    include_package_data=True,
    package_data={
        # If you want to include the default model, uncomment this
        # "gloss_translator": ["model/*"],
    },
)