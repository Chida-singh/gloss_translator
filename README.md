# Gloss Translator

A simple Python package that translates English text to gloss notation.

## What is this?

This package lets you translate English text to gloss notation using a trained AI model. Once installed, you can translate text with just a few lines of code.

## Installation

```bash
# Install from pip
pip install gloss-translator

# Or install from source
git clone https://github.com/Chida-singh/gloss_translator.git
cd gloss_translator
pip install -e .
```

## Simple Usage

### In Python

```python
from gloss_translator import GlossTranslator

# Create translator
translator = GlossTranslator()

# Translate text
english = "The cat sat on the mat."
gloss = translator.translate(english)

print(gloss)  # Will show gloss translation
```

### Command Line

```bash
# Translate a sentence
translate-gloss "The cat sat on the mat."

# Translate a file
translate-gloss --input sentences.txt --output translations.txt
```

## Using Your Own Model

If you trained a custom model, you can use it like this:

```python
translator = GlossTranslator(model_path="/path/to/your/model")
```

## Requirements

- Python 3.8 or higher
- PyTorch
- Transformers library

## Created By

[Chida Singh](https://github.com/Chida-singh)

## License

MIT License