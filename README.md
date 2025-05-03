# gloss_translato# Gloss Translator

A simple English-to-gloss translation utility using MarianMT from HuggingFace Transformers.

## Structure
- `train.py`: Load training data and prepare model.
- `evaluate.py`: Run model on test data.
- `utils.py`: Utility functions for loading CSV data.
- `model.py`: MarianMT-based translation model.

## Usage
```bash
python gloss_translator/train.py
python gloss_translator/evaluate.py
