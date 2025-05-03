"""
Model handling functions for the English to Gloss translation.
"""

import os
import torch
from transformers import MarianMTModel, MarianTokenizer
import pkg_resources

# Default model path - points to the bundled model if available
DEFAULT_MODEL_PATH = pkg_resources.resource_filename("gloss_translator", "model")

def load_model(model_path=None):
    """
    Load the English to Gloss translation model.
    
    Args:
        model_path (str, optional): Path to the saved model directory.
            If None, uses the default bundled model.
    
    Returns:
        tuple: (model, tokenizer) - The loaded model and tokenizer.
    """
    # If no model path provided, use default
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
        
        # Check if the default model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Default model not found at {model_path}. "
                "Please provide a valid model path or install the model."
            )
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and model
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_path)
        model = MarianMTModel.from_pretrained(model_path)
        model = model.to(device)
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

def translate_to_gloss(text, model=None, tokenizer=None, model_path=None):
    """
    Translate English text to gloss notation.
    
    Args:
        text (str): English text to translate.
        model (optional): Pre-loaded model. If None, model will be loaded.
        tokenizer (optional): Pre-loaded tokenizer. If None, tokenizer will be loaded.
        model_path (str, optional): Path to model if model and tokenizer are None.
    
    Returns:
        str: The translated gloss text.
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)
    
    # Get device that model is on
    device = next(model.parameters()).device
    
    # Tokenize input
    input_ids = tokenizer.encode(
        text, 
        return_tensors="pt", 
        max_length=128, 
        padding="max_length",
        truncation=True
    )
    
    # Move input to the same device as model
    input_ids = input_ids.to(device)
    
    # Generate output
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(
            input_ids,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode output
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the special token if present
    if translation.startswith("<GLOSS>"):
        translation = translation[len("<GLOSS>"):].strip()
        
    return translation

def translate_batch(texts, model=None, tokenizer=None, model_path=None, batch_size=8):
    """
    Translate a batch of English texts to gloss notation.
    
    Args:
        texts (list): List of English texts to translate.
        model (optional): Pre-loaded model. If None, model will be loaded.
        tokenizer (optional): Pre-loaded tokenizer. If None, tokenizer will be loaded.
        model_path (str, optional): Path to model if model and tokenizer are None.
        batch_size (int): Size of batches for processing.
    
    Returns:
        list: The translated gloss texts.
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)
    
    results = []
    
    # Process in batches to save memory
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Process each text in the batch
        batch_results = [translate_to_gloss(text, model, tokenizer) for text in batch]
        results.extend(batch_results)
    
    return results