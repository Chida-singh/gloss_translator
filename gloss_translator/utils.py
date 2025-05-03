"""
Utility functions and classes for the English to Gloss translation model.
"""

import torch
import pandas as pd
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GlossDataset(torch.utils.data.Dataset):
    """
    Dataset class for English-Gloss translation data.
    """
    def __init__(self, dataframe, tokenizer, device, max_len=128):
        """
        Initialize dataset.
        
        Args:
            dataframe: DataFrame with 'text' and 'gloss' columns
            tokenizer: Tokenizer for processing text
            device: Device to use ('cpu' or 'cuda')
            max_len: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        gloss = self.data.iloc[index]['gloss']

        # Tokenize input
        input_encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize target with special token
        target_text = f"<GLOSS> {gloss}"
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = input_encoding["input_ids"].squeeze()
        attention_mask = input_encoding["attention_mask"].squeeze()
        labels = target_encoding["input_ids"].squeeze()

        # Replace padding token id with -100 in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def read_csv_to_texts(csv_path, text_column='text'):
    """
    Read English texts from a CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        text_column (str): Name of column containing English texts
        
    Returns:
        list: List of English texts
    """
    try:
        df = pd.read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {csv_path}")
        
        texts = df[text_column].dropna().tolist()
        return texts
    except Exception as e:
        logger.error(f"Error reading {csv_path}: {str(e)}")
        raise

def read_text_file(file_path):
    """
    Read English texts from a text file (one sentence per line).
    
    Args:
        file_path (str): Path to text file
        
    Returns:
        list: List of English texts
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        return texts
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        raise

def save_translations(english_texts, gloss_translations, output_path):
    """
    Save English-Gloss translations to file.
    
    Args:
        english_texts (list): List of English texts
        gloss_translations (list): List of Gloss translations
        output_path (str): Path to output file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Check file extension
        if output_path.endswith('.csv'):
            # Save as CSV
            df = pd.DataFrame({'english': english_texts, 'gloss': gloss_translations})
            df.to_csv(output_path, index=False)
        else:
            # Save as text
            with open(output_path, 'w', encoding='utf-8') as f:
                for eng, gloss in zip(english_texts, gloss_translations):
                    f.write(f"English: {eng}\n")
                    f.write(f"Gloss: {gloss}\n\n")
        
        logger.info(f"Translations saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving translations to {output_path}: {str(e)}")
        raise