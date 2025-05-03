"""
Training script for the English to Gloss translation model.
This is only needed if you want to train a new model.
"""

import pandas as pd
import torch
import os
from transformers import MarianMTModel, MarianTokenizer
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import logging
from .utils import GlossDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(
    train_csv, 
    output_dir="./gloss-translation-model",
    model_name="Helsinki-NLP/opus-mt-en-ROMANCE",
    epochs=3,
    batch_size=None,
    test_size=0.1,
    max_seq_length=128
):
    """
    Train an English to Gloss translation model.
    
    Args:
        train_csv (str): Path to training CSV file with 'text' and 'gloss' columns.
        output_dir (str): Directory to save the trained model.
        model_name (str): Base model name to use for fine-tuning.
        epochs (int): Number of training epochs.
        batch_size (int, optional): Batch size. If None, automatically chosen based on device.
        test_size (float): Proportion of data to use for validation.
        max_seq_length (int): Maximum sequence length to use.
        
    Returns:
        tuple: (model, tokenizer, results) - Trained model, tokenizer, and evaluation results.
    """
    # Check if GPU is available and set device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set batch size based on device if not specified
    if batch_size is None:
        batch_size = 16 if torch.cuda.is_available() else 4
    
    # Load dataset
    logger.info(f"Loading data from {train_csv}")
    df = pd.read_csv(train_csv)
    logger.info(f"Loaded dataset with {len(df)} rows")
    
    # Clean data
    df['text'] = df['text'].str.strip()
    df['gloss'] = df['gloss'].str.strip()
    df = df.dropna(subset=['text', 'gloss'])
    logger.info(f"After cleaning: {len(df)} rows")
    
    # Optional: Filter very long sequences
    df = df[df['text'].str.split().str.len() < max_seq_length]
    logger.info(f"After filtering long sequences: {len(df)} rows")
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    logger.info(f"Training set: {len(train_df)} rows, Validation set: {len(val_df)} rows")
    
    # Initialize model and tokenizer
    logger.info(f"Initializing model: {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Add special tokens if needed
    special_tokens_dict = {'additional_special_tokens': ['<GLOSS>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    # Move model to the appropriate device
    model = model.to(device)
    
    # Create datasets
    train_dataset = GlossDataset(train_df, tokenizer, device, max_len=max_seq_length)
    val_dataset = GlossDataset(val_df, tokenizer, device, max_len=max_seq_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,   # Helps with memory efficiency
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA is available
        no_cuda=not torch.cuda.is_available() # Disable CUDA if not available
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")
    
    return model, tokenizer, eval_results

def main():
    """CLI entry point for training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train English to Gloss translation model")
    parser.add_argument("--train_csv", required=True, help="Path to training CSV file")
    parser.add_argument("--output_dir", default="./gloss-translation-model", help="Output directory")
    parser.add_argument("--model_name", default="Helsinki-NLP/opus-mt-en-ROMANCE", help="Base model name")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size (auto if not specified)")
    parser.add_argument("--test_size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    train_model(
        args.train_csv,
        args.output_dir,
        args.model_name,
        args.epochs,
        args.batch_size,
        args.test_size,
        args.max_seq_length
    )

if __name__ == "__main__":
    main()