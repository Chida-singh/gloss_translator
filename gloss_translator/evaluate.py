"""
Evaluation functions for the English to Gloss translation model.
"""

import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from jiwer import wer, cer
import logging
from .model import load_model, translate_to_gloss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
def ensure_nltk_resources():
    """Ensure required NLTK resources are available"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

# Call this function to make sure resources are available
ensure_nltk_resources()

def calculate_metrics(reference, hypothesis):
    """
    Calculate various evaluation metrics for a translation.
    
    Args:
        reference (str): Reference (gold standard) translation.
        hypothesis (str): Model-generated translation.
        
    Returns:
        dict: Dictionary containing various metrics
    """
    # Tokenize reference and hypothesis
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    
    # BLEU calculation with smoothing
    smooth = SmoothingFunction().method1
    
    # Calculate BLEU with n-grams from 1 to 4
    bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu_2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu_3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smooth)
    bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    # Calculate WER and CER
    word_error_rate = wer(reference, hypothesis)
    char_error_rate = cer(reference, hypothesis)
    
    # Calculate exact match
    exact_match = 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0
    
    return {
        "bleu-1": bleu_1,
        "bleu-2": bleu_2,
        "bleu-3": bleu_3,
        "bleu-4": bleu_4,
        "wer": word_error_rate,
        "cer": char_error_rate,
        "exact_match": exact_match
    }

def evaluate_model(test_csv, model=None, tokenizer=None, model_path=None, num_samples=None, verbose=True):
    """
    Evaluate the model on a test dataset.
    
    Args:
        test_csv (str): Path to CSV file with test data (must have 'text' and 'gloss' columns).
        model (optional): Pre-loaded model. If None, model will be loaded.
        tokenizer (optional): Pre-loaded tokenizer. If None, tokenizer will be loaded.
        model_path (str, optional): Path to model if model and tokenizer are None.
        num_samples (int, optional): Number of samples to evaluate. If None, all samples are used.
        verbose (bool): Whether to print progress and examples.
        
    Returns:
        dict: Evaluation metrics
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_path)
    
    # Load test data
    test_df = pd.read_csv(test_csv)
    if 'text' not in test_df.columns or 'gloss' not in test_df.columns:
        raise ValueError("Test CSV must contain 'text' and 'gloss' columns")
    
    # Clean data
    test_df = test_df.dropna(subset=['text', 'gloss'])
    test_df['text'] = test_df['text'].str.strip()
    test_df['gloss'] = test_df['gloss'].str.strip()
    
    # Limit samples if specified
    if num_samples is not None:
        test_df = test_df.sample(min(num_samples, len(test_df)), random_state=42)
    
    # Metrics storage
    results = {
        "bleu-1": [],
        "bleu-2": [],
        "bleu-3": [],
        "bleu-4": [],
        "wer": [],
        "cer": [],
        "exact_match": []
    }
    
    # Track examples for display
    examples = []
    
    if verbose:
        logger.info(f"Evaluating model on {len(test_df)} test examples...")
    
    # Process each example
    for idx, row in test_df.iterrows():
        english_text = row['text']
        reference_gloss = row['gloss']
        
        # Generate translation
        predicted_gloss = translate_to_gloss(english_text, model, tokenizer)
        
        # Calculate metrics
        metrics = calculate_metrics(reference_gloss, predicted_gloss)
        
        # Store results
        for key in results:
            results[key].append(metrics[key])
        
        # Store example
        example = {
            "english": english_text,
            "reference": reference_gloss,
            "prediction": predicted_gloss,
            **metrics
        }
        examples.append(example)
        
        # Print sample translations if verbose
        if verbose and (idx % 20 == 0 or idx == len(test_df) - 1):
            logger.info(f"\nExample {idx}:")
            logger.info(f"English: {english_text}")
            logger.info(f"Reference: {reference_gloss}")
            logger.info(f"Prediction: {predicted_gloss}")
            logger.info(f"BLEU-4: {metrics['bleu-4']:.4f}, WER: {metrics['wer']:.4f}")
    
    # Calculate average metrics
    avg_results = {key: np.mean(values) for key, values in results.items()}
    
    # Add examples to results
    avg_results["examples"] = examples
    
    # Print results if verbose
    if verbose:
        logger.info("\n===== MODEL EVALUATION RESULTS =====")
        logger.info(f"BLEU-1: {avg_results['bleu-1']:.4f}")
        logger.info(f"BLEU-2: {avg_results['bleu-2']:.4f}")
        logger.info(f"BLEU-3: {avg_results['bleu-3']:.4f}")
        logger.info(f"BLEU-4: {avg_results['bleu-4']:.4f}")
        logger.info(f"Word Error Rate: {avg_results['wer']:.4f} (lower is better)")
        logger.info(f"Character Error Rate: {avg_results['cer']:.4f} (lower is better)")
        logger.info(f"Exact Match Accuracy: {avg_results['exact_match']:.4f}")
        logger.info("====================================")
    
    return avg_results

def save_evaluation_results(results, output_file):
    """
    Save evaluation results to a CSV file.
    
    Args:
        results (dict): Results from evaluate_model().
        output_file (str): Path to output CSV file.
    """
    # Extract examples
    examples = results.pop("examples")
    
    # Create DataFrame from examples
    df = pd.DataFrame(examples)
    
    # Add overall metrics as extra columns
    for key, value in results.items():
        df[f"avg_{key}"] = value
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Evaluation results saved to {output_file}")
    
    # Restore examples to results
    results["examples"] = examples
    
    return df

def main():
    """CLI entry point for evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate English to Gloss translation model")
    parser.add_argument("--test_csv", required=True, help="Path to test CSV file")
    parser.add_argument("--model_path", default=None, help="Path to model directory")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    parser.add_argument("--output_file", help="Path to save detailed results (optional)")
    
    args = parser.parse_args()
    
    results = evaluate_model(
        args.test_csv,
        model_path=args.model_path,
        num_samples=args.num_samples,
        verbose=True
    )
    
    if args.output_file:
        save_evaluation_results(results, args.output_file)

if __name__ == "__main__":
    main()