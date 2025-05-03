"""
Command-line interface for gloss_translator.
"""

import argparse
import sys
from .model import GlossTranslator


def translate_text(text, model_path=None):
    """
    Translate a single text string.
    
    Args:
        text (str): The English text to translate
        model_path (str, optional): Path to the model
        
    Returns:
        str: The translated gloss
    """
    translator = GlossTranslator(model_path=model_path)
    return translator.translate(text)


def translate_file(input_file, output_file, model_path=None):
    """
    Translate all lines in a file.
    
    Args:
        input_file (str): Path to input file
        output_file (str): Path to output file
        model_path (str, optional): Path to the model
    """
    translator = GlossTranslator(model_path=model_path)
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()
    
    translations = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            translation = translator.translate(line)
            translations.append(translation)
        else:
            translations.append('')
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for translation in translations:
            f_out.write(f"{translation}\n")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Translate English text to gloss notation."
    )
    
    # Input group - either text or file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "text",
        nargs="?",
        help="English text to translate"
    )
    input_group.add_argument(
        "--input", "-i",
        help="Input file containing English text (one sentence per line)"
    )
    
    # Output file (required if input file is provided)
    parser.add_argument(
        "--output", "-o",
        help="Output file for gloss translations"
    )
    
    # Model path
    parser.add_argument(
        "--model", "-m",
        help="Path to the model directory"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input and not args.output:
        parser.error("--output is required when using --input")
    
    try:
        if args.text:
            # Translate single text
            translation = translate_text(args.text, model_path=args.model)
            print(translation)
        else:
            # Translate file
            translate_file(args.input, args.output, model_path=args.model)
            print(f"Translation completed. Output saved to {args.output}")
    
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()