#!/usr/bin/env python3
"""
Preprocess Amharic Telegram messages for NER.
- Tokenize and normalize Amharic text
- Clean and structure data
- Save processed data for labeling/modeling
"""
import os
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Optional: Use Amharic-specific NLP tools if available
# from amharic_tokenizer import AmharicTokenizer
# For now, use simple whitespace tokenization

def normalize_amharic(text):
    """Basic normalization for Amharic text."""
    if not isinstance(text, str):
        return ""
    # Remove extra spaces, normalize punctuation, etc.
    text = re.sub(r'[፡።፣፤፥፦፧፨]', ' ', text)  # Remove Amharic punctuation
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_amharic(text):
    """Tokenize Amharic text (simple whitespace for now)."""
    return text.split()

def preprocess_row(row):
    text = row.get('message_text', '')
    norm_text = normalize_amharic(text)
    tokens = tokenize_amharic(norm_text)
    row['normalized_text'] = norm_text
    row['tokens'] = ' '.join(tokens)
    return row

def preprocess_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    tqdm.pandas(desc="Preprocessing Amharic messages")
    df = df.progress_apply(preprocess_row, axis=1)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Processed data saved to {output_csv}")

def main():
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all channel CSVs
    for csv_file in raw_dir.glob('*_messages.csv'):
        output_csv = processed_dir / csv_file.name.replace('_messages.csv', '_processed.csv')
        preprocess_csv(csv_file, output_csv)

if __name__ == "__main__":
    main()




