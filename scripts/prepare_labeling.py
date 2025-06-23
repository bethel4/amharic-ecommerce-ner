#!/usr/bin/env python3
"""
Prepare messages for manual labeling in CoNLL format.
This script samples messages from processed data and formats them for easy labeling.
"""

import pandas as pd
import random
from pathlib import Path

def sample_messages_for_labeling(csv_file, num_samples=50, output_file="data/processed/messages_for_labeling.txt"):
    """
    Sample messages and prepare them for manual labeling.
    
    Args:
        csv_file: Path to processed CSV file
        num_samples: Number of messages to sample
        output_file: Output file path
    """
    
    # Read the processed data
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    # Filter out messages with no text or very short text
    df = df[df['normalized_text'].notna() & (df['normalized_text'].str.len() > 10)]
    
    print(f"Found {len(df)} messages with text content")
    
    # Sample messages
    if len(df) < num_samples:
        print(f"Warning: Only {len(df)} messages available, using all of them")
        sampled_df = df
    else:
        sampled_df = df.sample(n=num_samples, random_state=42)
    
    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Write messages for labeling
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Messages for Manual Labeling in CoNLL Format\n")
        f.write("# Instructions:\n")
        f.write("# 1. For each message, label each token with:\n")
        f.write("#    - B-Product: Beginning of product name\n")
        f.write("#    - I-Product: Inside product name\n")
        f.write("#    - B-LOC: Beginning of location\n")
        f.write("#    - I-LOC: Inside location\n")
        f.write("#    - B-PRICE: Beginning of price\n")
        f.write("#    - I-PRICE: Inside price\n")
        f.write("#    - O: Outside any entity\n")
        f.write("# 2. Add blank line between messages\n")
        f.write("# 3. Save as .conll file\n\n")
        
        for idx, row in sampled_df.iterrows():
            message_id = row['message_id']
            channel = row['channel_name']
            text = row['normalized_text']
            tokens = row['tokens'].split() if pd.notna(row['tokens']) else text.split()
            
            f.write(f"# Message {idx+1}: ID={message_id}, Channel={channel}\n")
            f.write(f"# Original: {text}\n")
            f.write(f"# Tokens: {' | '.join(tokens)}\n")
            f.write("# Label each token below:\n")
            
            for token in tokens:
                f.write(f"{token}\tO\n")  # Default to O, you'll change this manually
            
            f.write("\n")  # Blank line between messages
    
    print(f"Created labeling file: {output_file}")
    print(f"Sampled {len(sampled_df)} messages")
    
    # Also create a summary
    summary_file = output_file.replace('.txt', '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Message Summary for Labeling\n")
        f.write("=" * 50 + "\n\n")
        
        for idx, row in sampled_df.iterrows():
            f.write(f"Message {idx+1}:\n")
            f.write(f"  ID: {row['message_id']}\n")
            f.write(f"  Channel: {row['channel_name']}\n")
            f.write(f"  Text: {row['normalized_text']}\n")
            f.write(f"  Tokens: {' | '.join(row['tokens'].split())}\n")
            f.write("\n")
    
    print(f"Created summary file: {summary_file}")
    
    return output_file, summary_file

def main():
    """Main function to prepare messages for labeling."""
    csv_file = "data/processed/all_processed.csv"
    
    if not Path(csv_file).exists():
        print(f"Error: {csv_file} not found!")
        return
    
    # Sample 50 messages for labeling
    output_file, summary_file = sample_messages_for_labeling(
        csv_file, 
        num_samples=50,
        output_file="data/processed/messages_for_labeling.txt"
    )
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("1. Open the file: data/processed/messages_for_labeling.txt")
    print("2. For each message, manually label each token")
    print("3. Save the labeled file as: data/processed/ner_sample.conll")
    print("4. Remove the comment lines (#) from the final .conll file")
    print("="*60)

if __name__ == "__main__":
    main() 