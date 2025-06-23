#!/usr/bin/env python3
"""
Complete the labeling of 30-50 messages for Task 2.
This script provides a template and guidance for manual labeling.
"""

import pandas as pd
import random

def create_complete_labeling_template():
    """Create a complete labeling template with 50 messages."""
    
    # Read the processed data
    df = pd.read_csv('data/processed/all_processed.csv')
    df_clean = df[df['normalized_text'].notna() & (df['normalized_text'].str.len() > 20)]
    
    # Sample 50 messages for labeling
    sampled = df_clean.sample(n=50, random_state=42)
    
    # Create the labeling template
    with open('data/processed/complete_labeling_template.txt', 'w', encoding='utf-8') as f:
        f.write('# Complete Labeling Template for Task 2\n')
        f.write('# Instructions:\n')
        f.write('# 1. For each message, label each token with:\n')
        f.write('#    - B-Product: Beginning of product name\n')
        f.write('#    - I-Product: Inside product name\n')
        f.write('#    - B-LOC: Beginning of location\n')
        f.write('#    - I-LOC: Inside location\n')
        f.write('#    - B-PRICE: Beginning of price\n')
        f.write('#    - I-PRICE: Inside price\n')
        f.write('#    - O: Outside any entity\n')
        f.write('# 2. Replace "O" with the correct label\n')
        f.write('# 3. Add blank line between messages\n')
        f.write('# 4. Save as ner_sample.conll\n\n')
        
        for idx, row in sampled.iterrows():
            text = row['normalized_text']
            tokens = text.split()
            
            f.write(f'# Message {idx+1}: {text}\n')
            f.write('# Label each token (replace O with correct label):\n')
            
            for token in tokens:
                f.write(f'{token}\tO\n')
            
            f.write('\n')
    
    print("Created complete labeling template: data/processed/complete_labeling_template.txt")
    print("This file contains 50 messages ready for manual labeling")
    print("\nNext steps:")
    print("1. Open the template file")
    print("2. Label each token manually")
    print("3. Save as data/processed/ner_sample.conll")
    print("4. Remove comment lines (#) from final file")

def create_final_conll_template():
    """Create the final CoNLL format template."""
    
    # This is what the final file should look like (without comments)
    final_template = """# Final CoNLL format (remove all # lines)
# Example of how the final file should look:

👑Shower	B-Product
Cap	I-Product
🛡	O
የፀጉር	O
መሸፈኛ	O
የሻወር	O
ኮፍያ	O
✔️	O
ዘወትር	O
የሚጠቀሙበት	O
🛡ውሃ	O
አያስገባም	O
🛡	O
Free	O
size	O
💵💵200	B-PRICE
ብር💵💵	I-PRICE
🛵**	O
ከ100	O
እስከ	O
200	B-PRICE
ብር	I-PRICE
ብቻ	O
ከፍለው	O
አዲስ	B-LOC
አበባ	I-LOC
ውስጥ	O
ካሉበት	O
እንልካለን**	O

"""
    
    with open('data/processed/final_conll_template.txt', 'w', encoding='utf-8') as f:
        f.write(final_template)
    
    print("Created final CoNLL template: data/processed/final_conll_template.txt")

if __name__ == "__main__":
    create_complete_labeling_template()
    create_final_conll_template() 