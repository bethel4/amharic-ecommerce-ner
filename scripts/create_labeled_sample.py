#!/usr/bin/env python3
"""
Create a sample of labeled messages in CoNLL format.
This demonstrates how to label Amharic e-commerce messages.
"""

def create_labeled_sample():
    """Create a sample of labeled messages in CoNLL format."""
    
    # Sample labeled messages (first few from the dataset)
    labeled_messages = [
        # Message 1: Shower Cap
        {
            'tokens': ['👑Shower', 'Cap', '🛡', 'የፀጉር', 'መሸፈኛ', 'የሻወር', 'ኮፍያ', '✔️', 'ዘወትር', 'የሚጠቀሙበት', '🛡ውሃ', 'አያስገባም', '🛡', 'Free', 'size', '💵💵200', 'ብር💵💵', '🛵**', 'ከ100', 'እስከ', '200', 'ብር', 'ብቻ', 'ከፍለው', 'አዲስ', 'አበባ', 'ውስጥ', 'ካሉበት', 'እንልካለን**'],
            'labels': ['B-Product', 'I-Product', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PRICE', 'I-PRICE', 'O', 'O', 'O', 'B-PRICE', 'I-PRICE', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O']
        },
        
        # Message 2: Toilet Seat Cover
        {
            'tokens': ['🚽', 'Toilet', 'Seat', 'Plastic', 'Cover', '✔️', 'የWC', 'መፀዳጃ', 'መቀመጫ', 'መሸፈኛ', 'ፕላስቲክ', '✔️', 'የህዝብ', 'መፀዳጃ', 'ቤት', 'ውስጥ', 'ሳይሸክኩ', 'ደስ', 'ብሎዎት', 'ለመቀመጥ', 'ይረዳዎታል', '🛡', 'ደግመው', 'ደጋግመው', 'እያጠቡ', 'የሚጠቀሙበት', '💵💵', '20pc.....', '700', 'ብር💵💵'],
            'labels': ['O', 'B-Product', 'I-Product', 'I-Product', 'I-Product', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PRICE', 'I-PRICE', 'I-PRICE']
        },
        
        # Message 3: Kids Tent
        {
            'tokens': ['⛺', 'Kids', 'indoor', 'playing', 'Tent', '✔️', 'የልጆች', 'የቤት', 'ውስጥ', 'መጫወቻ', 'ድንኳን', '✔️በቀላሉ', 'የሚዘረጋ', 'እና', 'የሚታጠፍ', '✔️', 'አንድ', 'በር', 'አለው', '🛡', 'በሁለት', 'በኩል', 'ባለ', 'ወንፊት', '⛺', 'ዕድሜያቸው', '3', 'አመትና', 'ከዚያ', 'በላይ', 'ለሆኑ', 'ልጆች', 'ተመራጭ', '💵💵', '2100', 'ብር💵💵'],
            'labels': ['O', 'B-Product', 'I-Product', 'I-Product', 'I-Product', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PRICE', 'I-PRICE', 'I-PRICE']
        }
    ]
    
    # Write to CoNLL file
    with open('data/processed/ner_sample.conll', 'w', encoding='utf-8') as f:
        for msg_idx, message in enumerate(labeled_messages, 1):
            f.write(f'# Message {msg_idx}\n')
            for token, label in zip(message['tokens'], message['labels']):
                f.write(f'{token}\t{label}\n')
            f.write('\n')  # Blank line between messages
    
    print("Created sample labeled file: data/processed/ner_sample.conll")
    print(f"Labeled {len(labeled_messages)} messages as examples")

if __name__ == "__main__":
    create_labeled_sample() 