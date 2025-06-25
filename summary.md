# EthioMart Amharic E-commerce NER Project: Data Preparation & Labeling Summary

## Project Overview
EthioMart aims to centralize e-commerce data from multiple Ethiopian Telegram channels, making it easier to search, analyze, and support business decisions such as vendor credit scoring. Our project's first phase focuses on collecting, cleaning, and labeling Amharic e-commerce messages for Named Entity Recognition (NER).

---

## 1. Data Collection & Ingestion

**Channels Targeted:**  
We identified and connected to 9 major Ethiopian e-commerce Telegram channels, including:  
`@MerttEka`, `@forfreemarket`, `@classybrands`, `@marakibrand`, `@aradabrand2`, `@marakisat2`, `@belaclassic`, `@AwasMart`, `@qnashcom`.

**Automated Scraper:**  
Using the Telethon library, we built a custom Python scraper to:
- Join and monitor the selected channels.
- Download recent posts (text and images).
- Store all messages, media, and metadata (sender, timestamp, etc.) in a structured format.

**Volume Collected:**  
Over 4,400 messages were collected, including both text and media, and stored in CSV and database formats for further processing.

---

## 2. Data Preprocessing

**Text Cleaning:**  
- Removed duplicates, junk, and irrelevant content.
- Normalized Amharic text (handled punctuation, whitespace, and special characters).
- Tokenized messages for easier downstream processing.

**Metadata Structuring:**  
- Separated message content from metadata (sender, channel, timestamp, etc.).
- Ensured all data is stored in a unified, analysis-ready format.

**Output:**  
- Cleaned and tokenized messages are saved in `data/processed/` as CSV files.
- Media files are organized in dedicated folders.

---

## 3. Data Labeling for NER

**Labeling Objective:**  
To prepare high-quality training data for NER, we manually labeled a subset of messages to identify:
- **Product** (B-Product, I-Product)
- **Price** (B-PRICE, I-PRICE)
- **Location** (B-LOC, I-LOC)
- **Other** (O: tokens outside any entity)

**Labeling Process:**  
- Sampled 50 representative messages from the cleaned dataset.
- Used the CoNLL format: each token is labeled on its own line, with blank lines separating messages.
- Provided clear instructions and examples to ensure consistent labeling.

**Example (CoNLL Format):**
```
የሴቶች    B-Product
ጫማ      I-Product
799      B-PRICE
ብር      I-PRICE
በአዲስ    B-LOC
አበባ    I-LOC
```

**Tools:**  
- Labeling was done manually using prepared templates, with future plans to use tools like Label Studio for larger-scale annotation.

---

## 4. Current Status & Next Steps

**Data collection and cleaning:** Complete.
**Initial manual labeling:** Complete for 50 messages; more can be labeled as needed.
**Next:** Use labeled data to train and evaluate NER models (e.g., Amharic BERT, XLM-RoBERTa).

---

## Conclusion
The project's data preparation and labeling phase is complete and ready for model training. This foundation will enable EthioMart to build a robust, searchable e-commerce intelligence platform for the Ethiopian market. 