# amharic-ecommerce-ner
# Amharic E-commerce NER

This project extracts product, price, location, and other key business entities from Amharic Telegram e-commerce posts. Built for EthioMart's goal to centralize vendor intelligence and improve credit scoring.

## Objectives
- Ingest Telegram e-commerce messages (text/images)
- Preprocess and structure Amharic data
- Label and fine-tune transformer models (mBERT, XLM-R) for NER
- Evaluate and explain model predictions (SHAP, LIME)

## Project Structure

```
amharic-ecommerce-ner/
│
├── data/
│   ├── raw/                # Unprocessed scraped data
│   ├── processed/          # Cleaned, structured data
│   └── external/           # Any downloaded datasets (e.g., Amharic NER dataset)
│
├── notebooks/              # Jupyter notebooks for EDA, training, etc.
│   ├── 01_scraping.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_labeling.ipynb
│   └── 04_model_training.ipynb
│
├── scripts/                # Script versions of notebooks or automation
│   ├── scraper.py
│   ├── preprocess.py
│   └── fine_tune.py
│
├── src/                    # Core logic as importable modules
│   ├── data_ingestion/
│   ├── preprocessing/
│   ├── labeling/
│   └── modeling/
│
├── models/                 # Saved fine-tuned model files
│
├── outputs/                # Visualizations, results, metrics, SHAP/LIME outputs
│
├── tests/                  # Unit tests
│
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
```
