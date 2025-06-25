import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from datetime import datetime

# 1. Load data
csv_path = "data/processed/all_processed.csv"
df = pd.read_csv(csv_path)
# Assume columns: channel_name, message_text, message_date, views, etc.

# 2. Load NER pipeline
model_dir = "outputs/ner_xlm-roberta-base"  # Change if needed
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)
ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_price(text):
    entities = ner_pipe(str(text))
    prices = [e['word'] for e in entities if 'PRICE' in e['entity_group']]
    for p in prices:
        try:
            return float(''.join(filter(str.isdigit, p)))
        except:
            continue
    return None

scorecard = []
for vendor, group in df.groupby("channel_name"):
    # Posting frequency (posts per week)
    group = group.copy()
    group['message_date'] = pd.to_datetime(group['message_date'])
    weeks = (group['message_date'].max() - group['message_date'].min()).days / 7 + 1
    posts_per_week = len(group) / weeks if weeks > 0 else len(group)
    # Average views
    avg_views = group['views'].mean()
    # Top post
    top_post = group.loc[group['views'].idxmax()]
    top_post_text = top_post['message_text']
    top_post_price = extract_price(top_post_text)
    # Average price
    prices = group['message_text'].apply(extract_price).dropna()
    avg_price = prices.mean() if not prices.empty else None
    # Lending score (customizable)
    lending_score = (avg_views * 0.5) + (posts_per_week * 0.5)
    scorecard.append({
        "Vendor": vendor,
        "Avg. Views/Post": round(avg_views, 2),
        "Posts/Week": round(posts_per_week, 2),
        "Avg. Price (ETB)": round(avg_price, 2) if avg_price else "N/A",
        "Lending Score": round(lending_score, 2),
        "Top Post": top_post_text[:50] + "...",
        "Top Post Price": top_post_price
    })

# 4. Create and save scorecard
scorecard_df = pd.DataFrame(scorecard)
scorecard_df = scorecard_df.sort_values("Lending Score", ascending=False)
scorecard_df.to_csv("outputs/vendor_scorecard.csv", index=False)
print(scorecard_df[["Vendor", "Avg. Views/Post", "Posts/Week", "Avg. Price (ETB)", "Lending Score"]]) 