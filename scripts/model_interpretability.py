import shap
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import numpy as np

# Optional: For LIME
try:
    from lime.lime_text import LimeTextExplainer
    lime_available = True
except ImportError:
    lime_available = False
    print("LIME is not installed. To use LIME, run: pip install lime")

# 1. Load model and tokenizer (change path to your best model if needed)
model_dir = "outputs/ner_xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# 2. Create NER pipeline
ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 3. SHAP interpretability
print("\n=== SHAP Explanation ===")
def predict_fn(texts):
    # Returns the number of detected entities for each text (simple demo)
    return np.array([[len(ner_pipe(text))] for text in texts])

explainer = shap.Explainer(predict_fn, masker=shap.maskers.Text(tokenizer))
sample_text = "ሻይ 300 ብር አዲስ አበባ"
shap_values = explainer([sample_text])
shap.plots.text(shap_values)

# 4. LIME interpretability (if available)
if lime_available:
    print("\n=== LIME Explanation ===")
    class_names = ["O", "B-Product", "I-Product", "B-PRICE", "I-PRICE", "B-LOC", "I-LOC"]
    explainer = LimeTextExplainer(class_names=class_names)
    def lime_predict(texts):
        # For LIME, return a probability for each class for each token
        # Here, we use a dummy output for demonstration
        # You can adapt this to use model outputs if needed
        return np.ones((len(texts), len(class_names))) / len(class_names)
    exp = explainer.explain_instance(sample_text, lime_predict, num_features=6)
    exp.show_in_notebook(text=True)
else:
    print("LIME not available. Skipping LIME explanation.")

print("\nInterpretability script complete. Review SHAP/LIME plots for insights.") 