import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import ClassLabel, load_metric
import numpy as np

def parse_conll(filepath):
    """Parse a CoNLL file into sentences and labels."""
    sentences, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if words:
                    sentences.append(words)
                    labels.append(tags)
                    words, tags = [], []
                continue
            splits = line.split()
            if len(splits) == 2:
                words.append(splits[0])
                tags.append(splits[1])
    if words:
        sentences.append(words)
        labels.append(tags)
    return sentences, labels

if __name__ == "__main__":
    conll_path = 'data/processed/ner_sample.conll'
    sentences, labels = parse_conll(conll_path)
    print(f"Loaded {len(sentences)} sentences.")
    # Show a sample
    for i in range(2):
        print('Sentence:', sentences[i])
        print('Labels:', labels[i])

    # Step 1: Convert to Hugging Face Dataset
    data = {"tokens": sentences, "ner_tags": labels}
    dataset = Dataset.from_dict(data)
    print(f"\nHugging Face Dataset example:\n{dataset[0]}")

    # Step 2: Load XLM-Roberta tokenizer
    model_checkpoint = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    print("\nTokenizer loaded.")

    # Step 3: Tokenize and align labels
    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=False,
            return_offsets_mapping=True
        )
        labels = []
        word_ids = tokenized_inputs.word_ids()
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(example["ner_tags"][word_idx])
            else:
                # For subword tokens, use -100 so they're ignored in loss
                labels.append(-100)
            previous_word_idx = word_idx
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Map string labels to IDs
    unique_labels = sorted(set(l for label_seq in labels for l in label_seq))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    print(f"\nLabel mapping: {label2id}")

    # Convert string labels to IDs in the dataset
    def encode_labels(example):
        example["ner_tags"] = [label2id[l] for l in example["ner_tags"]]
        return example
    dataset = dataset.map(encode_labels)

    # Tokenize and align
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=False)
    print(f"\nTokenized example:\n{tokenized_dataset[0]}")

    # Step 4: Split dataset into train/validation (80/20 split)
    dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    print(f"\nTrain size: {len(train_dataset)}, Validation size: {len(val_dataset)}")

    # Step 5: Load model for token classification
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    print("\nModel loaded.")

    # Step 6: Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/ner_finetune",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./outputs/logs",
        logging_steps=10,
        report_to="none"
    )

    # Step 7: Metrics
    metric = load_metric("seqeval")
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id2label[pred] for (pred, lab) in zip(prediction, label) if lab != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Step 8: Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nTrainer is set up. To start training, uncomment the next line.")
    trainer.train()
    print("\nAfter training, you can save the model with:")
    print("trainer.save_model('./outputs/ner_finetune')")

