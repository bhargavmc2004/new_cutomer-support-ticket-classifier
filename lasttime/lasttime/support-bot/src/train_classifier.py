"""
Train classifier model.
"""

import pandas as pd
import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_data():
    """Load training and test data."""
    project_root = Path(__file__).parent.parent
    
    # Load CSV files
    train_df = pd.read_csv(project_root / 'data' / 'processed' / 'train.csv')
    test_df = pd.read_csv(project_root / 'data' / 'processed' / 'test.csv')
    
    # Load label mapping
    with open(project_root / 'data' / 'processed' / 'label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    # Convert id_to_label keys to integers
    id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
    num_labels = len(id_to_label)
    
    return train_df, test_df, id_to_label, num_labels


def create_tokenize_function(tokenizer, max_length=512):
    """Create a tokenization function."""
    def tokenize_function(examples):
        """Tokenize the text."""
        return tokenizer(
            examples['Ticket Description'],
            truncation=True,
            padding='max_length',
            max_length=max_length
        )
    return tokenize_function


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Load data
    print("Loading data...")
    train_df, test_df, id_to_label, num_labels = load_data()
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {list(id_to_label.values())}")
    
    # Initialize tokenizer and model
    model_name = 'distilbert-base-uncased'
    print(f"\nLoading tokenizer and model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Convert DataFrames to Hugging Face Datasets
    print("\nConverting to Hugging Face datasets...")
    train_dataset = Dataset.from_pandas(train_df[['Ticket Description', 'Ticket Type Encoded']])
    test_dataset = Dataset.from_pandas(test_df[['Ticket Description', 'Ticket Type Encoded']])
    
    # Rename columns for Trainer
    train_dataset = train_dataset.rename_column('Ticket Type Encoded', 'labels')
    test_dataset = test_dataset.rename_column('Ticket Type Encoded', 'labels')
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenize_fn = create_tokenize_function(tokenizer)
    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=['Ticket Description']
    )
    test_dataset = test_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=['Ticket Description']
    )
    
    # Set format for PyTorch
    train_dataset.set_format('torch')
    test_dataset.set_format('torch')
    
    # Define output directory
    output_dir = project_root / 'models' / 'classifier'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        save_total_limit=2,
        seed=42,
    )
    
    # Initialize Trainer
    print("\nInitializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("\nStarting training...")
    print("="*50)
    train_result = trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    print("="*50)
    eval_results = trainer.evaluate()
    
    print("\nEvaluation Results:")
    print("-" * 50)
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("-" * 50)
    
    # Save the model and tokenizer
    print(f"\nSaving model and tokenizer to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping for inference
    label_mapping_file = output_dir / 'label_mapping.json'
    with open(project_root / 'data' / 'processed' / 'label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    
    with open(label_mapping_file, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print("\n" + "="*50)
    print("[SUCCESS] Training completed!")
    print("="*50)
    print(f"Model saved to: {output_dir}")
    print(f"Tokenizer saved to: {output_dir}")
    print(f"Label mapping saved to: {label_mapping_file}")
    print("\nFinal Test Metrics:")
    print(f"  Accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    print(f"  Precision: {eval_results.get('eval_precision', 0):.4f}")
    print(f"  Recall: {eval_results.get('eval_recall', 0):.4f}")
    print(f"  F1 Score: {eval_results.get('eval_f1', 0):.4f}")
    print("="*50)


if __name__ == '__main__':
    main()
