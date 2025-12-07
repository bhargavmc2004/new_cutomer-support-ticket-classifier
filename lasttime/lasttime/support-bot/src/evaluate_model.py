"""
Model Evaluation Script
Tests classification accuracy on test samples and calculates metrics.
"""

import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
from inference_api import classify_ticket_via_api


def map_true_label_to_api_label(true_label):
    """
    Map true labels from dataset to API label format.
    
    Args:
        true_label (str): True label from dataset
        
    Returns:
        str: Mapped label for API comparison
    """
    label_mapping = {
        'Billing inquiry': 'billing',
        'Technical issue': 'bug',
        'Product inquiry': 'feature',
        'Cancellation request': 'account',
        'Refund request': 'account',
    }
    
    return label_mapping.get(true_label, 'other')


def load_test_data(num_samples=30):
    """
    Load test data samples.
    
    Args:
        num_samples (int): Number of samples to load (default: 30)
        
    Returns:
        tuple: (DataFrame, dict) - Test samples and label mapping
    """
    project_root = Path(__file__).parent.parent
    test_csv = project_root / 'data' / 'processed' / 'test.csv'
    label_mapping_file = project_root / 'data' / 'processed' / 'label_mapping.json'
    
    # Load test data
    df = pd.read_csv(test_csv)
    
    # Load label mapping
    with open(label_mapping_file, 'r') as f:
        label_mapping = json.load(f)
    
    id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
    
    # Sample rows
    if len(df) > num_samples:
        df_sample = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    else:
        df_sample = df.copy()
    
    return df_sample, id_to_label


def evaluate_classification(num_samples=30):
    """
    Evaluate classification model on test samples.
    
    Args:
        num_samples (int): Number of samples to evaluate (default: 30)
    """
    print("="*70)
    print("Model Evaluation - Classification Accuracy")
    print("="*70)
    
    # Load test data
    print(f"\nLoading {num_samples} random samples from test set...")
    df_sample, id_to_label = load_test_data(num_samples)
    
    print(f"Loaded {len(df_sample)} samples")
    print("-" * 70)
    
    # Store results
    true_labels = []
    predicted_labels = []
    results = []
    
    # Evaluate each sample
    print("\nEvaluating samples...")
    print("-" * 70)
    
    for idx, row in df_sample.iterrows():
        ticket_text = str(row['Ticket Description'])
        true_label_id = int(row['Ticket Type Encoded'])
        true_label = id_to_label.get(true_label_id, f"Unknown (ID: {true_label_id})")
        
        # Map true label to API format
        true_label_api = map_true_label_to_api_label(true_label)
        
        # Predict label
        try:
            predicted_label = classify_ticket_via_api(ticket_text)
        except Exception as e:
            print(f"Error predicting sample {idx + 1}: {e}")
            predicted_label = "error"
        
        # Store results
        true_labels.append(true_label_api)
        predicted_labels.append(predicted_label)
        
        results.append({
            'sample_id': idx + 1,
            'ticket_text': ticket_text[:150] + "..." if len(ticket_text) > 150 else ticket_text,
            'true_label': true_label,
            'true_label_api': true_label_api,
            'predicted_label': predicted_label,
            'correct': true_label_api == predicted_label
        })
        
        # Print progress every 5 samples
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx + 1}/{len(df_sample)} samples...")
    
    print(f"\nCompleted evaluation of {len(df_sample)} samples")
    print("-" * 70)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Precision, Recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, 
        predicted_labels, 
        average='weighted',
        zero_division=0
    )
    
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    
    # Per-class metrics
    print("\n" + "-" * 70)
    print("Per-Class Metrics:")
    print("-" * 70)
    
    unique_labels = sorted(set(true_labels + predicted_labels))
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=unique_labels,
        average=None,
        zero_division=0
    )
    
    print(f"\n{'Label':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    for i, label in enumerate(unique_labels):
        print(f"{label:<15} {precision_per_class[i]:<12.4f} {recall_per_class[i]:<12.4f} "
              f"{f1_per_class[i]:<12.4f} {support_per_class[i]:<10}")
    
    # Classification report
    print("\n" + "-" * 70)
    print("Detailed Classification Report:")
    print("-" * 70)
    print(classification_report(true_labels, predicted_labels, zero_division=0))
    
    # Sample comparisons
    print("\n" + "="*70)
    print("SAMPLE TICKET COMPARISONS")
    print("="*70)
    
    # Show 5-10 sample comparisons
    num_samples_to_show = min(10, len(results))
    sample_indices = np.random.choice(len(results), num_samples_to_show, replace=False)
    
    print(f"\nShowing {num_samples_to_show} sample tickets:")
    print("-" * 70)
    
    for i, idx in enumerate(sample_indices, 1):
        result = results[idx]
        match_symbol = "✓" if result['correct'] else "✗"
        
        print(f"\n[{i}] {match_symbol} Sample {result['sample_id']}")
        print(f"    Ticket: {result['ticket_text']}")
        print(f"    True Label:    {result['true_label']:20} (API: {result['true_label_api']})")
        print(f"    Predicted:     {result['predicted_label']:20}")
        print(f"    Match: {'CORRECT' if result['correct'] else 'INCORRECT'}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    correct_count = sum(1 for r in results if r['correct'])
    total_count = len(results)
    
    print(f"\nTotal Samples Evaluated: {total_count}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Incorrect Predictions: {total_count - correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Label distribution
    print("\n" + "-" * 70)
    print("True Label Distribution:")
    print("-" * 70)
    true_label_counts = pd.Series(true_labels).value_counts().sort_index()
    for label, count in true_label_counts.items():
        print(f"  {label}: {count}")
    
    print("\n" + "-" * 70)
    print("Predicted Label Distribution:")
    print("-" * 70)
    pred_label_counts = pd.Series(predicted_labels).value_counts().sort_index()
    for label, count in pred_label_counts.items():
        print(f"  {label}: {count}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Evaluation complete!")
    print("="*70)
    
    return results, accuracy, precision, recall, f1


def main():
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate classification model')
    parser.add_argument(
        '--num_samples',
        type=int,
        default=30,
        help='Number of samples to evaluate (default: 30, range: 20-50)'
    )
    
    args = parser.parse_args()
    
    # Clamp num_samples to 20-50 range
    num_samples = max(20, min(50, args.num_samples))
    
    results, accuracy, precision, recall, f1 = evaluate_classification(num_samples)
    
    print(f"\nFinal Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("="*70)


if __name__ == '__main__':
    main()

