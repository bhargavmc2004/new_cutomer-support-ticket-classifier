"""
Quick demo evaluation script.
Runs evaluation on test samples and saves results.
"""

import pandas as pd
from pathlib import Path
from inference_api import quick_eval_on_test


def main():
    """Run quick evaluation and save results."""
    project_root = Path(__file__).parent.parent
    
    print("="*60)
    print("Quick Demo Evaluation")
    print("="*60)
    print("\nRunning evaluation on 20 samples from test.csv...")
    print()
    
    # Run evaluation
    results = quick_eval_on_test(num_samples=20)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    
    # Count matches
    matches = df_results['match'].sum()
    total = len(df_results)
    accuracy = (matches / total) * 100 if total > 0 else 0
    
    print(f"\nTotal samples evaluated: {total}")
    print(f"Matches: {matches}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Print each result
    print("\n" + "-"*60)
    print("Detailed Results:")
    print("-"*60)
    for _, row in df_results.iterrows():
        match_symbol = "✓" if row['match'] else "✗"
        print(f"\n{match_symbol} Sample {row['sample_id']}:")
        print(f"  True Label:    {row['true_label']}")
        print(f"  Predicted:     {row['predicted_label']}")
        print(f"  Ticket (preview): {row['ticket_text'][:100]}...")
    
    # Save results
    output_file = project_root / 'demo_eval_results.csv'
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\n" + "="*60)
    print("[SUCCESS] Evaluation complete!")
    print("="*60)
    print(f"\nResults saved to: {output_file}")
    print(f"Total samples: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()

