"""
Dataset cleaning and split utilities.
"""

import pandas as pd
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import re


def clean_text(text):
    """Remove extra spaces from text."""
    if pd.isna(text):
        return text
    # Remove extra whitespace and normalize spaces
    text = str(text)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()  # Remove leading/trailing spaces
    return text


def main():
    # Get the project root directory (parent of src)
    project_root = Path(__file__).parent.parent
    
    # Define paths - check both data/ and data/raw/ for the CSV
    input_csv = project_root / 'data' / 'customer_support_tickets.csv'
    if not input_csv.exists():
        input_csv = project_root / 'data' / 'raw' / 'customer_support_tickets.csv'
    
    if not input_csv.exists():
        raise FileNotFoundError(
            f"CSV file not found. Checked:\n"
            f"  - {project_root / 'data' / 'customer_support_tickets.csv'}\n"
            f"  - {project_root / 'data' / 'raw' / 'customer_support_tickets.csv'}"
        )
    
    output_dir = project_root / 'data' / 'processed'
    label_mapping_file = output_dir / 'label_mapping.json'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading CSV file from: {input_csv}")
    # Load the CSV file
    df = pd.read_csv(input_csv)
    
    print(f"Original dataset shape: {df.shape}")
    
    # Keep only the required columns
    required_columns = ['Ticket Description', 'Ticket Type', 'Resolution']
    df = df[required_columns].copy()
    
    print(f"After selecting columns: {df.shape}")
    
    # Clean text by removing extra spaces
    print("Cleaning text...")
    df['Ticket Description'] = df['Ticket Description'].apply(clean_text)
    df['Resolution'] = df['Resolution'].apply(clean_text)
    
    # Drop rows with missing values in Ticket Description or Ticket Type
    print("Dropping rows with missing values...")
    initial_count = len(df)
    df = df.dropna(subset=['Ticket Description', 'Ticket Type'])
    dropped_count = initial_count - len(df)
    print(f"Dropped {dropped_count} rows with missing values")
    print(f"Remaining rows: {len(df)}")
    
    # Encode ticket types into numeric labels
    print("Encoding ticket types...")
    unique_types = sorted(df['Ticket Type'].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_types)}
    reverse_mapping = {idx: label for label, idx in label_mapping.items()}
    
    # Create encoded column
    df['Ticket Type Encoded'] = df['Ticket Type'].map(label_mapping)
    
    # Save label mapping as JSON
    with open(label_mapping_file, 'w', encoding='utf-8') as f:
        json.dump({
            'label_to_id': label_mapping,
            'id_to_label': reverse_mapping
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Label mapping saved to {label_mapping_file}")
    
    # Split the dataset into training and testing sets (80/20 split)
    print("Splitting dataset into train/test (80/20)...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['Ticket Type Encoded']  # Stratify to maintain class distribution
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    
    # Save processed CSVs
    train_output = output_dir / 'train.csv'
    test_output = output_dir / 'test.csv'
    
    train_df.to_csv(train_output, index=False, encoding='utf-8')
    test_df.to_csv(test_output, index=False, encoding='utf-8')
    
    print(f"\n[SUCCESS] Processed datasets saved:")
    print(f"  - Training set: {train_output}")
    print(f"  - Testing set: {test_output}")
    print(f"  - Label mapping: {label_mapping_file}")
    
    print("\n" + "="*50)
    print("Label Mapping:")
    print("="*50)
    for label, idx in sorted(label_mapping.items(), key=lambda x: x[1]):
        print(f"  {idx}: {label}")
    print("="*50)


if __name__ == '__main__':
    main()
