"""
Combined classifier + generator inference CLI using Hugging Face API.
"""

import argparse
from pathlib import Path
from inference_api import classify_ticket_via_api, generate_response_via_api


def main():
    """Main inference function using Hugging Face API."""
    parser = argparse.ArgumentParser(
        description='Classify ticket and generate response using Hugging Face API'
    )
    parser.add_argument(
        'ticket_text',
        type=str,
        help='The ticket description text'
    )
    parser.add_argument(
        '--category',
        type=str,
        default=None,
        help='Optional: Pre-classified category (if not provided, will classify automatically)'
    )
    parser.add_argument(
        '--classify-only',
        action='store_true',
        help='Only classify the ticket, do not generate response'
    )
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate response (requires --category)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Support Bot Inference (Hugging Face API)")
    print("="*60)
    
    # Classify ticket
    if not args.generate_only:
        print("\n[1] Classifying ticket...")
        print("-" * 60)
        try:
            category = classify_ticket_via_api(args.ticket_text)
            print(f"✓ Classification successful!")
            print(f"  Category: {category}")
            
            if args.classify_only:
                print("\n" + "="*60)
                print("[SUCCESS] Classification complete!")
                print("="*60)
                return
        except Exception as e:
            print(f"✗ Classification failed: {e}")
            return
    else:
        if not args.category:
            print("Error: --generate-only requires --category to be specified")
            return
        category = args.category
        print(f"\nUsing provided category: {category}")
    
    # Generate response
    if not args.classify_only:
        print("\n[2] Generating response...")
        print("-" * 60)
        try:
            response = generate_response_via_api(args.ticket_text, category)
            print(f"✓ Response generation successful!")
            print(f"\nGenerated Response:")
            print("-" * 60)
            print(response)
            print("-" * 60)
        except Exception as e:
            print(f"✗ Response generation failed: {e}")
            return
    
    print("\n" + "="*60)
    print("[SUCCESS] Inference complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  Ticket: {args.ticket_text[:80]}...")
    print(f"  Category: {category}")
    if not args.classify_only:
        print(f"  Response generated: Yes")
    print("="*60)


if __name__ == '__main__':
    main()
