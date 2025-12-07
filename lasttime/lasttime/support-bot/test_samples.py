"""
Test script for sample tickets to verify classification accuracy.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from inference_api import classify_ticket_via_api, generate_response_via_api, get_model_name

# Test cases with expected categories
test_cases = [
    {
        "ticket": "The app crashes when I upload a large file.",
        "expected": "bug"
    },
    {
        "ticket": "I was charged twice this month.",
        "expected": "billing"
    },
    {
        "ticket": "Can you add a dark mode?",
        "expected": "feature"
    },
    {
        "ticket": "I forgot my password.",
        "expected": "account"
    }
]


def main():
    """Run test cases."""
    print("=" * 60)
    print("Testing Support Bot Classification")
    print("=" * 60)
    print(f"Model: {get_model_name()}")
    print("=" * 60)
    print()
    
    results = []
    
    for idx, test_case in enumerate(test_cases, 1):
        ticket = test_case["ticket"]
        expected = test_case["expected"]
        
        print(f"[Test {idx}/{len(test_cases)}]")
        print(f"Ticket: {ticket}")
        print(f"Expected Category: {expected}")
        print("-" * 60)
        
        # Classify
        try:
            predicted = classify_ticket_via_api(ticket)
            
            if predicted.startswith("[Error]"):
                print(f"[ERROR] Classification failed: {predicted}")
                results.append({
                    "ticket": ticket,
                    "expected": expected,
                    "predicted": "ERROR",
                    "match": False
                })
                print()
                continue
            
            match = predicted.lower() == expected.lower()
            status = "[SUCCESS]" if match else "[FAILED]"
            
            print(f"{status} Predicted Category: {predicted}")
            print(f"{status} Match: {match}")
            
            # Generate response
            try:
                response = generate_response_via_api(ticket, predicted)
                if response.startswith("[Error]"):
                    print(f"[WARNING] Response generation failed: {response}")
                    response = "N/A"
                else:
                    print(f"[SUCCESS] Response generated ({len(response)} chars)")
            except Exception as e:
                print(f"[WARNING] Response generation error: {e}")
                response = "N/A"
            
            results.append({
                "ticket": ticket,
                "expected": expected,
                "predicted": predicted,
                "match": match,
                "response": response[:100] + "..." if len(str(response)) > 100 else response
            })
            
        except Exception as e:
            print(f"[ERROR] Test failed with exception: {e}")
            results.append({
                "ticket": ticket,
                "expected": expected,
                "predicted": "EXCEPTION",
                "match": False
            })
        
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    total = len(results)
    matches = sum(1 for r in results if r.get("match", False))
    accuracy = (matches / total * 100) if total > 0 else 0
    
    print(f"Total Tests: {total}")
    print(f"Correct Predictions: {matches}")
    print(f"Accuracy: {accuracy:.1f}%")
    print()
    
    print("Detailed Results:")
    print("-" * 60)
    for idx, result in enumerate(results, 1):
        status = "[SUCCESS]" if result.get("match", False) else "[FAILED]"
        print(f"{status} Test {idx}: Expected={result['expected']}, Predicted={result['predicted']}")
    
    print("=" * 60)


if __name__ == '__main__':
    main()

