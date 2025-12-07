"""
Test script to verify that invalid/random inputs are properly rejected.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from inference_api import classify_ticket_via_api, check_domain_keywords

def test_invalid_inputs():
    """Test that random/unrelated inputs return 'invalid'."""
    
    print("=" * 60)
    print("Testing Invalid Input Detection")
    print("=" * 60)
    
    # Test cases that should return "invalid"
    invalid_test_cases = [
        "I have a headache",
        "aaaaaaaa",
        "what is your name",
        "hello world",
        "the weather is nice today",
        "how are you doing",
        "random text here",
        "123456789",
        "test test test",
        "lorem ipsum dolor sit amet",
        "can you help me with my homework",
        "what time is it",
        "tell me a joke",
        "how do I cook pasta",
    ]
    
    # Test cases that should pass (contain domain keywords)
    valid_test_cases = [
        "I have a billing issue with my account",
        "There's a bug in the login system",
        "I want to request a new feature",
        "My account is locked",
        "I was charged twice for my subscription",
        "The app crashes when I click the button",
        "Can you add a dark mode feature?",
        "I can't access my account settings",
    ]
    
    print("\n[1] Testing Invalid Inputs (should return 'invalid')")
    print("-" * 60)
    all_passed = True
    
    for test_input in invalid_test_cases:
        result = classify_ticket_via_api(test_input)
        keyword_check = check_domain_keywords(test_input)
        status = "[PASS]" if result == "invalid" else "[FAIL]"
        if result != "invalid":
            all_passed = False
        print(f"{status} | Keywords found: {keyword_check} | Result: {result}")
        print(f"  Input: '{test_input}'")
        print()
    
    print("\n[2] Testing Valid Inputs (should return a category)")
    print("-" * 60)
    
    for test_input in valid_test_cases:
        keyword_check = check_domain_keywords(test_input)
        result = classify_ticket_via_api(test_input)
        # Valid inputs should either return a category or an error (not "invalid")
        is_valid = result != "invalid" and not result.startswith("[Error]")
        status = "[PASS]" if is_valid else "[FAIL]"
        if not is_valid:
            all_passed = False
        print(f"{status} | Keywords found: {keyword_check} | Result: {result}")
        print(f"  Input: '{test_input}'")
        print()
    
    print("=" * 60)
    if all_passed:
        print("[SUCCESS] All tests passed!")
    else:
        print("[FAILURE] Some tests failed!")
    print("=" * 60)
    
    return all_passed

if __name__ == '__main__':
    test_invalid_inputs()

