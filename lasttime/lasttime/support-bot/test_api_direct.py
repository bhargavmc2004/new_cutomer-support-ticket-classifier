"""
Direct test of Hugging Face Inference API to diagnose the StopIteration issue.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import traceback

# Load environment variables
project_root = Path(__file__).parent
env_path = project_root / '.env'
load_dotenv(env_path)

HF_TOKEN = os.getenv('HF_TOKEN')
print(f"Token found: {bool(HF_TOKEN)}")

# Test with a simple model first
print("\n=== Testing with gpt2 (known to work) ===")
try:
    client_gpt2 = InferenceClient(model="gpt2", token=HF_TOKEN)
    response = client_gpt2.text_generation("Hello", max_new_tokens=5)
    print(f"GPT-2 response: {response}")
    print("GPT-2 test: SUCCESS")
except Exception as e:
    print(f"GPT-2 test: FAILED - {type(e).__name__}: {e}")
    traceback.print_exc()

# Test with Phi-3
print("\n=== Testing with microsoft/Phi-3-mini-4k-instruct ===")
try:
    client_phi3 = InferenceClient(model="microsoft/Phi-3-mini-4k-instruct", token=HF_TOKEN)
    print("Client initialized successfully")
    
    # Try text_generation
    print("Trying text_generation...")
    try:
        response = client_phi3.text_generation("Hello", max_new_tokens=5)
        print(f"text_generation response: {response}")
        print("text_generation: SUCCESS")
    except Exception as e:
        print(f"text_generation: FAILED - {type(e).__name__}: {e}")
        traceback.print_exc()
    
    # Try chat_completion
    print("Trying chat_completion...")
    try:
        messages = [{"role": "user", "content": "Hello"}]
        response = client_phi3.chat_completion(messages=messages, max_tokens=5)
        print(f"chat_completion response: {response}")
        print("chat_completion: SUCCESS")
    except Exception as e:
        print(f"chat_completion: FAILED - {type(e).__name__}: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"Client initialization: FAILED - {type(e).__name__}: {e}")
    traceback.print_exc()

