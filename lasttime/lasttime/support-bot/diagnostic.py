from huggingface_hub import HfApi, whoami, InferenceClient, HfFolder
import subprocess, os

print("=== Hugging Face Diagnostic ===")
print("Checking library version...")
import huggingface_hub
print("huggingface_hub version:", huggingface_hub.__version__)

token = HfFolder.get_token()
print("Token found?" , bool(token))

print("\nChecking API login:")
try:
    user = whoami(token=token)
    print("[SUCCESS] Logged in as:", user["name"])
except Exception as e:
    print("[ERROR] API whoami() failed:", e)

print("\nTesting model connection:")
try:
    # For older huggingface_hub versions, use model directly
    client = InferenceClient(model="gpt2", token=token)
    res = client.text_generation("Hello from Hugging Face!", max_new_tokens=10)
    print("[SUCCESS] Model test successful:", res)
except Exception as e:
    print("[ERROR] Model test failed:", str(e))
    import traceback
    traceback.print_exc()

