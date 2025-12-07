"""
Inference API for classification and response generation using Hugging Face InferenceClient.
"""

import os
import json
import re
import requests
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False
from huggingface_hub import InferenceClient

# Load environment variables
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)

# Get API token from environment
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACEHUB_API_TOKEN')
if not HF_TOKEN:
    try:
        import streamlit as st
        HF_TOKEN = st.secrets.get('HF_TOKEN') or st.secrets.get('HUGGINGFACEHUB_API_TOKEN')
    except Exception:
        pass

if not HF_TOKEN:
    print("[WARNING] No HF token found in environment or Streamlit secrets.")

# Model fallback sequence - try in order until one works
MODEL_OPTIONS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "tiiuae/falcon-7b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2"
]

client = None
model_name = None  # Will be updated when a model successfully responds

print("=" * 60)
print("Initializing Hugging Face Inference Client")
print("=" * 60)

# Try to initialize InferenceClient - initialize without model, specify in calls
# This avoids the provider mapping bug in huggingface_hub 0.36.0
try:
    print(f"[INFO] Initializing InferenceClient (will specify model per call)...")
    # Initialize client without model (will specify per call)
    # Note: We'll create model-specific clients in the functions
    client = InferenceClient(
        token=HF_TOKEN
    )
    model_name = MODEL_OPTIONS[0]  # Default to first model
    print(f"[SUCCESS] InferenceClient initialized (will use: {model_name})")
except Exception as e:
    print(f"[FAILED] Failed to initialize InferenceClient: {str(e)}")
    client = None
    model_name = None

if client is None:
    print(f"[WARNING] Failed to initialize any model. Tried: {', '.join(MODEL_OPTIONS)}")
    print("[WARNING] Check your HF token permissions if inference fails.")
    print("=" * 60)
else:
    print(f"[SUCCESS] Using model: {model_name}")
    print("=" * 60)


def check_domain_keywords(ticket_text):
    """
    Check if the ticket text contains keywords related to valid support categories.
    Uses word boundaries for single-word keywords to avoid false positives.
    
    Args:
        ticket_text (str): The ticket description text
        
    Returns:
        bool: True if keywords are found, False otherwise
    """
    if not ticket_text:
        return False
    
    ticket_lower = ticket_text.lower()
    
    # Define domain keywords for each category
    # Single-word keywords (will use word boundaries)
    billing_keywords_single = [
        'billing', 'bill', 'charge', 'charged', 'charges', 'payment', 'pay', 'paid', 
        'invoice', 'invoices', 'cost', 'price', 'pricing', 'fee', 'fees',
        'subscription', 'subscriptions', 'renewal', 'renew', 'refund', 'refunds',
        'credit', 'credits', 'debit', 'transaction', 'transactions', 'purchase',
        'purchases', 'money', 'dollar', 'dollars', 'expense', 'expenses',
        'overcharge', 'overcharged', 'discount', 'discounts', 'coupon', 'coupons',
        'voucher', 'vouchers', 'receipt', 'receipts', 'statement', 'statements'
    ]
    
    bug_technical_keywords_single = [
        'bug', 'bugs', 'error', 'errors', 'crash', 'crashes', 'crashed', 'crashing',
        'broken', 'break', 'breaks', 'issue', 'issues', 'problem', 'problems', 
        'glitch', 'glitches', 'malfunction', 'malfunctions', 'technical', 'technically',
        'failure', 'failures', 'failed', 'failing', 'freeze', 'frozen', 'freezing',
        'hang', 'hanging', 'stuck', 'slow', 'slower', 'slowly', 'lag', 'lagging',
        'timeout', 'timeouts', 'connection', 'connections', 'connect', 'connecting',
        'disconnect', 'disconnected', 'server', 'servers', 'database', 'api',
        'code', 'coding', 'programming', 'software', 'hardware', 'system', 'systems',
        'application', 'applications', 'app', 'website', 'web', 'page', 'pages',
        'button', 'buttons', 'link', 'links', 'login', 'signin', 'logout', 'signout',
        'password', 'passwords', 'authentication', 'authorization', 'access', 'accessing',
        'permission', 'permissions', 'security', 'secure', 'encryption', 'encrypted'
    ]
    
    feature_keywords_single = [
        'feature', 'features', 'request', 'requests', 'suggest', 'suggestion', 'suggestions',
        'add', 'adding', 'addition', 'new', 'enhancement', 'enhancements', 'enhance',
        'improve', 'improvement', 'improvements', 'upgrade', 'upgrades', 'update', 'updates',
        'functionality', 'function', 'functions', 'capability', 'capabilities', 'option',
        'options', 'setting', 'settings', 'preference', 'preferences', 'customize',
        'customization', 'integrate', 'integration', 'integrations', 'tool', 'tools',
        'widget', 'widgets', 'plugin', 'plugins', 'extension', 'extensions',
        'wish', 'wishlist', 'roadmap', 'roadmaps', 'future', 'coming', 'planned',
        'want', 'wants', 'need', 'needs', 'require', 'requires', 'requirement',
        'requirements', 'idea', 'ideas', 'proposal', 'proposals', 'recommend', 'recommendation'
    ]
    
    account_keywords_single = [
        'account', 'accounts', 'profile', 'profiles', 'user', 'users', 'username', 'usernames',
        'email', 'emails', 'address', 'addresses', 'contact', 'contacts', 'information',
        'info', 'details', 'personal', 'privacy', 'private', 'data',
        'verify', 'verification', 'verified', 'unverified', 'confirm', 'confirmation',
        'activate', 'activation', 'deactivate', 'deactivation', 'suspend', 'suspended',
        'suspension', 'close', 'closed', 'closing', 'delete', 'deleted', 'deletion',
        'remove', 'removed', 'removal', 'cancel', 'cancelled', 'cancellation', 'cancellations',
        'reactivate', 'reactivation', 'restore', 'restoration', 'recover', 'recovery',
        'locked', 'lock', 'unlock', 'unlocked', 'blocked', 'block', 'unblock', 'unblocked',
        'authorization', 'authorize', 'membership', 'memberships', 'plan', 'plans',
        'tier', 'tiers', 'level', 'levels', 'status', 'statuses', 'state', 'states'
    ]
    
    # Multi-word phrases (will use simple substring matching)
    multi_word_phrases = [
        'not working', "doesn't work", "don't work", 'technical issue', 'technical problem',
        'server error', 'server issue', 'database error', 'api error', 'log in', 'sign in',
        'log out', 'sign out', 'personal data', 'would like'
    ]
    
    # Check single-word keywords with word boundaries
    all_single_keywords = (billing_keywords_single + bug_technical_keywords_single + 
                          feature_keywords_single + account_keywords_single)
    
    for keyword in all_single_keywords:
        # Use word boundary regex to match whole words only
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, ticket_lower):
            return True
    
    # Check multi-word phrases
    for phrase in multi_word_phrases:
        if phrase in ticket_lower:
            return True
    
    return False


def classify_ticket_via_api(ticket_text):
    """
    Classify a ticket using Hugging Face InferenceClient.
    
    Args:
        ticket_text (str): The ticket description text
        
    Returns:
        str: The predicted label (one of: billing, feature, bug, account) or error message
    """
    if not ticket_text or not ticket_text.strip():
        return "[Error] API call failed."
    
    # Check for domain keywords before calling the ML model
    if not check_domain_keywords(ticket_text):
        print("[INFO] Input does not contain keywords related to billing, technical issues, features, or account issues.")
        return "invalid"
    
    if client is None:
        print("[ERROR] Classification API call failed: Model not initialized")
        print("[WARNING] Check your HF token permissions if inference fails.")
        return "[Error] API call failed."
    
    # Create classification prompt as specified
    prompt = f"""You are a customer support assistant. Read the following ticket and classify it as one of: billing, bug, feature, or account issue.
Return only the category name.
Ticket: {ticket_text}"""
    
    try:
        print(f"[INFO] Classifying ticket (length: {len(ticket_text)} chars)...")
        
        # Try each model in the fallback list until one works
        last_error = None
        for model_to_try in MODEL_OPTIONS:
            try:
                print(f"[INFO] Trying model: {model_to_try}")
                
                # Create a client for this specific model (don't use base_url with model)
                model_client = InferenceClient(
                    model=model_to_try,
                    token=HF_TOKEN
                )
                
                # Try chat_completion first for instruction-tuned models
                try:
                    messages = [{"role": "user", "content": prompt}]
                    result = model_client.chat_completion(
                        messages=messages,
                        max_tokens=20,
                        temperature=0.1   # temperature 0.1 for Phi-3-mini-4k-instruct, 0.5 for other models
                    )
                    # Extract text from chat completion response
                    if isinstance(result, dict):
                        if "choices" in result and len(result["choices"]) > 0:
                            response = result["choices"][0].get("message", {}).get("content", "")
                        elif "generated_text" in result:
                            response = result["generated_text"]
                        else:
                            response = str(result)
                    else:
                        response = str(result)
                except (AttributeError, KeyError, TypeError) as e:
                    # Fallback to text_generation
                    print(f"[INFO] chat_completion failed, trying text_generation...")
                    response = model_client.text_generation(
                        prompt=prompt,
                        max_new_tokens=20,
                        temperature=0.1
                    )
                    # Remove prompt if included
                    if isinstance(response, str) and prompt in response:
                        response = response.replace(prompt, "").strip()
                
                # Update global model_name if this worked
                global model_name
                model_name = model_to_try
                print(f"[SUCCESS] Model {model_to_try} worked!")
                break
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                # Handle HTTP errors specifically
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        error_msg = f"{type(e).__name__}: {e.response.status_code} - {error_detail}"
                    except:
                        error_msg = f"{type(e).__name__}: {e.response.status_code} - {e.response.text[:100]}"
                print(f"[INFO] Model {model_to_try} failed: {error_msg}")
                if model_to_try != MODEL_OPTIONS[-1]:
                    print(f"   -> Trying next model...")
                continue
        
        if 'response' not in locals() or not response:
            raise last_error if last_error else Exception("All models failed")
        
        if response is None:
            print("[ERROR] Classification API call failed: Response is None")
            return "[Error] API call failed."
        
        if not response:
            print("[ERROR] Classification API call failed: Empty response from API")
            return "[Error] API call failed."
        
        # Extract and clean the label
        label = str(response).strip().lower()
        
        if not label:
            print("[ERROR] Classification API call failed: Empty label extracted from response")
            return "[Error] API call failed."
        
        # Validate label is one of the expected values
        valid_labels = ['billing', 'bug', 'feature', 'account']
        
        # Check if the response contains a valid label
        for valid_label in valid_labels:
            if valid_label in label:
                print(f"[SUCCESS] Classification result: {valid_label}")
                return valid_label
        
        # If no exact match, try to extract from response
        # Check for "account issue" or similar
        if 'account' in label or 'issue' in label:
            print(f"[SUCCESS] Classification result: account")
            return 'account'
        
        # Default to first valid label found or 'billing' as fallback
        words = label.split()
        for word in words:
            if word in valid_labels:
                print(f"[SUCCESS] Classification result: {word}")
                return word
        
        # Fallback
        print(f"[WARNING] Classification fallback: billing (could not extract valid category)")
        return 'billing'
        
    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e) if str(e) else "Unknown error (empty error message)"
        print(f"[ERROR] Classification API call failed: {error_type} - {error_details}")
        print("[WARNING] Check your HF token permissions if inference fails.")
        return "[Error] API call failed."


def generate_response_via_api(ticket_text, category):
    """
    Generate a response using Hugging Face InferenceClient.
    
    Args:
        ticket_text (str): The ticket description text
        category (str): The ticket category/label
        
    Returns:
        str: A generated response (professional and empathetic) or error message
    """
    if not ticket_text or not ticket_text.strip():
        return "[Error] API call failed."
    
    if client is None:
        print("[ERROR] Response generation API call failed: Model not initialized")
        print("[WARNING] Check your HF token permissions if inference fails.")
        return "[Error] API call failed."
    
    if not category:
        category = "general"
    
    # Create prompt for response generation as specified
    prompt = f"""You are a polite, professional customer support assistant. Write a short helpful response to this customer ticket:
{ticket_text}"""
    
    try:
        print(f"[INFO] Generating response (category: {category}, ticket length: {len(ticket_text)} chars)...")
        
        # Try each model in the fallback list until one works
        last_error = None
        for model_to_try in MODEL_OPTIONS:
            try:
                print(f"[INFO] Trying model: {model_to_try}")
                
                # Create a client for this specific model (don't use base_url with model)
                model_client = InferenceClient(
                    model=model_to_try,
                    token=HF_TOKEN
                )
                
                # Try chat_completion first for instruction-tuned models
                try:
                    messages = [{"role": "user", "content": prompt}]
                    result = model_client.chat_completion(
                        messages=messages,
                        max_tokens=150,
                        temperature=0.7
                    )
                    # Extract text from chat completion response
                    if isinstance(result, dict):
                        if "choices" in result and len(result["choices"]) > 0:
                            response = result["choices"][0].get("message", {}).get("content", "")
                        elif "generated_text" in result:
                            response = result["generated_text"]
                        else:
                            response = str(result)
                    else:
                        response = str(result)
                except (AttributeError, KeyError, TypeError) as e:
                    # Fallback to text_generation
                    print(f"[INFO] chat_completion failed, trying text_generation...")
                    response = model_client.text_generation(
                        prompt=prompt,
                        max_new_tokens=150,
                        temperature=0.7
                    )
                    # Remove prompt if included
                    if isinstance(response, str) and prompt in response:
                        response = response.replace(prompt, "").strip()
                
                # Update global model_name if this worked
                global model_name
                model_name = model_to_try
                print(f"[SUCCESS] Model {model_to_try} worked!")
                break
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                # Handle HTTP errors specifically
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        error_msg = f"{type(e).__name__}: {e.response.status_code} - {error_detail}"
                    except:
                        error_msg = f"{type(e).__name__}: {e.response.status_code} - {e.response.text[:100]}"
                print(f"[INFO] Model {model_to_try} failed: {error_msg}")
                if model_to_try != MODEL_OPTIONS[-1]:
                    print(f"   -> Trying next model...")
                continue
        
        if 'response' not in locals() or not response:
            raise last_error if last_error else Exception("All models failed")
        
        if response is None:
            print("[ERROR] Response generation API call failed: Response is None")
            return "[Error] API call failed."
        
        if not response:
            print("[ERROR] Response generation API call failed: Empty response from API")
            return "[Error] API call failed."
        
        # Extract and clean the response
        response_text = str(response).strip()
        
        # Clean up the response
        response_text = ' '.join(response_text.split())
        
        # Ensure response ends properly
        if response_text:
            # Remove any trailing incomplete sentences
            sentences = [s.strip() for s in response_text.split('.') if s.strip()]
            if sentences:
                # Join sentences and ensure it ends with a period
                response_text = '. '.join(sentences)
                if not response_text.endswith('.'):
                    response_text += '.'
            else:
                response_text = response_text.rstrip('.') + '.'
        
        # Limit response length
        if len(response_text) > 300:
            sentences = response_text.split('.')
            if len(sentences) > 1:
                response_text = '. '.join(sentences[:-1]) + '.'
            else:
                response_text = response_text[:300] + "..."
        
        # Print success message
        print(f"[SUCCESS] Response generated successfully (Category: {category})")
        
        return response_text
        
    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e) if str(e) else "Unknown error (empty error message)"
        print(f"[ERROR] Response generation API call failed: {error_type} - {error_details}")
        print("[WARNING] Check your HF token permissions if inference fails.")
        return "[Error] API call failed."


def get_model_name():
    """Return the currently initialized model name."""
    return model_name if model_name else "No model initialized"


def quick_eval_on_test(num_samples=20):
    """
    Quick evaluation on test samples.
    
    Args:
        num_samples (int): Number of samples to evaluate (default: 20)
        
    Returns:
        list: List of dictionaries with ticket_text, true_label, predicted_label
    """
    import pandas as pd
    import json
    
    # Load test data
    project_root = Path(__file__).parent.parent
    test_csv = project_root / 'data' / 'processed' / 'test.csv'
    label_mapping_file = project_root / 'data' / 'processed' / 'label_mapping.json'
    
    # Load test data
    df = pd.read_csv(test_csv)
    
    # Load label mapping
    with open(label_mapping_file, 'r') as f:
        label_mapping = json.load(f)
    
    id_to_label = {int(k): v for k, v in label_mapping['id_to_label'].items()}
    
    # Sample num_samples rows
    if len(df) > num_samples:
        df_sample = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    else:
        df_sample = df.copy()
    
    results = []
    
    print(f"Evaluating {len(df_sample)} samples...")
    print("-" * 60)
    
    for idx, row in df_sample.iterrows():
        ticket_text = str(row['Ticket Description'])
        true_label_id = int(row['Ticket Type Encoded'])
        true_label = id_to_label.get(true_label_id, f"Unknown (ID: {true_label_id})")
        
        # Predict label
        try:
            predicted_label = classify_ticket_via_api(ticket_text)
            # Check if it's an error message
            if predicted_label.startswith("[Error]"):
                predicted_label = "ERROR"
        except Exception as e:
            print(f"Error predicting sample {idx + 1}: {e}")
            predicted_label = "ERROR"
        
        # Create a mapping between true labels and API labels
        label_mapping_dict = {
            'billing inquiry': 'billing',
            'billing': 'billing',
            'technical issue': 'bug',
            'bug': 'bug',
            'product inquiry': 'feature',
            'feature': 'feature',
            'cancellation request': 'account',
            'refund request': 'account',
            'account': 'account',
            'other': 'other'
        }
        
        # Normalize true label for comparison
        true_label_normalized = true_label.lower()
        predicted_label_normalized = predicted_label.lower()
        
        # Check if they match directly or through mapping
        match = (
            true_label_normalized == predicted_label_normalized or
            label_mapping_dict.get(true_label_normalized) == predicted_label_normalized or
            any(word in true_label_normalized for word in predicted_label_normalized.split())
        )
        
        results.append({
            'sample_id': idx + 1,
            'ticket_text': ticket_text[:200] + "..." if len(ticket_text) > 200 else ticket_text,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'match': match
        })
        
        print(f"Sample {idx + 1}/{len(df_sample)}: True={true_label}, Predicted={predicted_label}")
    
    return results


def main():
    """Test the inference functions."""
    print("="*60)
    print("Testing Hugging Face Inference API")
    print("="*60)
    print(f"Using model: {get_model_name()}")
    
    # Test classification
    print("\n[1] Testing Classification...")
    print("-" * 60)
    test_ticket = "I'm having trouble logging into my account. The password reset isn't working."
    
    try:
        category = classify_ticket_via_api(test_ticket)
        if category.startswith("[Error]"):
            print(f"[ERROR] {category}")
            return
        print(f"[SUCCESS] Classification successful!")
        print(f"  Ticket: {test_ticket[:60]}...")
        print(f"  Category: {category}")
    except Exception as e:
        print(f"[ERROR] Classification failed: {e}")
        return
    
    # Test response generation
    print("\n[2] Testing Response Generation...")
    print("-" * 60)
    
    try:
        response = generate_response_via_api(test_ticket, category)
        if response.startswith("[Error]"):
            print(f"[ERROR] {response}")
            return
        print(f"[SUCCESS] Response generation successful!")
        print(f"  Category: {category}")
        print(f"  Generated Response:\n  {response}")
    except Exception as e:
        print(f"[ERROR] Response generation failed: {e}")
        return
    
    print("\n" + "="*60)
    print("[SUCCESS] All tests passed!")
    print("="*60)


if __name__ == '__main__':
    main()
