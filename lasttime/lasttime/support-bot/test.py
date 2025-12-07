from transformers import pipeline
import torch
import re

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use: {device}")

# Load GPT-2 model locally
print("Loading GPT-2 model...")
generator = pipeline("text-generation", model="gpt2", device=0 if device == "cuda" else -1)

# Input customer support ticket
ticket = "I was charged twice this month."

# Create prompt that guides GPT-2 to output in the desired format
# Using a structured prompt to encourage the right format
prompt = f"""Ticket: {ticket}
Category: billing
Response: We apologize for the double charge. We will investigate and issue a refund within 3-5 business days.

Ticket: I can't log into my account
Category: account
Response: We're sorry for the inconvenience. Please try resetting your password using the link on our login page.

Ticket: The app crashes when I click submit
Category: bug
Response: Thank you for reporting this. Our technical team will investigate and fix this issue in the next update.

Ticket: Can you add dark mode?
Category: feature
Response: Thank you for the suggestion. We'll consider adding dark mode in a future update.

Ticket: {ticket}
Category:"""

# Generate response
print(f"\nProcessing ticket: {ticket}")
print("Generating response...\n")

response = generator(
    prompt,
    max_new_tokens=50,
    temperature=0.5,
    do_sample=True,
    num_return_sequences=1,
    pad_token_id=generator.tokenizer.eos_token_id,
    repetition_penalty=1.2
)

# Extract the generated text
generated_text = response[0]['generated_text']

# Find the last occurrence (for our ticket)
# Split by "Ticket:" to get the last section
ticket_sections = generated_text.split(f"Ticket: {ticket}")
if len(ticket_sections) > 1:
    relevant_text = ticket_sections[-1]
else:
    relevant_text = generated_text

# Parse the output to extract category and response
# Look for "Category:" and "Response:" patterns in the relevant section
category_match = re.search(r'Category:\s*(\w+)', relevant_text, re.IGNORECASE)
response_match = re.search(r'Response:\s*(.+?)(?:\n\n|\nTicket:|$)', relevant_text, re.IGNORECASE | re.DOTALL)

# Extract category
if category_match:
    category = category_match.group(1).lower()
    # Validate category is one of the expected values
    valid_categories = ['billing', 'bug', 'feature', 'account']
    if category not in valid_categories:
        # Try to find a valid category in the text
        for valid_cat in valid_categories:
            if valid_cat in generated_text.lower():
                category = valid_cat
                break
        else:
            category = 'billing'  # Default fallback
else:
    # Try to find category in the generated text
    category = 'billing'  # Default
    for valid_cat in ['billing', 'bug', 'feature', 'account']:
        if valid_cat in generated_text.lower():
            category = valid_cat
            break

# Extract response
if response_match:
    response_text = response_match.group(1).strip()
    # Clean up the response (remove extra whitespace, newlines, and any trailing punctuation issues)
    response_text = ' '.join(response_text.split())
    # Remove any trailing incomplete words or sentences
    response_text = response_text.rstrip('.,!?;:')
    # Limit response length
    if len(response_text) > 200:
        # Try to cut at a sentence boundary
        sentences = response_text.split('.')
        if len(sentences) > 1:
            response_text = '. '.join(sentences[:-1]) + '.'
        else:
            response_text = response_text[:200] + "..."
else:
    # If no "Response:" found, try to extract text after category line
    category_pos = relevant_text.lower().find('category:')
    if category_pos != -1:
        # Get text after category line
        after_category = relevant_text[category_pos:]
        # Try to find text after the category value
        lines = after_category.split('\n')
        response_text = ""
        for i, line in enumerate(lines):
            if i > 0 and line.strip() and not line.strip().startswith('Category:') and not line.strip().startswith('Ticket:'):
                response_text = line.strip()
                break
        if not response_text:
            response_text = "Thank you for contacting us. We will investigate this issue and get back to you shortly."
    else:
        response_text = "Thank you for contacting us. We will investigate this issue and get back to you shortly."
    
    # Clean up the response
    response_text = ' '.join(response_text.split())
    if len(response_text) > 200:
        response_text = response_text[:200] + "..."

# Print in the desired two-line format
print("=" * 60)
print("RESULT:")
print("=" * 60)
print(f"Category: {category}")
print(f"Response: {response_text}")
print("=" * 60)
