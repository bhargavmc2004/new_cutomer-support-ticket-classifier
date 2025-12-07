"""
Prompt Engineering Experiments
Tests different prompt templates to compare response variations.
"""

import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# Load environment variables
project_root = Path(__file__).parent.parent
env_path = project_root / '.env'
load_dotenv(env_path)

# Get API token and model from environment
HF_TOKEN = os.getenv('HF_TOKEN')
HF_GEN_MODEL = os.getenv('HF_GEN_MODEL', 'tiiuae/falcon-7b-instruct')

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found in .env file. Please create a .env file with HF_TOKEN=your_token"
    )

# Initialize InferenceClient
client = InferenceClient(token=HF_TOKEN)


def load_policy():
    """Load company policy from file."""
    policy_path = project_root / 'data' / 'policy.txt'
    if policy_path.exists():
        try:
            with open(policy_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Warning: Could not load policy file: {e}")
    return ""


def create_formal_prompt(ticket_text, category, policy):
    """Create a formal, professional prompt template."""
    return f"""You are a professional customer support representative. Write a formal, business-appropriate response to the customer's inquiry.

Ticket Category: {category}
Customer Ticket: {ticket_text}
{f'Company Policy: {policy}' if policy else ''}

Requirements:
- Use formal, professional language
- Maintain a respectful, business-appropriate tone
- Be clear and direct
- Write 3-5 sentences
- End with a specific next step

Response:"""


def create_empathetic_prompt(ticket_text, category, policy):
    """Create an empathetic, understanding prompt template."""
    return f"""You are a compassionate customer support agent. Write a warm, empathetic response that shows genuine understanding of the customer's situation.

Ticket Category: {category}
Customer Ticket: {ticket_text}
{f'Company Policy: {policy}' if policy else ''}

Requirements:
- Express genuine empathy and understanding
- Acknowledge the customer's feelings
- Use warm, caring language
- Be supportive and reassuring
- Write 3-5 sentences
- End with a helpful next step

Response:"""


def create_concise_policy_prompt(ticket_text, category, policy):
    """Create a concise prompt that heavily references policy."""
    return f"""You are a customer support agent. Write a brief, policy-focused response.

Ticket Category: {category}
Customer Ticket: {ticket_text}
Company Policy: {policy if policy else 'Standard support procedures apply.'}

Requirements:
- Be concise and to the point (2-4 sentences)
- Reference specific policy points when relevant
- Follow policy guidelines strictly
- Be direct and clear
- End with one actionable step based on policy

Response:"""


def create_friendly_prompt(ticket_text, category, policy):
    """Create a friendly, conversational prompt template."""
    return f"""You are a friendly customer support agent. Write a casual, approachable response that feels like talking to a helpful friend.

Ticket Category: {category}
Customer Ticket: {ticket_text}
{f'Company Policy: {policy}' if policy else ''}

Requirements:
- Use friendly, conversational language
- Be approachable and personable
- Use a warm, casual tone
- Make the customer feel comfortable
- Write 3-5 sentences
- End with a friendly next step

Response:"""


def generate_response_with_prompt(prompt, tone_name):
    """Generate a response using a custom prompt."""
    try:
        response = client.text_generation(
            prompt=prompt,
            model=HF_GEN_MODEL,
            max_new_tokens=200,
            temperature=0.7,
            return_full_text=False
        )
        
        # Clean the response
        response = response.strip()
        
        # Ensure proper sentence structure
        if response:
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if sentences:
                response = '. '.join(sentences)
                if not response.endswith('.'):
                    response += '.'
        
        print(f"✓ Generated response for {tone_name} tone")
        return response
        
    except Exception as e:
        print(f"✗ Error generating response for {tone_name} tone: {e}")
        return f"ERROR: {str(e)}"


def main():
    """Run prompt engineering experiments."""
    print("="*70)
    print("Prompt Engineering Experiments")
    print("="*70)
    
    # Test ticket
    test_ticket = "I was charged twice this month"
    test_category = "billing"
    
    # Load policy
    policy = load_policy()
    
    print(f"\nTest Ticket: {test_ticket}")
    print(f"Category: {test_category}")
    print(f"\nGenerating responses with different prompt templates...")
    print("-" * 70)
    
    # Define prompt templates
    prompt_templates = [
        ("Formal", create_formal_prompt),
        ("Empathetic", create_empathetic_prompt),
        ("Concise + Policy-Referenced", create_concise_policy_prompt),
        ("Friendly", create_friendly_prompt),
    ]
    
    results = []
    
    # Generate responses for each template
    for tone_name, prompt_func in prompt_templates:
        print(f"\n[{tone_name} Tone]")
        print("-" * 70)
        
        prompt = prompt_func(test_ticket, test_category, policy)
        response = generate_response_with_prompt(prompt, tone_name)
        
        # Calculate response metrics
        word_count = len(response.split())
        sentence_count = len([s for s in response.split('.') if s.strip()])
        
        results.append({
            'tone': tone_name,
            'ticket_text': test_ticket,
            'category': test_category,
            'response': response,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'prompt_template': prompt_func.__name__
        })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    output_file = project_root / 'data' / 'prompt_experiment_results.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    
    print(f"\n{'Tone':<25} {'Words':<8} {'Sentences':<10} {'Response Preview':<30}")
    print("-" * 70)
    
    for _, row in df_results.iterrows():
        preview = row['response'][:50] + "..." if len(row['response']) > 50 else row['response']
        print(f"{row['tone']:<25} {row['word_count']:<8} {row['sentence_count']:<10} {preview:<30}")
    
    print("\n" + "="*70)
    print("DETAILED RESPONSES")
    print("="*70)
    
    for _, row in df_results.iterrows():
        print(f"\n[{row['tone']} Tone]")
        print("-" * 70)
        print(f"Words: {row['word_count']} | Sentences: {row['sentence_count']}")
        print(f"\nResponse:\n{row['response']}")
        print()
    
    print("="*70)
    print("[SUCCESS] Prompt experiments complete!")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    print(f"Total templates tested: {len(results)}")
    print("="*70)


if __name__ == '__main__':
    main()

