"""
Streamlit app for support bot UI.
This is the main entry point for Streamlit Cloud deployment.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add the support-bot src directory to path for imports
current_dir = Path(__file__).parent
support_bot_src = current_dir / 'lasttime' / 'lasttime' / 'support-bot' / 'src'
if str(support_bot_src) not in sys.path:
    sys.path.insert(0, str(support_bot_src))

try:
    from inference_api import classify_ticket_via_api, generate_response_via_api, get_model_name
except ImportError as e:
    error_msg = str(e)
    st.error(f"❌ Failed to import inference_api: {error_msg}")
    st.error("💡 This usually means 'huggingface-hub' package is not installed on the cloud server.")
    st.error("📋 Fix: The app is trying to install dependencies. If this persists:")
    st.error("   1. Check that requirements.txt includes 'huggingface-hub>=0.19.0'")
    st.error("   2. Redeploy the app on Streamlit Cloud (hard refresh)")
    st.error("   3. Check app logs in Streamlit Cloud dashboard")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Support Bot Demo",
    page_icon="🧾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 900px;
        margin: 0 auto;
    }
    .stTextArea > div > div > textarea {
        min-height: 150px;
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
        font-family: 'Courier New', monospace;
    }
    .category-box {
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        color: white;
    }
    .category-box p {
        color: white !important;
        font-size: 1.3rem !important;
    }
    .response-box {
        padding: 1.5rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border: none;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.3);
        color: white;
    }
    .response-box p {
        color: white !important;
        font-weight: 500;
        line-height: 1.7;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 2rem;
    }
    .title-section {
        text-align: center;
        margin-bottom: 2rem;
    }
    hr {
        margin: 1.5rem 0;
        opacity: 0.2;
    }
    </style>
""", unsafe_allow_html=True)


def is_valid_ticket_input(text):
    """
    Validate if the input is a meaningful ticket description.
    Checks for:
    - Minimum length (at least 15 characters)
    - Contains real English words (common dictionary words)
    - Not just random character gibberish
    - Has proper structure (spaces, punctuation, etc.)
    """
    import re
    
    if not text or len(text.strip()) < 15:
        return False, "Input too short. Please provide at least 15 characters with a complete sentence."
    
    # List of common English words that appear in support tickets
    common_words = {
        'i', 'my', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had',
        'do', 'does', 'did', 'can', 'could', 'would', 'should', 'will', 'want', 'need',
        'issue', 'problem', 'error', 'bug', 'crash', 'help', 'support', 'please', 'thanks',
        'account', 'login', 'password', 'email', 'charge', 'billing', 'payment', 'refund',
        'feature', 'request', 'app', 'system', 'page', 'button', 'not', 'work', 'working',
        'get', 'got', 'getting', 'getting', 'cannot', 'can\'t', 'don\'t', 'doesn\'t',
        'when', 'why', 'how', 'what', 'where', 'which', 'who', 'that', 'this', 'these',
        'with', 'from', 'to', 'for', 'in', 'on', 'at', 'by', 'about', 'as', 'it', 'me',
        'you', 'he', 'she', 'we', 'they', 'or', 'and', 'but', 'not', 'no', 'yes'
    }
    
    # Extract words from text
    words = re.findall(r'\b[a-z]+\b', text.lower())
    
    if not words:
        return False, "Input must contain actual words. Please describe your issue clearly."
    
    # Check if at least 20% of words are recognizable English words
    recognized_words = sum(1 for word in words if word in common_words)
    recognition_ratio = recognized_words / len(words) if words else 0
    
    if recognition_ratio < 0.2:
        return False, "Input appears to be random characters or gibberish. Please write a meaningful support ticket."
    
    # Check for space/punctuation (real sentences have them)
    if ' ' not in text:
        return False, "Input appears incomplete. Please write a complete sentence or description."
    
    # Check if it's mostly vowels or mostly consonants (gibberish indicator)
    vowels = len(re.findall(r'[aeiou]', text.lower()))
    consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', text.lower()))
    
    if consonants > 0 and vowels > 0:
        vowel_ratio = vowels / (vowels + consonants)
        if vowel_ratio < 0.15 or vowel_ratio > 0.70:
            return False, "Input appears to be random characters. Please describe your actual support issue."
    
    return True, "Valid"


def main():
    """Main Streamlit app function."""
    
    # Title
    st.title("🧾 Customer Support Ticket Classifier + Auto-Responder")
    st.markdown("---")
    
    # Description
    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-bottom: 2rem;'>
            <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'><strong>Automated ticket triaging and response drafting using Open LLMs.</strong></p>
            <p style='font-size: 0.95rem;'>Paste a customer support ticket below and click Generate to get a predicted category and draft response.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Text area for ticket input
    ticket_text = st.text_area(
        "Customer Support Ticket",
        placeholder="Enter or paste the customer support ticket here...",
        height=150,
        help="Paste the full ticket description from the customer"
    )
    
    # Generate button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        generate_button = st.button(
            "Generate",
            type="primary",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Process when button is clicked
    if generate_button:
        if not ticket_text or not ticket_text.strip():
            st.warning("⚠️ Please enter a ticket description before generating.")
        else:
            # Validate input
            is_valid, validation_message = is_valid_ticket_input(ticket_text)
            
            if not is_valid:
                st.error(f"❌ **Invalid Input:** {validation_message}")
                st.info("💡 **Tips for a valid ticket:**\n- Describe your issue or request clearly\n- Use at least 10 characters\n- Write meaningful sentences (not random characters)")
            else:
                # Classification
                with st.spinner("🔍 Classifying ticket category..."):
                    try:
                        category = classify_ticket_via_api(ticket_text)
                        # Check if it's an error message
                        if category.startswith("[Error]"):
                            st.error(f"❌ **Classification Error:** {category}")
                            st.warning("⚠️ Check your HF token permissions if inference fails.")
                            st.stop()
                        # Check if input is invalid (no domain keywords found)
                        if category == "invalid":
                            st.error("❌ **Invalid Input:** Please enter a valid support issue related to billing, technical issues, features, or account.")
                            st.info("💡 **Your input does not appear to be related to customer support.**\n\nPlease describe an issue or request related to:\n- **Billing**: payments, charges, invoices, refunds, subscriptions\n- **Technical Issues**: bugs, errors, crashes, login problems, system issues\n- **Feature Requests**: new features, enhancements, improvements, suggestions\n- **Account Issues**: account settings, profile, verification, access, cancellation")
                            st.stop()
                        st.session_state['category'] = category
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        st.error(f"❌ **Classification failed:** {str(e)}")
                        st.warning("⚠️ Check your HF token permissions if inference fails.")
                        with st.expander("Error Details (click to expand)"):
                            st.code(error_details)
                        st.stop()
                
                # Response generation
                with st.spinner("✍️ Generating draft response..."):
                    try:
                        response = generate_response_via_api(ticket_text, category)
                        # Check if it's an error message
                        if response.startswith("[Error]"):
                            st.error(f"❌ **Response Generation Error:** {response}")
                            st.warning("⚠️ Check your HF token permissions if inference fails.")
                            st.stop()
                        st.session_state['response'] = response
                    except Exception as e:
                        st.error(f"❌ **Response generation failed:** {str(e)}")
                        st.warning("⚠️ Check your HF token permissions if inference fails.")
                        st.stop()
                
                # Display results
                st.success("✅ Processing complete!")
                st.markdown("---")
                
                # Predicted Category - Clean labeled section
                st.markdown("### 📋 Predicted Category")
                st.markdown(
                    f"""
                    <div class="category-box">
                        <p style='margin: 0;'><strong>{category.upper()}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Draft Response - Clean labeled section
                st.markdown("### 💬 Draft Response")
                st.markdown(
                    f"""
                    <div class="response-box">
                        <p style='margin: 0; white-space: pre-wrap; line-height: 1.6;'>{response}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    # Display previous results if available
    elif 'category' in st.session_state and 'response' in st.session_state:
        st.info("💡 Click Generate to process a new ticket or view previous results below.")
        st.markdown("---")
        
        # Predicted Category - Clean labeled section
        st.markdown("### 📋 Predicted Category")
        st.markdown(
            f"""
            <div class="category-box">
                <p style='margin: 0;'><strong>{st.session_state['category'].upper()}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Draft Response - Clean labeled section
        st.markdown("### 💬 Draft Response")
        st.markdown(
            f"""
            <div class="response-box">
                <p style='margin: 0; white-space: pre-wrap; line-height: 1.6;'>{st.session_state['response']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #999; font-size: 0.8rem; margin-top: 2rem;'>
            Powered by Hugging Face Inference API
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
