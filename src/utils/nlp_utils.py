import nltk
from .gemini_helper import generate_text  # Use centralized AI helper

# Download the NLTK 'punkt' tokenizer only if it isn't already cached.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def generate_content(topic):
    """Generates research paper content using Gemini AI."""
    prompt = f"""
    Generate a well-structured research paper on the topic: {topic}.
    The paper should include:
    - Abstract
    - Introduction
    - Literature Review
    - Methodology
    - Results & Discussion
    - Conclusion
    - References
    Ensure the content is coherent, formal, and well-researched.
    """
    return generate_text(prompt)

def check_grammar(text):
    """Enhances grammar and clarity using Gemini AI."""
    prompt = f"Improve the grammar, clarity, and coherence of the following text:\n{text}"
    return generate_text(prompt)