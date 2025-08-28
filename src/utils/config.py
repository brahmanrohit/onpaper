"""
Configuration settings for the Research Paper Assistant application.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent.parent

# ML Models directory
ML_DIR = BASE_DIR / 'src' / 'ML'

# Model file paths
PLAGIARISM_MODEL_PATH = ML_DIR / 'enhanced_models_plagiarism.pkl'
PLAGIARISM_VECTORIZER_PATH = ML_DIR / 'enhanced_models_vectorizer.pkl'
PAPER_TYPE_MODEL_PATH = ML_DIR / 'enhanced_models_paper_type.pkl'
PAPER_TYPE_VECTORIZER_PATH = ML_DIR / 'enhanced_models_vectorizer.pkl'
TOPIC_TYPE_MODEL_PATH = ML_DIR / 'topic_type_classifier.pkl'
TOPIC_TYPE_VECTORIZER_PATH = ML_DIR / 'topic_type_vectorizer.pkl'

# Application settings
DEFAULT_PLAGIARISM_THRESHOLD = 0.7
DEFAULT_SUMMARY_SENTENCES = 3
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Paper types
PAPER_TYPES = {
    "empirical": "Empirical Research Paper (Data-based)",
    "theoretical": "Theoretical Research Paper", 
    "review": "Review Paper (Literature or Systematic Review)",
    "comparative": "Comparative Research Paper",
    "case_study": "Case Study",
    "analytical": "Analytical Research Paper",
    "methodological": "Methodological Research Paper",
    "position": "Position Paper / Opinion Paper",
    "technical": "Technical Report",
    "interdisciplinary": "Interdisciplinary Research Paper"
}

# Citation styles
CITATION_STYLES = ["APA", "IEEE", "MLA"]

# Grammar checking uses Gemini AI API (GOOGLE_API_KEY)

# File upload settings
ALLOWED_FILE_TYPES = ["pdf", "txt"]
