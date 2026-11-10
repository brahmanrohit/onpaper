from pypdf import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


from .limits import MAX_PDF_PAGES, MAX_TEXT_CHARS
from .nltk_setup import ensure_nltk_data

ensure_nltk_data()

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF.

    Uploads are untrusted, so parsing stops after MAX_PDF_PAGES pages or
    MAX_TEXT_CHARS characters: a crafted or enormous PDF cannot tie up the server.
    """
    try:
        parts, size = [], 0
        for i, page in enumerate(PdfReader(pdf_file).pages):
            if i >= MAX_PDF_PAGES or size >= MAX_TEXT_CHARS:
                break
            page_text = page.extract_text()
            if page_text:
                parts.append(page_text)
                size += len(page_text) + 1
        text = "\n".join(parts).strip()[:MAX_TEXT_CHARS]
        return text if text else "Error: No extractable text found."
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def summarize_text(text, num_sentences=3):
    """Summarizes the input text using TF-IDF sentence ranking."""
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return text  # Return original text if it's too short

    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    
        top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]  # Highest ranked sentences
        summary = [sentences[i] for i in sorted(top_sentence_indices)]  # Preserve order
    
        return " ".join(summary)
    except Exception as e:
        return f"Error summarizing text: {str(e)}"
