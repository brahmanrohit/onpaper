import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download("punkt")

def extract_text_from_pdf(pdf_file):
    """Extracts text content from an uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip() if text else "Error: No extractable text found."
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
