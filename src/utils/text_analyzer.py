import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import joblib
import os
from pathlib import Path
import re
from typing import List, Dict, Tuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

from .config import PLAGIARISM_MODEL_PATH, PLAGIARISM_VECTORIZER_PATH

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Download data at module import
download_nltk_data()

class PlagiarismDetector:
    """ML-based plagiarism detection system using TF-IDF and cosine similarity."""
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        if model_path is None:
            model_path = str(PLAGIARISM_MODEL_PATH)
        if vectorizer_path is None:
            vectorizer_path = str(PLAGIARISM_VECTORIZER_PATH)
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.vectorizer = None
        self.reference_documents = []
        self.model = None
        self.load_model()
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract and clean sentences from text."""
        try:
            sentences = sent_tokenize(text)
            return [self.preprocess_text(sent) for sent in sentences if len(sent.strip()) > 10]
        except Exception as e:
            # Fallback to simple sentence splitting if NLTK fails
            print(f"Warning: NLTK sentence tokenization failed, using fallback: {e}")
            sentences = text.split('.')
            return [self.preprocess_text(sent) for sent in sentences if len(sent.strip()) > 10]
    
    def add_reference_document(self, text: str, doc_id: str = None):
        """Add a reference document to the plagiarism detection database."""
        processed_text = self.preprocess_text(text)
        if doc_id is None or doc_id.strip() == "":
            doc_id = f"doc_{len(self.reference_documents)}"
        
        self.reference_documents.append({
            'id': doc_id,
            'text': processed_text,
            'sentences': self.extract_sentences(text)
        })
        self._update_model()
    
    def _update_model(self):
        """Update the TF-IDF model with current reference documents."""
        if not self.reference_documents:
            return
        # Support both list of dicts and list of strings
        if isinstance(self.reference_documents[0], dict):
            all_texts = [doc['text'] for doc in self.reference_documents]
        else:
            all_texts = self.reference_documents
        # Fit the vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=1.0
        )
        self.vectorizer.fit(all_texts)
        
        # Create the model pipeline
        self.model = Pipeline([
            ('vectorizer', self.vectorizer)
        ])
    
    def check_plagiarism(self, text: str, threshold: float = 0.7) -> Dict:
        """Check for plagiarism in the given text."""
        if not self.reference_documents:
            return {
                'plagiarism_score': 0.0,
                'is_plagiarized': False,
                'similar_sentences': [],
                'message': 'No reference documents available for comparison.'
            }
        
        processed_text = self.preprocess_text(text)
        input_sentences = self.extract_sentences(text)
        
        if not input_sentences:
            return {
                'plagiarism_score': 0.0,
                'is_plagiarized': False,
                'similar_sentences': [],
                'message': 'No valid sentences found in the input text.'
            }
        
        # Ensure vectorizer is available (in case model was loaded without fitting in this session)
        if self.vectorizer is None:
            self._update_model()
        
        # Vectorize input text
        input_vector = self.vectorizer.transform([processed_text])
        
        # Calculate similarities with all reference documents
        similarities = []
        similar_sentences = []
        
        for idx, ref_doc in enumerate(self.reference_documents):
            # Support both dict-style and plain string reference documents
            if isinstance(ref_doc, dict):
                ref_text = ref_doc.get('text', '')
                ref_id = ref_doc.get('id', f"doc_{idx}")
                ref_sentences = ref_doc.get('sentences', self.extract_sentences(ref_text))
            else:
                ref_text = str(ref_doc)
                ref_id = f"doc_{idx}"
                ref_sentences = self.extract_sentences(ref_text)

            ref_vector = self.vectorizer.transform([ref_text])
            similarity = cosine_similarity(input_vector, ref_vector)[0][0]
            similarities.append(similarity)
            
            # Check sentence-level similarities
            for i, input_sent in enumerate(input_sentences):
                for j, ref_sent in enumerate(ref_sentences):
                    if len(input_sent) > 20 and len(ref_sent) > 20:  # Only compare substantial sentences
                        sent_similarity = self._calculate_sentence_similarity(input_sent, ref_sent)
                        if sent_similarity > threshold:
                            similar_sentences.append({
                                'input_sentence': input_sent,
                                'reference_sentence': ref_sent,
                                'similarity': sent_similarity,
                                'reference_doc': ref_id
                            })
        
        # Calculate overall plagiarism score
        max_similarity = max(similarities) if similarities else 0.0
        plagiarism_score = max_similarity * 100
        
        # Sort similar sentences by similarity score
        similar_sentences.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'plagiarism_score': round(plagiarism_score, 2),
            'is_plagiarized': plagiarism_score > (threshold * 100),
            'similar_sentences': similar_sentences[:5],  # Top 5 most similar sentences
            'message': self._generate_plagiarism_message(plagiarism_score, threshold)
        }
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate similarity between two sentences using TF-IDF."""
        try:
            vectors = self.vectorizer.transform([sent1, sent2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return similarity
        except:
            return 0.0
    
    def _generate_plagiarism_message(self, score: float, threshold: float) -> str:
        """Generate a human-readable message based on plagiarism score."""
        if score < 30:
            return "✅ Low plagiarism risk. The text appears to be original."
        elif score < 60:
            return "⚠️ Moderate similarity detected. Review for potential paraphrasing."
        elif score < threshold * 100:
            return "⚠️ High similarity detected. Consider revising the content."
        else:
            return "🚨 High plagiarism risk detected. Significant similarity with reference documents."
    
    def save_model(self):
        """Save the trained model to disk."""
        if self.model:
            model_data = {
                'model': self.model,
                'reference_documents': self.reference_documents,
                'vectorizer': self.vectorizer
            }
            joblib.dump(model_data, self.model_path)
    
    def load_model(self):
        """Load the trained model from disk."""
        import pickle
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            if isinstance(model_data, dict):
                self.model = model_data.get('classifier', None)
                self.vectorizer = model_data.get('vectorizer', None)
                self.reference_documents = model_data.get('reference_documents', [])
            else:
                self.model = model_data
        else:
            print(f"Plagiarism model file not found: {self.model_path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the plagiarism detection system."""
        return {
            'total_reference_documents': len(self.reference_documents),
            'total_reference_sentences': sum(len(doc['sentences']) for doc in self.reference_documents),
            'model_trained': self.model is not None
        }

# Initialize global detector instance
plagiarism_detector = PlagiarismDetector()

def check_plagiarism(text: str, threshold: float = 0.7) -> Dict:
    """Main function to check plagiarism - maintains backward compatibility."""
    return plagiarism_detector.check_plagiarism(text, threshold)

def add_reference_document(text: str, doc_id: str = None):
    """Add a reference document to the plagiarism detection system."""
    # Ensure doc_id is always a string
    if doc_id is None or doc_id.strip() == "":
        doc_id = f"doc_{len(plagiarism_detector.reference_documents)}"
    plagiarism_detector.add_reference_document(text, doc_id)

def get_plagiarism_statistics() -> Dict:
    """Get statistics about the plagiarism detection system."""
    return plagiarism_detector.get_statistics()

def save_plagiarism_model():
    """Save the current plagiarism detection model."""
    plagiarism_detector.save_model()

# Add some sample reference documents for testing
def initialize_sample_data():
    """Initialize the system with some sample reference documents."""
    sample_docs = [
        {
            'id': 'sample_ai_paper',
            'text': """
            Artificial Intelligence (AI) has revolutionized various industries including healthcare, 
            finance, and transportation. Machine learning algorithms can now process vast amounts 
            of data to identify patterns and make predictions with remarkable accuracy. Deep learning 
            models, particularly neural networks, have shown exceptional performance in image 
            recognition and natural language processing tasks.
            """
        },
        {
            'id': 'sample_climate_paper',
            'text': """
            Climate change represents one of the most pressing challenges of our time. Rising global 
            temperatures, melting polar ice caps, and increasing sea levels threaten ecosystems 
            worldwide. Scientists have documented significant changes in weather patterns, with 
            more frequent extreme weather events occurring across the globe.
            """
        }
    ]
    
    for doc in sample_docs:
        plagiarism_detector.add_reference_document(doc['text'], doc['id'])
    
    plagiarism_detector.save_model()

# Initialize with sample data if no model exists
if not os.path.exists(plagiarism_detector.model_path):
    initialize_sample_data()