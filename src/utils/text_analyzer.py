import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
import joblib
import os
import re
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize
import warnings
warnings.filterwarnings('ignore')

from .config import PLAGIARISM_MODEL_PATH, PLAGIARISM_VECTORIZER_PATH

# Ensure required NLTK data (punkt_tab/punkt/stopwords) is present.
from .nltk_setup import ensure_nltk_data
ensure_nltk_data()

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
        # Cache of (ref_id, sentence) pairs, built once and reused across
        # check_plagiarism() calls. Invalidated whenever the reference set changes.
        self._ref_sentences_cache = None
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
        self._ref_sentences_cache = None  # reference set changed
        self._update_model()
    
    def _update_model(self):
        """Update the TF-IDF model with current reference documents."""
        if not self.reference_documents:
            return
        # Extract text per item so a MIX of dict refs (user uploads) and plain
        # string refs (the bundled corpus) is handled — checking only [0] broke
        # when the bundled strings were combined with an added dict.
        all_texts = [(doc['text'] if isinstance(doc, dict) else str(doc))
                     for doc in self.reference_documents]
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
        # Keep the ORIGINAL sentences alongside the preprocessed ones so matches
        # can be reported back to the user in their real wording (not lowercased,
        # punctuation-stripped form).
        try:
            raw_sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip()) > 10]
        except Exception:
            raw_sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
        input_sentences = [self.preprocess_text(s) for s in raw_sentences]

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
        if self.vectorizer is None:
            return {
                'plagiarism_score': 0.0,
                'is_plagiarized': False,
                'similar_sentences': [],
                'message': 'Plagiarism model is not ready (no vectorizer available).'
            }

        # --- Document-level similarity (single batched transform) ---
        ref_texts = [
            (ref_doc.get('text', '') if isinstance(ref_doc, dict) else str(ref_doc))
            for ref_doc in self.reference_documents
        ]
        input_vector = self.vectorizer.transform([processed_text])
        ref_matrix = self.vectorizer.transform(ref_texts)
        doc_similarities = cosine_similarity(input_vector, ref_matrix)[0]
        max_similarity = float(doc_similarities.max()) if doc_similarities.size else 0.0
        plagiarism_score = max_similarity * 100

        # --- Sentence-level similarity (single batched matrix comparison) ---
        # Previously this did O(input_sentences x reference_sentences) individual
        # vectorizer.transform() calls, which made real papers take 10s+. Now we
        # vectorize all sentences once and compute one cosine-similarity matrix.
        similar_sentences = self._find_similar_sentences(list(zip(raw_sentences, input_sentences)), threshold)

        return {
            'plagiarism_score': round(plagiarism_score, 2),
            'is_plagiarized': plagiarism_score > (threshold * 100),
            'similar_sentences': similar_sentences[:5],  # Top 5 most similar sentences
            'message': self._generate_plagiarism_message(plagiarism_score, threshold)
        }

    def _get_reference_sentences(self) -> List[tuple]:
        """Return cached (ref_id, sentence) pairs for all reference documents.

        The reference sentences are tokenized only once and reused across calls,
        instead of re-running sentence tokenization on every check.
        """
        if self._ref_sentences_cache is None:
            cache = []
            for idx, ref_doc in enumerate(self.reference_documents):
                if isinstance(ref_doc, dict):
                    ref_id = ref_doc.get('id', f"doc_{idx}")
                    sentences = ref_doc.get('sentences')
                    if sentences is None:
                        sentences = self.extract_sentences(ref_doc.get('text', ''))
                else:
                    ref_id = f"doc_{idx}"
                    sentences = self.extract_sentences(str(ref_doc))
                for sent in sentences:
                    if len(sent) > 20:  # Only keep substantial sentences
                        cache.append((ref_id, sent))
            self._ref_sentences_cache = cache
        return self._ref_sentences_cache

    def _find_similar_sentences(self, input_pairs, threshold: float) -> List[Dict]:
        """Find input/reference sentence pairs above the similarity threshold.

        input_pairs is a list of (original_sentence, preprocessed_sentence). The
        preprocessed form is used for matching; the original is reported so the
        user sees their real wording.
        """
        ref_pairs = self._get_reference_sentences()
        pairs = [(orig, proc) for (orig, proc) in input_pairs if len(proc) > 20]
        if not ref_pairs or not pairs:
            return []

        input_origs = [orig for orig, _ in pairs]
        input_procs = [proc for _, proc in pairs]
        ref_ids = [rid for rid, _ in ref_pairs]
        ref_sent_texts = [s for _, s in ref_pairs]
        try:
            input_matrix = self.vectorizer.transform(input_procs)
            ref_matrix = self.vectorizer.transform(ref_sent_texts)
            sim_matrix = cosine_similarity(input_matrix, ref_matrix)
        except Exception:
            return []

        similar_sentences = []
        for i, row in enumerate(sim_matrix):
            for j in np.where(row > threshold)[0]:
                similar_sentences.append({
                    'input_sentence': input_origs[i],
                    'reference_sentence': ref_sent_texts[j],
                    'similarity': float(row[j]),
                    'reference_doc': ref_ids[j]
                })
        similar_sentences.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_sentences

    def _generate_plagiarism_message(self, score: float, threshold: float) -> str:
        """Human-readable message, consistent with is_plagiarized (score > threshold)."""
        # Check the user's threshold FIRST so the message never contradicts the
        # is_plagiarized flag or the UI band (which both use score > threshold*100).
        if score > threshold * 100:
            return "🚨 High plagiarism risk detected. Significant similarity with reference documents."
        if score < 30:
            return "✅ Low plagiarism risk. The text appears to be original."
        if score < 60:
            return "⚠️ Moderate similarity detected. Review for potential paraphrasing."
        return "⚠️ Notable similarity detected, but below your threshold. Consider reviewing."
    
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
                # Saved files use the 'model' key; some older/enhanced files use
                # 'classifier'. Accept either so the model isn't silently dropped.
                self.model = model_data.get('model', model_data.get('classifier', None))
                self.vectorizer = model_data.get('vectorizer', None)
                self.reference_documents = model_data.get('reference_documents', [])
            else:
                self.model = model_data
            self._ref_sentences_cache = None  # reference set (re)loaded
        else:
            print(f"Plagiarism model file not found: {self.model_path}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about the plagiarism detection system."""
        # Reference documents may be stored as dicts (with a 'sentences' list)
        # or as plain strings, depending on how the model file was created.
        total_sentences = 0
        for doc in self.reference_documents:
            if isinstance(doc, dict):
                total_sentences += len(doc.get('sentences', []))
            else:
                total_sentences += len(self.extract_sentences(str(doc)))
        return {
            'total_reference_documents': len(self.reference_documents),
            'total_reference_sentences': total_sentences,
            'model_trained': self.vectorizer is not None
        }

# Lazy global detector — built on first use so importing this module (and thus
# launching the app) doesn't pay the pickle-load cost up front. As a module
# global it loads once per process and is reused across Streamlit reruns
# (equivalent to @st.cache_resource, without coupling this module to the UI).
_plagiarism_detector = None


def get_plagiarism_detector() -> "PlagiarismDetector":
    """Return the shared PlagiarismDetector, constructing it on first use."""
    global _plagiarism_detector
    if _plagiarism_detector is None:
        det = PlagiarismDetector()  # loads the bundled corpus + vectorizer
        # First-ever run on a writable FS with no saved model: seed sample data.
        # On a deployed env the bundled pkl already exists, so this is a no-op.
        if not os.path.exists(det.model_path):
            _seed_sample_data(det)
        _plagiarism_detector = det
    return _plagiarism_detector


def __getattr__(name):
    """Backward-compat: ``text_analyzer.plagiarism_detector`` still resolves (lazily)."""
    if name == "plagiarism_detector":
        return get_plagiarism_detector()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def check_plagiarism(text: str, threshold: float = 0.7) -> Dict:
    """Main function to check plagiarism - maintains backward compatibility."""
    return get_plagiarism_detector().check_plagiarism(text, threshold)

def add_reference_document(text: str, doc_id: str = None):
    """Add a reference document to the plagiarism detection system."""
    det = get_plagiarism_detector()
    # Ensure doc_id is always a string
    if doc_id is None or doc_id.strip() == "":
        doc_id = f"doc_{len(det.reference_documents)}"
    det.add_reference_document(text, doc_id)

def get_plagiarism_statistics() -> Dict:
    """Get statistics about the plagiarism detection system."""
    return get_plagiarism_detector().get_statistics()

def build_custom_detector(user_docs, include_bundled: bool = True) -> "PlagiarismDetector":
    """Build a SESSION-SCOPED detector that compares against the user's own
    documents (optionally plus the bundled 93-paper corpus). Does NOT mutate the
    global singleton, so user uploads never pollute the shared corpus.

    user_docs: list of (doc_name, text). Returns a fresh PlagiarismDetector.
    """
    det = PlagiarismDetector()  # loads the bundled corpus + vectorizer
    if not include_bundled:
        det.reference_documents = []
        det._ref_sentences_cache = None
        det.vectorizer = None
    for name, text in (user_docs or []):
        if text and text.strip():
            det.add_reference_document(text, name or None)
    return det

def save_plagiarism_model():
    """Save the current plagiarism detection model."""
    get_plagiarism_detector().save_model()

# Seed a couple of sample reference documents (first-run only, when no saved
# model exists). Invoked by get_plagiarism_detector(); on a deployed env the
# bundled pkl is already present so this never runs.
def _seed_sample_data(detector: "PlagiarismDetector"):
    """Initialize a detector with some sample reference documents."""
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
        detector.add_reference_document(doc['text'], doc['id'])

    detector.save_model()