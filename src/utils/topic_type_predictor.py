import pickle
import os
from pathlib import Path
from .config import TOPIC_TYPE_MODEL_PATH, TOPIC_TYPE_VECTORIZER_PATH

MODEL_PATH = str(TOPIC_TYPE_MODEL_PATH)
VECTORIZER_PATH = str(TOPIC_TYPE_VECTORIZER_PATH)

class TopicTypePredictor:
    def __init__(self):
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
        except Exception as e:
            print(f"Error loading models from {ML_DIR}: {str(e)}")
            raise

    def predict(self, topic: str):
        X = self.vectorizer.transform([topic])
        pred = self.model.predict(X)[0]
        proba = max(self.model.predict_proba(X)[0]) if hasattr(self.model, 'predict_proba') else None
        return pred, proba

# Usage example:
# predictor = TopicTypePredictor()
# paper_type, confidence = predictor.predict("Machine learning in healthcare") 