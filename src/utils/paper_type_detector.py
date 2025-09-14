import re
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from pathlib import Path
from .gemini_helper import generate_text
from .config import PAPER_TYPE_MODEL_PATH, PAPER_TYPE_VECTORIZER_PATH

class ResearchPaperTypeDetector:
    """ML model to auto-detect research paper types and generate type-specific content."""
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        self.model = None
        self.vectorizer = None
        self.paper_types = {
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
        self.type_keywords = self._load_type_keywords()
        if model_path is None:
            model_path = str(PAPER_TYPE_MODEL_PATH)
        if vectorizer_path is None:
            vectorizer_path = str(PAPER_TYPE_VECTORIZER_PATH)
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.load_model()
        
    def _load_type_keywords(self) -> Dict[str, List[str]]:
        """Load type-specific keywords for detection."""
        return {
            "empirical": ["data analysis", "statistics", "sample size", "survey", "experiment", "quantitative", "statistical analysis"],
            "theoretical": ["theoretical framework", "conceptual model", "theoretical concepts", "framework development"],
            "review": ["literature review", "systematic review", "meta-analysis", "research gaps", "existing research"],
            "comparative": ["comparison", "contrast", "versus", "compared to", "similarities", "differences"],
            "case_study": ["case study", "specific case", "particular instance", "case analysis", "detailed examination"],
            "analytical": ["critical analysis", "analytical framework", "logical reasoning", "systematic analysis"],
            "methodological": ["methodology development", "method innovation", "procedural innovation", "method validation"],
            "position": ["position paper", "opinion paper", "stance", "viewpoint", "argument", "perspective"],
            "technical": ["technical report", "technical analysis", "technical implementation", "technical specifications"],
            "interdisciplinary": ["interdisciplinary", "cross-disciplinary", "multi-disciplinary", "interdisciplinary approach"]
        }
    
    def detect_paper_type(self, topic: str) -> Dict[str, Any]:
        """Detect the most appropriate paper type for a given topic."""
        # Fallback to keyword-based detection
        return self._keyword_based_detection(topic)
    
    def _keyword_based_detection(self, topic: str) -> Dict[str, Any]:
        """Keyword-based detection when ML model is not available."""
        topic_lower = topic.lower()
        
        # Count keyword matches for each type
        type_scores = {}
        for paper_type, keywords in self.type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in topic_lower)
            type_scores[paper_type] = score
        
        # Get the type with highest score
        detected_type = max(type_scores, key=type_scores.get)
        max_score = type_scores[detected_type]
        
        # Calculate confidence based on score
        confidence = min(max_score / 5, 0.95)  # Cap at 95%
        
        # Get top 3 predictions
        top_predictions = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "detected_type": detected_type,
            "confidence": confidence,
            "paper_type_name": self.paper_types[detected_type],
            "top_predictions": top_predictions,
            "reasoning": f"Keyword-based detection: {max_score} keyword matches found for {detected_type}",
            "method": "keyword_based"
        }
    
    def get_type_specific_guidance(self, paper_type: str) -> Dict[str, Any]:
        """Get type-specific guidance for content generation."""
        guidance = {
            "empirical": {
                "focus": "Data analysis, statistics, sample size, empirical findings",
                "key_sections": ["Methodology", "Results", "Discussion"],
                "content_style": "Data-driven with statistical analysis",
                "special_requirements": ["Sample size", "Statistical tests", "Data tables"]
            },
            "theoretical": {
                "focus": "Theoretical concepts, frameworks, conceptual analysis",
                "key_sections": ["Theoretical Framework", "Analysis", "Discussion"],
                "content_style": "Conceptual with framework development",
                "special_requirements": ["Conceptual models", "Theoretical assumptions", "Framework justification"]
            },
            "review": {
                "focus": "Past studies, research gaps, future directions",
                "key_sections": ["Literature Review", "Analysis", "Discussion"],
                "content_style": "Synthetic with gap identification",
                "special_requirements": ["Research gaps", "Future directions", "Literature synthesis"]
            },
            "comparative": {
                "focus": "Comparison between subjects, contrast analysis",
                "key_sections": ["Comparative Analysis", "Discussion"],
                "content_style": "Comparative with systematic contrast",
                "special_requirements": ["Comparison framework", "Similarities/differences", "Contrast analysis"]
            },
            "case_study": {
                "focus": "Deep dive on specific subject/event, context",
                "key_sections": ["Case Background", "Case Analysis", "Discussion"],
                "content_style": "Narrative with rich context",
                "special_requirements": ["Case context", "Unique factors", "Detailed examination"]
            },
            "analytical": {
                "focus": "Critical analysis, logical reasoning",
                "key_sections": ["Analytical Framework", "Analysis", "Discussion"],
                "content_style": "Critical with logical reasoning",
                "special_requirements": ["Analytical framework", "Critical examination", "Logical reasoning"]
            },
            "methodological": {
                "focus": "Method development, validation, innovation",
                "key_sections": ["Methodology Development", "Validation", "Discussion"],
                "content_style": "Procedural with innovation focus",
                "special_requirements": ["Method development", "Validation procedures", "Innovation details"]
            },
            "position": {
                "focus": "Clear position, supporting arguments",
                "key_sections": ["Position Statement", "Supporting Arguments", "Discussion"],
                "content_style": "Persuasive with clear stance",
                "special_requirements": ["Clear position", "Supporting evidence", "Counter-arguments"]
            },
            "technical": {
                "focus": "Technical details, implementation, specifications",
                "key_sections": ["Technical Background", "Technical Analysis", "Results"],
                "content_style": "Technical with implementation focus",
                "special_requirements": ["Technical specifications", "Implementation details", "Technical analysis"]
            },
            "interdisciplinary": {
                "focus": "Multiple disciplines, integration, cross-disciplinary insights",
                "key_sections": ["Theoretical Integration", "Cross-Disciplinary Analysis", "Discussion"],
                "content_style": "Integrative with multi-disciplinary approach",
                "special_requirements": ["Multi-disciplinary concepts", "Integration framework", "Cross-disciplinary insights"]
            }
        }
        
        return guidance.get(paper_type, guidance["empirical"])

    def load_model(self):
        """Load the trained model from disk."""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as vf:
                    self.vectorizer = pickle.load(vf)
                return True
            except Exception as e:
                print(f"Error loading enhanced paper type model: {e}")
                return False
        else:
            print(f"Enhanced paper type model or vectorizer not found: {self.model_path}, {self.vectorizer_path}")
            return False

# Global instance
paper_type_detector = ResearchPaperTypeDetector()

def auto_detect_paper_type(topic: str) -> Dict[str, Any]:
    """Auto-detect the most appropriate paper type for a given topic."""
    return paper_type_detector.detect_paper_type(topic)

def get_type_guidance(paper_type: str) -> Dict[str, Any]:
    """Get type-specific guidance for content generation."""
    return paper_type_detector.get_type_specific_guidance(paper_type)

def train_paper_type_model() -> None:
    """Train the paper type detection model."""
    paper_type_detector.train_model()

def load_paper_type_model() -> bool:
    """Load the trained paper type detection model."""
    return paper_type_detector.load_model() 