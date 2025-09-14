"""
Enhanced ML Models for Research Paper Analysis
Improved versions with 85% accuracy target and low bias/medium variance
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from typing import Dict, List, Tuple, Any
import pickle
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class EnhancedTextPreprocessor:
    """Enhanced text preprocessing with academic-specific handling"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Academic terms normalization
        self.academic_terms = {
            'machine learning': 'ML',
            'artificial intelligence': 'AI',
            'deep learning': 'DL',
            'neural networks': 'NN',
            'natural language processing': 'NLP',
            'computer vision': 'CV',
            'reinforcement learning': 'RL'
        }
        
        # Common academic phrases (don't indicate plagiarism)
        self.common_phrases = [
            'this study', 'research shows', 'according to',
            'in conclusion', 'furthermore', 'however',
            'the results indicate', 'previous studies',
            'literature review', 'methodology'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced preprocessing with academic context awareness"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove citations and references
        text = re.sub(r'\([^)]*\)', '', text)  # Parenthetical citations
        text = re.sub(r'\[\d+\]', '', text)    # Numbered citations
        text = re.sub(r'[A-Za-z]+ et al\.', '', text)  # Author citations
        
        # Normalize academic terms
        for term, abbreviation in self.academic_terms.items():
            text = text.replace(term, abbreviation)
        
        # Remove common academic phrases
        for phrase in self.common_phrases:
            text = text.replace(phrase, '')
        
        # Tokenize and clean
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalnum() and token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def calculate_academic_complexity(self, text: str) -> float:
        """Calculate academic complexity score"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Vocabulary diversity
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        # Technical term density
        technical_terms = sum(1 for word in words if word in self.academic_terms.values())
        technical_density = technical_terms / len(words) if words else 0
        
        # Normalized complexity score
        complexity = (avg_sentence_length * 0.3 + 
                     vocabulary_diversity * 0.4 + 
                     technical_density * 0.3)
        
        return min(complexity, 1.0)

class EnhancedPlagiarismDetector:
    """Enhanced plagiarism detection with multi-level similarity analysis"""
    
    def __init__(self):
        self.preprocessor = EnhancedTextPreprocessor()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=1.0,
            stop_words='english'
        )
        self.reference_docs = []
        self.base_threshold = 0.7
        
    def add_reference_documents(self, documents: List[str]):
        """Add reference documents for comparison"""
        self.reference_docs = [self.preprocessor.preprocess_text(doc) for doc in documents]
        if self.reference_docs:
            self.tfidf_vectorizer.fit(self.reference_docs)
    
    def calculate_enhanced_similarity(self, text1: str, text2: str) -> float:
        """Multi-level similarity detection"""
        # Preprocess texts
        processed_text1 = self.preprocessor.preprocess_text(text1)
        processed_text2 = self.preprocessor.preprocess_text(text2)
        
        if not processed_text1 or not processed_text2:
            return 0.0
        
        # Level 1: TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.transform([processed_text1, processed_text2])
            tfidf_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            tfidf_similarity = 0.0
        
        # Level 2: Structural similarity (sentence patterns)
        structural_similarity = self.calculate_structural_similarity(text1, text2)
        
        # Level 3: Keyword overlap
        keyword_similarity = self.calculate_keyword_similarity(processed_text1, processed_text2)
        
        # Weighted combination
        final_similarity = (
            0.5 * tfidf_similarity + 
            0.3 * structural_similarity + 
            0.2 * keyword_similarity
        )
        
        return min(max(final_similarity, 0.0), 1.0)
    
    def calculate_structural_similarity(self, text1: str, text2: str) -> float:
        """Calculate structural similarity based on sentence patterns"""
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        
        if not sentences1 or not sentences2:
            return 0.0
        
        # Compare sentence structures
        structure_matches = 0
        total_comparisons = min(len(sentences1), len(sentences2))
        
        for i in range(total_comparisons):
            # Compare sentence length patterns
            len1 = len(word_tokenize(sentences1[i]))
            len2 = len(word_tokenize(sentences2[i]))
            
            if abs(len1 - len2) / max(len1, len2) < 0.3:  # Similar length
                structure_matches += 1
        
        return structure_matches / total_comparisons if total_comparisons > 0 else 0.0
    
    def calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate keyword overlap similarity"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def calculate_adaptive_threshold(self, text: str) -> float:
        """Dynamic threshold based on text characteristics"""
        complexity = self.preprocessor.calculate_academic_complexity(text)
        text_length = len(text.split())
        
        # Adjust threshold based on factors
        if text_length < 100:
            threshold = self.base_threshold - 0.1  # More lenient for short texts
        elif complexity > 0.8:
            threshold = self.base_threshold + 0.05  # Stricter for complex academic text
        else:
            threshold = self.base_threshold
        
        return min(max(threshold, 0.5), 0.9)
    
    def detect_plagiarism(self, text: str) -> Dict[str, Any]:
        """Enhanced plagiarism detection with adaptive thresholds"""
        if not self.reference_docs:
            return {
                'plagiarism_score': 0.0,
                'category': 'No reference documents available',
                'confidence': 0.0,
                'similar_documents': []
            }
        
        similarities = []
        for i, ref_doc in enumerate(self.reference_docs):
            similarity = self.calculate_enhanced_similarity(text, ref_doc)
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Calculate adaptive threshold
        adaptive_threshold = self.calculate_adaptive_threshold(text)
        
        # Get highest similarity
        max_similarity = similarities[0][0] if similarities else 0.0
        
        # Determine category
        if max_similarity >= adaptive_threshold + 0.1:
            category = 'High Plagiarism'
        elif max_similarity >= adaptive_threshold:
            category = 'Medium Plagiarism'
        elif max_similarity >= adaptive_threshold - 0.2:
            category = 'Low Plagiarism'
        else:
            category = 'No Plagiarism'
        
        # Calculate confidence based on similarity distribution
        confidence = min(max_similarity * 1.2, 0.95)
        
        # Get similar documents
        similar_docs = [(sim, idx) for sim, idx in similarities if sim > adaptive_threshold - 0.3]
        
        return {
            'plagiarism_score': max_similarity,
            'category': category,
            'confidence': confidence,
            'similar_documents': similar_docs[:5],
            'threshold_used': adaptive_threshold
        }

class EnhancedPaperTypeDetector:
    """Enhanced paper type detection with ensemble approach"""
    
    def __init__(self):
        self.preprocessor = EnhancedTextPreprocessor()
        self.scaler = StandardScaler()
        
        # Enhanced keyword sets for each paper type
        self.type_keywords = {
            'empirical': [
                'experiment', 'study', 'participants', 'data collection', 'results',
                'statistical analysis', 'hypothesis', 'sample size', 'survey',
                'interview', 'observation', 'measurement', 'correlation'
            ],
            'theoretical': [
                'theory', 'framework', 'conceptual', 'model', 'proposition',
                'hypothesis', 'theoretical foundation', 'conceptual framework',
                'theoretical analysis', 'theoretical implications'
            ],
            'review': [
                'literature review', 'systematic review', 'meta-analysis',
                'previous studies', 'existing research', 'overview',
                'comprehensive review', 'state of the art'
            ],
            'comparative': [
                'comparison', 'compare', 'versus', 'contrast', 'different',
                'similarities', 'differences', 'analysis of', 'evaluation of'
            ],
            'case_study': [
                'case study', 'case analysis', 'specific case', 'real-world',
                'practical example', 'implementation', 'case-based'
            ],
            'analytical': [
                'analysis', 'analytical', 'critical analysis', 'examination',
                'investigation', 'evaluation', 'assessment', 'detailed analysis'
            ],
            'methodological': [
                'methodology', 'method', 'approach', 'technique', 'procedure',
                'methodological framework', 'research design', 'methodological analysis'
            ],
            'position': [
                'position', 'argument', 'perspective', 'viewpoint', 'stance',
                'proposal', 'recommendation', 'advocacy', 'position paper'
            ],
            'technical': [
                'technical', 'implementation', 'algorithm', 'system design',
                'technical specification', 'technical analysis', 'technical solution'
            ],
            'interdisciplinary': [
                'interdisciplinary', 'multidisciplinary', 'cross-disciplinary',
                'integration', 'collaboration', 'multiple fields', 'diverse approaches'
            ]
        }
        
        # Initialize ensemble models
        self.models = {
            'keyword_based': self._create_keyword_classifier(),
            'ml_based': self._create_ml_classifier(),
            'rule_based': self._create_rule_classifier()
        }
        
        self.weights = {'keyword_based': 0.3, 'ml_based': 0.5, 'rule_based': 0.2}
    
    def _create_keyword_classifier(self):
        """Create keyword-based classifier"""
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _create_ml_classifier(self):
        """Create ML-based classifier"""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    
    def _create_rule_classifier(self):
        """Create rule-based classifier"""
        return RandomForestClassifier(n_estimators=50, random_state=42)
    
    def extract_advanced_features(self, topic: str) -> Dict[str, float]:
        """Extract advanced features for classification"""
        features = {}
        
        # Keyword density features
        for paper_type, keywords in self.type_keywords.items():
            feature_name = f"{paper_type}_keyword_density"
            features[feature_name] = self.calculate_keyword_density(topic, keywords)
        
        # Linguistic features
        features['academic_complexity'] = self.preprocessor.calculate_academic_complexity(topic)
        features['topic_specificity'] = self.calculate_topic_specificity(topic)
        features['methodology_indicators'] = self.count_methodology_indicators(topic)
        features['analysis_indicators'] = self.count_analysis_indicators(topic)
        
        # Contextual features
        features['field_indicator'] = self.detect_academic_field(topic)
        features['research_approach'] = self.detect_research_approach(topic)
        
        return features
    
    def calculate_keyword_density(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword density in text"""
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
        
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
        return keyword_count / len(words)
    
    def calculate_topic_specificity(self, text: str) -> float:
        """Calculate topic specificity score"""
        # Count domain-specific terms
        domain_terms = ['algorithm', 'methodology', 'framework', 'model', 'theory']
        text_lower = text.lower()
        
        specificity_score = sum(1 for term in domain_terms if term in text_lower)
        return min(specificity_score / 5.0, 1.0)
    
    def count_methodology_indicators(self, text: str) -> float:
        """Count methodology-related indicators"""
        methodology_terms = ['method', 'approach', 'technique', 'procedure', 'design']
        text_lower = text.lower()
        
        count = sum(1 for term in methodology_terms if term in text_lower)
        return min(count / 5.0, 1.0)
    
    def count_analysis_indicators(self, text: str) -> float:
        """Count analysis-related indicators"""
        analysis_terms = ['analysis', 'examination', 'investigation', 'evaluation', 'assessment']
        text_lower = text.lower()
        
        count = sum(1 for term in analysis_terms if term in text_lower)
        return min(count / 5.0, 1.0)
    
    def detect_academic_field(self, text: str) -> float:
        """Detect academic field indicator"""
        fields = {
            'computer_science': ['algorithm', 'programming', 'software', 'data'],
            'medicine': ['clinical', 'patient', 'treatment', 'medical'],
            'social_sciences': ['social', 'behavior', 'society', 'human'],
            'engineering': ['engineering', 'technical', 'system', 'design']
        }
        
        text_lower = text.lower()
        max_score = 0.0
        
        for field, terms in fields.items():
            score = sum(1 for term in terms if term in text_lower) / len(terms)
            max_score = max(max_score, score)
        
        return max_score
    
    def detect_research_approach(self, text: str) -> float:
        """Detect research approach indicator"""
        approaches = {
            'quantitative': ['statistical', 'numerical', 'data', 'measurement'],
            'qualitative': ['interview', 'observation', 'narrative', 'descriptive'],
            'mixed': ['both', 'combination', 'integrated', 'mixed methods']
        }
        
        text_lower = text.lower()
        max_score = 0.0
        
        for approach, terms in approaches.items():
            score = sum(1 for term in terms if term in text_lower) / len(terms)
            max_score = max(max_score, score)
        
        return max_score
    
    def calculate_calibrated_confidence(self, features: Dict[str, float], 
                                      prediction: str) -> float:
        """Calculate calibrated confidence score"""
        # Base confidence from feature strength
        feature_strength = features.get(f"{prediction}_keyword_density", 0.0)
        
        # Ambiguity penalty (if multiple types have similar scores)
        keyword_scores = [features.get(f"{pt}_keyword_density", 0.0) 
                         for pt in self.type_keywords.keys()]
        max_score = max(keyword_scores)
        second_max = sorted(keyword_scores)[-2] if len(keyword_scores) > 1 else 0
        
        ambiguity_penalty = (second_max / max_score) if max_score > 0 else 0
        
        # Domain expertise
        domain_expertise = features.get('field_indicator', 0.0)
        
        # Calibrated confidence
        calibrated_confidence = (
            feature_strength * 0.6 +
            (1 - ambiguity_penalty) * 0.2 +
            domain_expertise * 0.2
        )
        
        return min(max(calibrated_confidence, 0.1), 0.95)
    
    def detect_paper_type(self, topic: str) -> Dict[str, Any]:
        """Enhanced paper type detection with ensemble approach"""
        if not topic or not isinstance(topic, str):
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'features': {}
            }
        
        # Extract features
        features = self.extract_advanced_features(topic)
        
        # Get predictions from each model
        predictions = {}
        confidences = {}
        
        # Keyword-based prediction
        keyword_scores = {pt: features.get(f"{pt}_keyword_density", 0.0) 
                         for pt in self.type_keywords.keys()}
        best_keyword_type = max(keyword_scores, key=keyword_scores.get)
        predictions['keyword_based'] = best_keyword_type
        confidences['keyword_based'] = keyword_scores[best_keyword_type]
        
        # ML-based prediction (simplified for now)
        ml_prediction = best_keyword_type  # Placeholder
        predictions['ml_based'] = ml_prediction
        confidences['ml_based'] = 0.8  # Placeholder confidence
        
        # Rule-based prediction
        rule_prediction = best_keyword_type  # Placeholder
        predictions['rule_based'] = rule_prediction
        confidences['rule_based'] = 0.7  # Placeholder confidence
        
        # Weighted ensemble prediction
        weighted_scores = {}
        for paper_type in self.type_keywords.keys():
            score = 0.0
            for model_name, weight in self.weights.items():
                if predictions[model_name] == paper_type:
                    score += weight * confidences[model_name]
            weighted_scores[paper_type] = score
        
        final_prediction = max(weighted_scores, key=weighted_scores.get)
        final_confidence = self.calculate_calibrated_confidence(features, final_prediction)
        
        return {
            'type': final_prediction,
            'confidence': final_confidence,
            'features': features,
            'ensemble_scores': weighted_scores
        }

def test_enhanced_models():
    """Test the enhanced models"""
    print("🧪 Testing Enhanced ML Models...")
    
    # Test plagiarism detector
    print("\n📝 Testing Enhanced Plagiarism Detector:")
    plagiarism_detector = EnhancedPlagiarismDetector()
    
    # Sample reference documents
    reference_docs = [
        "Machine learning algorithms have shown remarkable performance in various applications.",
        "The study conducted experiments with 100 participants to evaluate the effectiveness.",
        "Previous research has demonstrated the importance of data preprocessing in ML.",
        "This paper presents a novel approach to natural language processing."
    ]
    
    plagiarism_detector.add_reference_documents(reference_docs)
    
    # Test cases
    test_cases = [
        "Machine learning algorithms have shown remarkable performance in various applications.",  # High plagiarism
        "AI systems demonstrate excellent results across different domains.",  # Medium plagiarism
        "The weather today is sunny and warm.",  # No plagiarism
        "This research investigates the impact of data quality on model performance."  # Low plagiarism
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        result = plagiarism_detector.detect_plagiarism(test_text)
        print(f"Test {i}: {result['category']} (Score: {result['plagiarism_score']:.3f}, Confidence: {result['confidence']:.3f})")
    
    # Test paper type detector
    print("\n📄 Testing Enhanced Paper Type Detector:")
    type_detector = EnhancedPaperTypeDetector()
    
    test_topics = [
        "Experimental study on machine learning algorithms with 200 participants",
        "Theoretical framework for understanding neural networks",
        "Systematic review of deep learning applications in healthcare",
        "Comparative analysis of different optimization techniques"
    ]
    
    for i, topic in enumerate(test_topics, 1):
        result = type_detector.detect_paper_type(topic)
        print(f"Test {i}: {result['type']} (Confidence: {result['confidence']:.3f})")
    
    print("\n✅ Enhanced model testing completed!")

if __name__ == "__main__":
    test_enhanced_models() 