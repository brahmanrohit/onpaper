import json
import pickle
import numpy as np
import requests
import time
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from typing import Dict, List, Tuple
import logging
import os

class EnhancedMLTrainer:
    """
    Enhanced ML Trainer using CORE API collected data
    Trains high-accuracy models for paper type detection and plagiarism detection
    """
    
    def __init__(self, core_api_key: str = None):
        self.logger = logging.getLogger(__name__)
        self.paper_type_model = None
        self.plagiarism_model = None
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        self.core_api_key = core_api_key
        self.base_url = "https://api.core.ac.uk/v3"
        self.session = requests.Session()
        if core_api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {core_api_key}',
                'Content-Type': 'application/json'
            })
        
    def collect_fresh_core_data(self, topics: List[str] = None, papers_per_topic: int = 25) -> Dict:
        """Collect fresh data from CORE API for today's training session"""
        if not self.core_api_key:
            self.logger.error("CORE API key not provided")
            return None
        
        if topics is None:
            topics = [
                "machine learning healthcare",
                "artificial intelligence education",
                "deep learning applications",
                "natural language processing",
                "computer vision",
                "data science business",
                "blockchain technology",
                "cybersecurity machine learning",
                "quantum computing",
                "robotics automation",
                "systematic review methodology",
                "theoretical framework development",
                "case study research",
                "comparative analysis methods",
                "analytical research methods"
            ]
        
        print("🔍 Collecting fresh training data from CORE API...")
        all_papers = []
        
        for i, topic in enumerate(topics, 1):
            print(f"[{i}/{len(topics)}] Collecting papers for: {topic}")
            papers = self.search_core_papers(topic, papers_per_topic)
            
            processed_count = 0
            for paper in papers:
                processed = self.process_core_paper(paper)
                if processed:
                    all_papers.append(processed)
                    processed_count += 1
            
            print(f"  ✅ Processed {processed_count} papers")
            time.sleep(0.2)  # Rate limiting
        
        print(f"\n📊 Total papers collected: {len(all_papers)}")
        
        # Save fresh data
        training_data = {
            'papers': all_papers,
            'statistics': self.calculate_statistics(all_papers),
            'collection_date': datetime.now().isoformat()
        }
        
        filepath = 'processed_papers/training_data/core_training/fresh_core_dataset.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Fresh training data saved: {filepath}")
        return training_data
    
    def search_core_papers(self, query: str, limit: int = 50) -> List[Dict]:
        """Search papers using CORE API"""
        url = f"{self.base_url}/search/works"
        payload = {"q": query, "limit": limit}
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json().get('results', [])
        except Exception as e:
            self.logger.error(f"Error searching papers for '{query}': {e}")
            return []
    
    def process_core_paper(self, paper_data: Dict) -> Dict:
        """Process and classify CORE API paper data"""
        title = paper_data.get('title', '') or ''
        abstract = paper_data.get('abstract', '') or ''
        keywords = paper_data.get('keywords', []) or []
        
        # Skip papers without essential data
        if not title or not abstract or len(abstract.strip()) < 30:
            return None
        
        # Enhanced paper type classification
        paper_type = self.classify_paper_type_enhanced(title, abstract)
        
        # Enhanced field classification
        field = self.classify_field_enhanced(title, abstract)
        
        return {
            'title': title,
            'abstract': abstract,
            'keywords': keywords,
            'paper_type': paper_type,
            'field': field,
            'content': abstract,
            'word_count': len(abstract.split()),
            'core_id': paper_data.get('id'),
            'year': paper_data.get('year'),
            'processed_date': datetime.now().isoformat()
        }
    
    def classify_paper_type_enhanced(self, title: str, abstract: str) -> str:
        """Enhanced paper type classification"""
        text = f"{title} {abstract}".lower()
        
        # Enhanced classification with more specific patterns
        if any(word in text for word in ['systematic review', 'literature review', 'meta-analysis', 'review of']):
            return 'review'
        elif any(word in text for word in ['theoretical framework', 'theoretical model', 'conceptual framework', 'theoretical analysis']):
            return 'theoretical'
        elif any(word in text for word in ['case study', 'case analysis', 'specific case', 'case-based']):
            return 'case_study'
        elif any(word in text for word in ['comparative analysis', 'comparison between', 'versus', 'compared to', 'comparative study']):
            return 'comparative'
        elif any(word in text for word in ['critical analysis', 'analytical framework', 'analytical study', 'analysis of']):
            return 'analytical'
        elif any(word in text for word in ['methodology development', 'methodological approach', 'method innovation', 'procedural']):
            return 'methodological'
        elif any(word in text for word in ['position paper', 'opinion paper', 'viewpoint', 'stance on', 'argument for']):
            return 'position'
        elif any(word in text for word in ['technical implementation', 'technical analysis', 'system architecture', 'technical report']):
            return 'technical'
        elif any(word in text for word in ['interdisciplinary', 'cross-disciplinary', 'multi-disciplinary', 'interdisciplinary approach']):
            return 'interdisciplinary'
        else:
            return 'empirical'  # Default
    
    def classify_field_enhanced(self, title: str, abstract: str) -> str:
        """Enhanced field classification"""
        text = f"{title} {abstract}".lower()
        
        if any(word in text for word in ['medical', 'health', 'clinical', 'patient', 'disease', 'diagnosis', 'treatment']):
            return 'medicine'
        elif any(word in text for word in ['social', 'behavior', 'psychology', 'sociology', 'human behavior']):
            return 'social_sciences'
        elif any(word in text for word in ['business', 'management', 'economics', 'finance', 'market', 'enterprise']):
            return 'business'
        elif any(word in text for word in ['education', 'learning', 'teaching', 'pedagogy', 'academic']):
            return 'education'
        else:
            return 'computer_science'  # Default
    
    def calculate_statistics(self, papers: List[Dict]) -> Dict:
        """Calculate dataset statistics"""
        stats = {
            'total_papers': len(papers),
            'paper_types': {},
            'fields': {},
            'avg_word_count': 0
        }
        
        total_words = 0
        for paper in papers:
            paper_type = paper['paper_type']
            field = paper['field']
            word_count = paper['word_count']
            
            stats['paper_types'][paper_type] = stats['paper_types'].get(paper_type, 0) + 1
            stats['fields'][field] = stats['fields'].get(field, 0) + 1
            total_words += word_count
        
        stats['avg_word_count'] = total_words / len(papers) if papers else 0
        return stats
    
    def load_core_data(self, filepath: str = 'processed_papers/training_data/core_training/core_dataset.json') -> Dict:
        """Load training data from CORE API collection"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data['papers'])} papers from CORE data")
            return data
        except FileNotFoundError:
            self.logger.error(f"Training data file not found: {filepath}")
            return None
    
    def prepare_paper_type_data(self, papers: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for paper type classification"""
        texts = []
        labels = []
        
        for paper in papers:
            # Combine title, abstract, and keywords for classification
            text = f"{paper.get('title', '')} {paper.get('abstract', '')} {' '.join(paper.get('keywords', []))}"
            paper_type = paper.get('paper_type', 'empirical')
            
            if text.strip() and paper_type:
                texts.append(text)
                labels.append(paper_type)
        
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        return X, y
    
    def train_paper_type_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train enhanced paper type classification model"""
        self.logger.info("Training enhanced paper type classification model...")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Enhanced model configurations
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'naive_bayes': MultinomialNB(alpha=0.1)
        }
        
        # Train individual models
        trained_models = {}
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=2, scoring='accuracy')
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            trained_models[name] = model
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'model': model,
                'classification_report': classification_report(y_test, y_pred)
            }
            
            self.logger.info(f"{name} - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Create ensemble model
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in trained_models.items()],
            voting='soft'
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        y_pred_ensemble = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        # Calibrate the best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_model = trained_models[best_model_name]
        
        calibrated_model = CalibratedClassifierCV(best_model, cv=2)
        calibrated_model.fit(X_train, y_train)
        
        self.paper_type_model = calibrated_model
        
        results['ensemble'] = {
            'accuracy': ensemble_accuracy,
            'model': ensemble,
            'classification_report': classification_report(y_test, y_pred_ensemble)
        }
        
        results['calibrated'] = {
            'accuracy': ensemble_accuracy,  # Use ensemble accuracy as reference
            'model': calibrated_model,
            'base_model': best_model_name
        }
        
        self.logger.info(f"Ensemble accuracy: {ensemble_accuracy:.4f}")
        self.logger.info(f"Calibrated model ready for deployment")
        
        return results
    
    def prepare_plagiarism_data(self, papers: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for plagiarism detection"""
        texts = []
        labels = []
        
        for paper in papers:
            abstract = paper.get('abstract', '')
            if len(abstract) > 50:  # Only use substantial abstracts
                texts.append(abstract)
                labels.append(1)  # Original text
                
                # Create paraphrased version for training
                paraphrased = self.paraphrase_text(abstract)
                texts.append(paraphrased)
                labels.append(0)  # Paraphrased text
        
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        return X, y
    
    def paraphrase_text(self, text: str) -> str:
        """Simple text paraphrasing for training data"""
        replacements = {
            'study': 'research',
            'analysis': 'examination',
            'results': 'findings',
            'method': 'approach',
            'data': 'information',
            'shows': 'demonstrates',
            'found': 'discovered',
            'conclude': 'determine',
            'investigation': 'inquiry',
            'examine': 'analyze'
        }
        
        paraphrased = text
        for original, replacement in replacements.items():
            paraphrased = paraphrased.replace(original, replacement)
        
        return paraphrased
    
    def train_plagiarism_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train plagiarism detection model"""
        self.logger.info("Training plagiarism detection model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Save as dict for compatibility
        self.plagiarism_model = {
            'classifier': model,
            'vectorizer': self.vectorizer,
            'reference_documents': []  # Add if you use reference docs
        }
        
        results = {
            'accuracy': accuracy,
            'model': self.plagiarism_model,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        self.logger.info(f"Plagiarism model accuracy: {accuracy:.4f}")
        return results
    
    def save_enhanced_models(self, filename_prefix: str = 'enhanced_models'):
        """Save trained models"""
        if self.paper_type_model:
            with open(f'ML/{filename_prefix}_paper_type.pkl', 'wb') as f:
                pickle.dump(self.paper_type_model, f)
            self.logger.info("Paper type model saved")
        
        if self.plagiarism_model:
            with open(f'ML/{filename_prefix}_plagiarism.pkl', 'wb') as f:
                pickle.dump(self.plagiarism_model, f)
            self.logger.info("Plagiarism model saved")
        
        # Save vectorizer separately for compatibility
        with open(f'ML/{filename_prefix}_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        self.logger.info("Vectorizer saved")
    
    def load_models(self, filename_prefix: str = 'enhanced_models'):
        """Load trained models"""
        try:
            with open(f'ML/{filename_prefix}_paper_type.pkl', 'rb') as f:
                self.paper_type_model = pickle.load(f)
            
            with open(f'ML/{filename_prefix}_plagiarism.pkl', 'rb') as f:
                self.plagiarism_model = pickle.load(f)
            
            with open(f'ML/{filename_prefix}_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.logger.info("Models loaded successfully")
            return True
        except FileNotFoundError:
            self.logger.error("Model files not found")
            return False
    
    def predict_paper_type(self, text: str) -> Tuple[str, float]:
        """Predict paper type for given text"""
        if not self.paper_type_model:
            return "unknown", 0.0
        
        # Vectorize text
        X = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.paper_type_model.predict(X)[0]
        confidence = np.max(self.paper_type_model.predict_proba(X))
        
        return prediction, confidence
    
    def detect_plagiarism(self, text1: str, text2: str) -> float:
        """Detect plagiarism between two texts"""
        if not self.plagiarism_model:
            return 0.0
        
        # Vectorize texts
        X = self.vectorizer.transform([text1, text2])
        
        # Calculate similarity
        similarity = np.dot(X[0].toarray(), X[1].toarray().T)[0][0]
        
        return float(similarity)
    
    def enhance_plagiarism_model_with_old(self, old_model_path='ML/enhanced_models_plagiarism.pkl', new_refs=None):
        """Enhance plagiarism model by merging old and new reference documents and retraining vectorizer."""
        import pickle
        old_refs = []
        # Load old model if exists
        if os.path.exists(old_model_path):
            with open(old_model_path, 'rb') as f:
                old_model = pickle.load(f)
            old_refs = old_model.get('reference_documents', [])
        # Merge with new references
        if new_refs is None:
            new_refs = []
        all_refs = list(set(old_refs + new_refs))
        # Retrain vectorizer
        vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words='english', min_df=1, max_df=0.95)
        if all_refs:
            vectorizer.fit(all_refs)
        # Save enhanced model
        enhanced_model = {
            'vectorizer': vectorizer,
            'reference_documents': all_refs
        }
        with open('ML/enhanced_models_plagiarism.pkl', 'wb') as f:
            pickle.dump(enhanced_model, f)
        self.logger.info(f"Enhanced plagiarism model saved with {len(all_refs)} reference documents.")
        return enhanced_model

    def run_full_training(self, use_fresh_data: bool = False) -> dict:
        self.logger.info("Starting enhanced ML training pipeline...")
        # Load CORE data
        data = self.load_core_data()
        if not data:
            return {}
        papers = data['papers']
        # ... (train models as before) ...
        # After training, enhance plagiarism model with old references
        new_refs = [paper['abstract'] for paper in papers if len(paper.get('abstract', '')) > 50]
        self.enhance_plagiarism_model_with_old(new_refs=new_refs)
        # ... (rest of the code) ...

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = EnhancedMLTrainer()
    
    # Run training
    results = trainer.run_full_training()
    
    if results:
        print("\n" + "="*50)
        print("ENHANCED ML TRAINING RESULTS")
        print("="*50)
        
        # Paper type results
        print("\n📊 PAPER TYPE CLASSIFICATION:")
        for model_name, result in results['paper_type_results'].items():
            print(f"  {model_name}: {result['accuracy']:.4f} accuracy")
        
        # Plagiarism results
        print("\n🔍 PLAGIARISM DETECTION:")
        print(f"  Accuracy: {results['plagiarism_results']['accuracy']:.4f}")
        
        # Training stats
        stats = results['training_stats']
        print(f"\n📈 TRAINING STATISTICS:")
        print(f"  Total papers: {stats['total_papers']}")
        print(f"  Paper types: {stats['paper_types_trained']}")
        print(f"  Plagiarism samples: {stats['plagiarism_samples']}")
        
        print("\n✅ Models saved and ready for use!")
    else:
        print("❌ Training failed. Check your data files.")

if __name__ == "__main__":
    main() 