#!/usr/bin/env python3
"""
CORE API Training Script for Today's Session
Collects fresh academic papers and trains ML models with real data
"""

import os
import sys
import json
import time
import requests
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class CoreAPITrainer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.core.ac.uk/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        # Create directories
        os.makedirs('processed_papers/raw_papers/core_papers', exist_ok=True)
        os.makedirs('processed_papers/training_data/core_training', exist_ok=True)
        os.makedirs('ML', exist_ok=True)
        
        # Paper type mapping
        self.paper_types = {
            "empirical": "Empirical Research Paper",
            "theoretical": "Theoretical Research Paper", 
            "review": "Review Paper",
            "comparative": "Comparative Research Paper",
            "case_study": "Case Study",
            "analytical": "Analytical Research Paper",
            "methodological": "Methodological Research Paper",
            "position": "Position Paper",
            "technical": "Technical Report",
            "interdisciplinary": "Interdisciplinary Research Paper"
        }
    
    def search_papers(self, query: str, limit: int = 50) -> List[Dict]:
        """Search papers using CORE API"""
        url = f"{self.base_url}/search/works"
        payload = {"q": query, "limit": limit}
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json().get('results', [])
        except Exception as e:
            print(f"Error searching papers for '{query}': {e}")
            return []
    
    def classify_paper_type(self, title: str, abstract: str) -> str:
        """Classify paper type based on content"""
        text = f"{title} {abstract}".lower()
        
        # Enhanced classification logic
        if any(word in text for word in ['review', 'literature', 'systematic review', 'meta-analysis']):
            return 'review'
        elif any(word in text for word in ['theory', 'theoretical', 'framework', 'conceptual']):
            return 'theoretical'
        elif any(word in text for word in ['case study', 'case analysis', 'specific case']):
            return 'case_study'
        elif any(word in text for word in ['comparison', 'comparative', 'versus', 'compared to']):
            return 'comparative'
        elif any(word in text for word in ['analysis', 'analytical', 'critical analysis']):
            return 'analytical'
        elif any(word in text for word in ['methodology', 'method', 'procedure', 'approach']):
            return 'methodological'
        elif any(word in text for word in ['position', 'opinion', 'viewpoint', 'stance']):
            return 'position'
        elif any(word in text for word in ['technical', 'implementation', 'system', 'architecture']):
            return 'technical'
        elif any(word in text for word in ['interdisciplinary', 'cross-disciplinary', 'multi-disciplinary']):
            return 'interdisciplinary'
        else:
            return 'empirical'  # Default
    
    def process_paper(self, paper_data: Dict) -> Dict:
        """Process and format paper data for training"""
        title = paper_data.get('title', '') or ''
        abstract = paper_data.get('abstract', '') or ''
        keywords = paper_data.get('keywords', []) or []
        
        # Skip papers without essential data
        if not title or not abstract or len(abstract.strip()) < 30:
            return None
        
        # Classify paper type
        paper_type = self.classify_paper_type(title, abstract)
        
        # Classify field
        text = f"{title} {abstract}".lower()
        field = 'computer_science'  # Default
        if any(word in text for word in ['medical', 'health', 'clinical', 'patient', 'disease']):
            field = 'medicine'
        elif any(word in text for word in ['social', 'behavior', 'psychology', 'sociology']):
            field = 'social_sciences'
        elif any(word in text for word in ['business', 'management', 'economics', 'finance']):
            field = 'business'
        
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
    
    def collect_training_data(self) -> List[Dict]:
        """Collect comprehensive training data"""
        print("🔍 Collecting training data from CORE API...")
        
        # Enhanced topic list for better coverage
        topics = [
            # Machine Learning & AI
            "machine learning applications",
            "deep learning neural networks",
            "artificial intelligence systems",
            "natural language processing",
            "computer vision applications",
            "reinforcement learning",
            
            # Healthcare & Medicine
            "machine learning healthcare",
            "medical image analysis",
            "clinical decision support",
            "healthcare data analytics",
            "medical diagnosis AI",
            
            # Business & Finance
            "data science business",
            "financial machine learning",
            "business analytics",
            "predictive analytics",
            
            # Technology
            "blockchain technology",
            "cybersecurity machine learning",
            "internet of things",
            "cloud computing",
            "quantum computing",
            
            # Education
            "artificial intelligence education",
            "educational technology",
            "computer science education",
            "online learning systems",
            
            # Research Methods
            "systematic review methodology",
            "theoretical framework development",
            "case study research",
            "comparative analysis methods",
            "analytical research methods"
        ]
        
        all_papers = []
        
        for i, topic in enumerate(topics, 1):
            print(f"[{i}/{len(topics)}] Collecting papers for: {topic}")
            papers = self.search_papers(topic, limit=30)
            
            processed_count = 0
            for paper in papers:
                processed = self.process_paper(paper)
                if processed:
                    all_papers.append(processed)
                    processed_count += 1
            
            print(f"  ✅ Processed {processed_count} papers")
            time.sleep(0.2)  # Rate limiting
        
        print(f"\n📊 Total papers collected: {len(all_papers)}")
        return all_papers
    
    def save_training_data(self, papers: List[Dict]):
        """Save papers as training dataset"""
        training_data = {
            'papers': papers,
            'statistics': {
                'total_papers': len(papers),
                'paper_types': {},
                'fields': {},
                'collection_date': datetime.now().isoformat()
            }
        }
        
        # Calculate statistics
        for paper in papers:
            paper_type = paper['paper_type']
            field = paper['field']
            
            training_data['statistics']['paper_types'][paper_type] = \
                training_data['statistics']['paper_types'].get(paper_type, 0) + 1
            training_data['statistics']['fields'][field] = \
                training_data['statistics']['fields'].get(field, 0) + 1
        
        # Save to file
        filepath = 'processed_papers/training_data/core_training/core_dataset.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Training data saved: {filepath}")
        print(f"📈 Statistics:")
        print(f"  Total papers: {len(papers)}")
        print(f"  Paper types: {training_data['statistics']['paper_types']}")
        print(f"  Fields: {training_data['statistics']['fields']}")
        
        return training_data
    
    def train_paper_type_classifier(self, papers: List[Dict]):
        """Train paper type classification model"""
        print("\n🤖 Training Paper Type Classification Model...")
        
        # Prepare data
        texts = []
        labels = []
        
        for paper in papers:
            text = f"{paper['title']} {paper['abstract']}"
            paper_type = paper['paper_type']
            
            if text.strip() and paper_type:
                texts.append(text)
                labels.append(paper_type)
        
        if len(texts) < 10:
            print("❌ Insufficient data for training")
            return None
        
        # Vectorize
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        
        X = vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(classifier, X, y, cv=2)
        
        print(f"✅ Training completed!")
        print(f"📊 Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Training samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        
        # Save model
        model_path = 'ML/enhanced_paper_type_classifier.pkl'
        vectorizer_path = 'ML/enhanced_paper_type_vectorizer.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print(f"💾 Model saved: {model_path}")
        print(f"💾 Vectorizer saved: {vectorizer_path}")
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_path': model_path,
            'vectorizer_path': vectorizer_path
        }
    
    def train_plagiarism_detector(self, papers: List[Dict]):
        """Train plagiarism detection model"""
        print("\n🔍 Training Plagiarism Detection Model...")
        
        # Create reference documents from abstracts
        reference_docs = []
        for paper in papers:
            if paper['abstract'] and len(paper['abstract']) > 50:
                reference_docs.append(paper['abstract'])
        
        if len(reference_docs) < 10:
            print("❌ Insufficient reference documents")
            return None
        
        # Create vectorizer for plagiarism detection
        vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        
        # Fit vectorizer on reference documents
        vectorizer.fit(reference_docs)
        
        # Save model
        model_data = {
            'vectorizer': vectorizer,
            'reference_documents': reference_docs,
            'training_date': datetime.now().isoformat()
        }
        
        model_path = 'ML/enhanced_plagiarism_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Plagiarism model trained!")
        print(f"📊 Results:")
        print(f"  Reference documents: {len(reference_docs)}")
        print(f"  Features: {vectorizer.get_feature_names_out().shape[0]}")
        print(f"💾 Model saved: {model_path}")
        
        return {
            'reference_docs': len(reference_docs),
            'features': vectorizer.get_feature_names_out().shape[0],
            'model_path': model_path
        }
    
    def run_full_training(self):
        """Run complete training pipeline"""
        print("🚀 Starting CORE API Training Session")
        print("=" * 60)
        
        # Step 1: Collect data
        papers = self.collect_training_data()
        
        if not papers:
            print("❌ No papers collected. Check API key and internet connection.")
            return None
        
        # Step 2: Save training data
        training_data = self.save_training_data(papers)
        
        # Step 3: Train paper type classifier
        paper_type_results = self.train_paper_type_classifier(papers)
        
        # Step 4: Train plagiarism detector
        plagiarism_results = self.train_plagiarism_detector(papers)
        
        # Step 5: Generate training report
        report = {
            'training_date': datetime.now().isoformat(),
            'total_papers': len(papers),
            'paper_type_results': paper_type_results,
            'plagiarism_results': plagiarism_results,
            'statistics': training_data['statistics']
        }
        
        # Save report
        report_path = 'ML/training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Training report saved: {report_path}")
        print("\n🎉 Training session completed successfully!")
        
        return report

def main():
    """Main function"""
    print("🔑 CORE API Training Setup")
    print("=" * 40)
    
    # Get API key
    api_key = input("Enter your CORE API key: ").strip()
    
    if not api_key:
        print("❌ API key is required!")
        return
    
    # Test API key
    print("Testing API key...")
    try:
        response = requests.post(
            "https://api.core.ac.uk/v3/search/works",
            headers={'Authorization': f'Bearer {api_key}'},
            json={"q": "machine learning", "limit": 1}
        )
        
        if response.status_code != 200:
            print(f"❌ API key test failed: {response.status_code}")
            return
        else:
            print("✅ API key is valid!")
    except Exception as e:
        print(f"❌ API test error: {e}")
        return
    
    # Start training
    trainer = CoreAPITrainer(api_key)
    results = trainer.run_full_training()
    
    if results:
        print("\n📊 FINAL RESULTS:")
        print(f"Total papers collected: {results['total_papers']}")
        if results['paper_type_results']:
            print(f"Paper type accuracy: {results['paper_type_results']['accuracy']:.4f}")
        if results['plagiarism_results']:
            print(f"Plagiarism model: {results['plagiarism_results']['reference_docs']} reference docs")
        print("\n✅ Training completed successfully!")

if __name__ == "__main__":
    main() 