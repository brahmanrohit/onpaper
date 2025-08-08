import requests
import json
import time
import os
from typing import List, Dict
from datetime import datetime

class CoreAPICollector:
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
    
    def search_papers(self, query: str, limit: int = 50) -> List[Dict]:
        """Search papers using CORE API"""
        url = f"{self.base_url}/search/works"
        payload = {"q": query, "limit": limit}
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json().get('results', [])
        except Exception as e:
            print(f"Error searching papers: {e}")
            return []
    
    def process_paper(self, paper_data: Dict) -> Dict:
        """Process and format paper data for training"""
        title = paper_data.get('title', '') or ''
        abstract = paper_data.get('abstract', '') or ''
        keywords = paper_data.get('keywords', []) or []
        
        # Skip papers without essential data
        if not title or not abstract or len(abstract.strip()) < 50:
            return None
        
        # Classify paper type
        text = f"{title} {abstract}".lower()
        paper_type = 'empirical'  # Default
        if 'review' in text or 'literature' in text:
            paper_type = 'review'
        elif 'theory' in text or 'theoretical' in text:
            paper_type = 'theoretical'
        elif 'case study' in text:
            paper_type = 'case_study'
        
        # Classify field
        field = 'computer_science'  # Default
        if any(word in text for word in ['medical', 'health', 'clinical']):
            field = 'medicine'
        elif any(word in text for word in ['social', 'behavior', 'psychology']):
            field = 'social_sciences'
        
        return {
            'title': title,
            'abstract': abstract,
            'keywords': keywords,
            'paper_type': paper_type,
            'field': field,
            'content': abstract,  # Use abstract as content for now
            'word_count': len(abstract.split()),
            'core_id': paper_data.get('id'),
            'year': paper_data.get('year'),
            'processed_date': datetime.now().isoformat()
        }
    
    def collect_papers(self, topics: List[str], papers_per_topic: int = 20) -> List[Dict]:
        """Collect papers for multiple topics"""
        all_papers = []
        
        for topic in topics:
            print(f"Collecting papers for: {topic}")
            papers = self.search_papers(topic, papers_per_topic)
            
            for i, paper in enumerate(papers):
                processed = self.process_paper(paper)
                if processed:  # Only add if processing was successful
                    all_papers.append(processed)
                    print(f"  Processed: {processed['title'][:50]}...")
                
                time.sleep(0.1)  # Rate limiting
        
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
        
        print(f"Training data saved: {filepath}")
        print(f"Total papers: {len(papers)}")
        print(f"Paper types: {training_data['statistics']['paper_types']}")
        print(f"Fields: {training_data['statistics']['fields']}")

def main():
    # Get your CORE API key from: https://core.ac.uk/services/api/
    API_KEY = input("Enter your CORE API key: ").strip()
    
    if not API_KEY:
        print("Please provide a valid CORE API key")
        return
    
    collector = CoreAPICollector(API_KEY)
    
    # Topics to collect papers for
    topics = [
        "machine learning healthcare",
        "artificial intelligence education", 
        "data science business",
        "computer vision applications",
        "natural language processing",
        "robotics automation",
        "cybersecurity threats",
        "blockchain technology",
        "internet of things",
        "quantum computing"
    ]
    
    print("Starting paper collection...")
    papers = collector.collect_papers(topics, papers_per_topic=15)
    
    if papers:
        collector.save_training_data(papers)
        print("✅ Paper collection completed successfully!")
    else:
        print("❌ No papers collected. Check your API key and internet connection.")

if __name__ == "__main__":
    main() 