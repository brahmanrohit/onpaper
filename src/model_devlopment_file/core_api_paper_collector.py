import requests
import json
import time
import os
from typing import List, Dict, Any
import re
from datetime import datetime
import logging

class CoreAPIPaperCollector:
    """
    Automated paper collector using CORE API
    Fetches 250M+ research papers and prepares them for ML training
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://api.core.ac.uk/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}' if api_key else '',
            'Content-Type': 'application/json'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories for paper storage"""
        directories = [
            'processed_papers/raw_papers/core_papers',
            'processed_papers/processed_papers/core_processed',
            'processed_papers/training_data/core_training',
            'processed_papers/validation_data/core_validation',
            'processed_papers/test_data/core_test'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def search_papers(self, query: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Search papers using CORE API
        """
        url = f"{self.base_url}/search/works"
        
        payload = {
            "q": query,
            "limit": limit,
            "offset": offset,
            "scroll": True
        }
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json().get('results', [])
        except Exception as e:
            self.logger.error(f"Error searching papers: {e}")
            return []
    
    def get_paper_details(self, paper_id: str) -> Dict:
        """
        Get detailed information about a specific paper
        """
        url = f"{self.base_url}/works/{paper_id}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting paper details: {e}")
            return {}
    
    def extract_paper_content(self, paper_data: Dict) -> Dict:
        """
        Extract and format paper content for training
        """
        try:
            # Extract basic information
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            keywords = paper_data.get('keywords', [])
            authors = [author.get('name', '') for author in paper_data.get('authors', [])]
            
            # Extract full text if available
            full_text = paper_data.get('fullText', '')
            
            # Determine paper type based on content analysis
            paper_type = self.classify_paper_type(title, abstract, full_text)
            
            # Determine academic field
            field = self.classify_academic_field(title, abstract, keywords)
            
            # Extract sections if full text is available
            sections = self.extract_sections(full_text) if full_text else {}
            
            return {
                'title': title,
                'abstract': abstract,
                'keywords': keywords,
                'authors': authors,
                'paper_type': paper_type,
                'field': field,
                'sections': sections,
                'full_text': full_text,
                'word_count': len(full_text.split()) if full_text else len(abstract.split()),
                'core_id': paper_data.get('id'),
                'doi': paper_data.get('doi'),
                'year': paper_data.get('year'),
                'download_url': paper_data.get('downloadUrl'),
                'processed_date': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error extracting paper content: {e}")
            return {}
    
    def classify_paper_type(self, title: str, abstract: str, full_text: str) -> str:
        """
        Classify paper type based on content analysis
        """
        text = f"{title} {abstract} {full_text}".lower()
        
        # Keywords for different paper types
        type_keywords = {
            'empirical': ['study', 'experiment', 'survey', 'analysis', 'data', 'results', 'findings'],
            'theoretical': ['theory', 'theoretical', 'framework', 'model', 'conceptual'],
            'review': ['review', 'literature', 'systematic', 'meta-analysis', 'overview'],
            'comparative': ['comparison', 'compare', 'versus', 'vs', 'contrast'],
            'case_study': ['case study', 'case analysis', 'case report'],
            'analytical': ['analysis', 'analytical', 'critical analysis'],
            'methodological': ['method', 'methodology', 'approach', 'technique'],
            'position': ['position', 'opinion', 'viewpoint', 'perspective'],
            'technical': ['technical', 'implementation', 'system', 'algorithm'],
            'interdisciplinary': ['interdisciplinary', 'cross-disciplinary', 'multi-disciplinary']
        }
        
        scores = {}
        for paper_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[paper_type] = score
        
        # Return the type with highest score, default to empirical
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'empirical'
    
    def classify_academic_field(self, title: str, abstract: str, keywords: List[str]) -> str:
        """
        Classify academic field based on content
        """
        text = f"{title} {abstract} {' '.join(keywords)}".lower()
        
        field_keywords = {
            'computer_science': ['algorithm', 'machine learning', 'ai', 'artificial intelligence', 'software', 'programming', 'data science'],
            'medicine': ['clinical', 'medical', 'healthcare', 'patient', 'treatment', 'diagnosis', 'disease'],
            'social_sciences': ['social', 'society', 'human', 'behavior', 'psychology', 'sociology'],
            'engineering': ['engineering', 'mechanical', 'electrical', 'civil', 'chemical'],
            'physics': ['physics', 'quantum', 'particle', 'energy', 'matter'],
            'chemistry': ['chemistry', 'chemical', 'molecular', 'reaction', 'compound'],
            'biology': ['biology', 'biological', 'genetic', 'cell', 'organism'],
            'mathematics': ['mathematics', 'mathematical', 'equation', 'theorem', 'proof'],
            'economics': ['economics', 'economic', 'market', 'financial', 'business'],
            'education': ['education', 'learning', 'teaching', 'student', 'academic']
        }
        
        scores = {}
        for field, keywords in field_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            scores[field] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else 'general'
    
    def extract_sections(self, full_text: str) -> Dict:
        """
        Extract paper sections from full text
        """
        sections = {}
        
        # Common section headers
        section_headers = [
            'introduction', 'abstract', 'methodology', 'methods', 'results', 
            'discussion', 'conclusion', 'literature review', 'background',
            'theoretical framework', 'analysis', 'findings'
        ]
        
        # Simple section extraction (can be enhanced with NLP)
        lines = full_text.split('\n')
        current_section = 'content'
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line contains section header
            is_header = any(header in line_lower for header in section_headers)
            
            if is_header and len(line.strip()) < 100:  # Likely a header
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def save_paper(self, paper_data: Dict, filename: str):
        """
        Save processed paper to file
        """
        try:
            filepath = f"processed_papers/raw_papers/core_papers/{filename}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(paper_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Error saving paper: {e}")
            return False
    
    def collect_papers_by_topic(self, topics: List[str], papers_per_topic: int = 100) -> List[Dict]:
        """
        Collect papers for multiple topics
        """
        all_papers = []
        
        for topic in topics:
            self.logger.info(f"Collecting papers for topic: {topic}")
            
            papers = self.search_papers(topic, limit=papers_per_topic)
            
            for i, paper in enumerate(papers):
                self.logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')}")
                
                # Get detailed paper information
                paper_details = self.get_paper_details(paper.get('id'))
                if paper_details:
                    processed_paper = self.extract_paper_content(paper_details)
                    if processed_paper:
                        # Save individual paper
                        filename = f"{topic.replace(' ', '_')}_{i}_{processed_paper.get('paper_type', 'unknown')}"
                        self.save_paper(processed_paper, filename)
                        all_papers.append(processed_paper)
                
                # Rate limiting to be respectful to API
                time.sleep(0.1)
        
        return all_papers
    
    def create_training_dataset(self, papers: List[Dict]) -> Dict:
        """
        Create training dataset from collected papers
        """
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
            paper_type = paper.get('paper_type', 'unknown')
            field = paper.get('field', 'unknown')
            
            training_data['statistics']['paper_types'][paper_type] = \
                training_data['statistics']['paper_types'].get(paper_type, 0) + 1
            
            training_data['statistics']['fields'][field] = \
                training_data['statistics']['fields'].get(field, 0) + 1
        
        return training_data
    
    def save_training_data(self, training_data: Dict, filename: str = 'core_training_dataset.json'):
        """
        Save training dataset
        """
        try:
            filepath = f"processed_papers/training_data/core_training/{filename}"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Training data saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving training data: {e}")
            return False
    
    def generate_plagiarism_test_cases(self, papers: List[Dict]) -> List[Dict]:
        """
        Generate plagiarism detection test cases
        """
        test_cases = []
        
        for paper in papers:
            abstract = paper.get('abstract', '')
            if len(abstract) > 50:  # Only use substantial abstracts
                # Create paraphrased version (simplified)
                paraphrased = self.paraphrase_text(abstract)
                
                test_case = {
                    'original_text': abstract,
                    'test_text': paraphrased,
                    'variation_type': 'paraphrased',
                    'expected_similarity': 0.7,
                    'expected_category': 'medium',
                    'paper_id': paper.get('core_id'),
                    'paper_title': paper.get('title')
                }
                test_cases.append(test_case)
        
        return test_cases
    
    def paraphrase_text(self, text: str) -> str:
        """
        Simple text paraphrasing (can be enhanced with NLP models)
        """
        # Simple word replacement for demonstration
        replacements = {
            'study': 'research',
            'analysis': 'examination',
            'results': 'findings',
            'method': 'approach',
            'data': 'information',
            'shows': 'demonstrates',
            'found': 'discovered',
            'conclude': 'determine'
        }
        
        paraphrased = text
        for original, replacement in replacements.items():
            paraphrased = paraphrased.replace(original, replacement)
        
        return paraphrased
    
    def run_full_collection(self, topics: List[str], papers_per_topic: int = 50):
        """
        Run complete paper collection and processing pipeline
        """
        self.logger.info("Starting full paper collection pipeline...")
        
        # Collect papers
        papers = self.collect_papers_by_topic(topics, papers_per_topic)
        
        if not papers:
            self.logger.error("No papers collected!")
            return
        
        # Create training dataset
        training_data = self.create_training_dataset(papers)
        
        # Save training data
        self.save_training_data(training_data)
        
        # Generate plagiarism test cases
        test_cases = self.generate_plagiarism_test_cases(papers)
        
        # Save test cases
        test_cases_data = {
            'test_cases': test_cases,
            'statistics': {
                'total_test_cases': len(test_cases),
                'generation_date': datetime.now().isoformat()
            }
        }
        
        test_filepath = f"processed_papers/training_data/core_training/core_plagiarism_test_cases.json"
        with open(test_filepath, 'w', encoding='utf-8') as f:
            json.dump(test_cases_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Pipeline completed! Collected {len(papers)} papers")
        self.logger.info(f"Generated {len(test_cases)} plagiarism test cases")
        
        return training_data, test_cases_data

# Example usage
if __name__ == "__main__":
    # You'll need to get your CORE API key from: https://core.ac.uk/services/api/
    API_KEY = "YOUR_CORE_API_KEY_HERE"  # Replace with your actual API key
    
    # Initialize collector
    collector = CoreAPIPaperCollector(API_KEY)
    
    # Define topics to collect papers for
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
    
    # Run collection (start with small numbers for testing)
    training_data, test_cases = collector.run_full_collection(topics, papers_per_topic=20)
    
    print(f"Collection completed!")
    print(f"Papers collected: {len(training_data['papers'])}")
    print(f"Test cases generated: {len(test_cases['test_cases'])}") 