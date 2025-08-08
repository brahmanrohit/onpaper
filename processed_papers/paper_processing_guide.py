"""
Academic Paper Processing Guide
Step-by-step instructions for preparing collected papers for ML model training
"""

import os
import json
import re
from typing import Dict, List, Any
from pathlib import Path

class PaperProcessor:
    """Process collected academic papers for ML training"""
    
    def __init__(self):
        self.output_dir = "processed_papers"
        self.paper_types = [
            "empirical", "theoretical", "review", "comparative", 
            "case_study", "analytical", "methodological", "position", 
            "technical", "interdisciplinary"
        ]
        self.fields = [
            "computer_science", "medicine", "social_sciences", 
            "engineering", "humanities", "physics", "chemistry", 
            "biology", "economics", "psychology"
        ]
    
    def create_directory_structure(self):
        """Create organized directory structure for processed papers"""
        directories = [
            self.output_dir,
            f"{self.output_dir}/raw_papers",
            f"{self.output_dir}/processed_papers",
            f"{self.output_dir}/training_data",
            f"{self.output_dir}/validation_data",
            f"{self.output_dir}/test_data"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def extract_paper_metadata(self, paper_content: str) -> Dict[str, Any]:
        """Extract metadata from paper content"""
        metadata = {
            "title": "",
            "authors": [],
            "abstract": "",
            "keywords": [],
            "paper_type": "unknown",
            "field": "unknown",
            "year": "",
            "doi": "",
            "content": paper_content
        }
        
        # Extract title (usually first line or after "Title:")
        title_patterns = [
            r"Title:\s*(.+)",
            r"^([A-Z][^.!?]+[.!?])",
            r"^([A-Z][^.!?]+)"
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, paper_content, re.MULTILINE)
            if match:
                metadata["title"] = match.group(1).strip()
                break
        
        # Extract abstract
        abstract_patterns = [
            r"Abstract[:\s]*([^]*?)(?=\n\n|\n[A-Z]|Introduction)",
            r"ABSTRACT[:\s]*([^]*?)(?=\n\n|\n[A-Z]|Introduction)"
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, paper_content, re.IGNORECASE | re.DOTALL)
            if match:
                metadata["abstract"] = match.group(1).strip()
                break
        
        # Extract keywords
        keyword_patterns = [
            r"Keywords[:\s]*([^]*?)(?=\n\n|\n[A-Z])",
            r"Key words[:\s]*([^]*?)(?=\n\n|\n[A-Z])"
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, paper_content, re.IGNORECASE | re.DOTALL)
            if match:
                keywords_text = match.group(1).strip()
                metadata["keywords"] = [kw.strip() for kw in keywords_text.split(',')]
                break
        
        return metadata
    
    def classify_paper_type(self, paper_content: str, title: str = "") -> str:
        """Classify paper type based on content and title"""
        content_lower = (paper_content + " " + title).lower()
        
        # Define keywords for each paper type
        type_keywords = {
            "empirical": ["experiment", "study", "participants", "data collection", "results", "statistical analysis", "survey", "interview"],
            "theoretical": ["theory", "framework", "conceptual", "model", "proposition", "theoretical foundation"],
            "review": ["literature review", "systematic review", "meta-analysis", "previous studies", "existing research", "overview"],
            "comparative": ["comparison", "compare", "versus", "contrast", "different", "similarities", "differences"],
            "case_study": ["case study", "case analysis", "specific case", "real-world", "practical example"],
            "analytical": ["analysis", "analytical", "critical analysis", "examination", "investigation", "evaluation"],
            "methodological": ["methodology", "method", "approach", "technique", "procedure", "methodological framework"],
            "position": ["position", "argument", "perspective", "viewpoint", "stance", "proposal"],
            "technical": ["technical", "implementation", "algorithm", "system design", "technical specification"],
            "interdisciplinary": ["interdisciplinary", "multidisciplinary", "cross-disciplinary", "integration", "collaboration"]
        }
        
        # Count keyword matches for each type
        type_scores = {}
        for paper_type, keywords in type_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            type_scores[paper_type] = score
        
        # Return the type with highest score
        return max(type_scores, key=type_scores.get)
    
    def classify_field(self, paper_content: str, title: str = "", keywords: List[str] = []) -> str:
        """Classify academic field based on content"""
        content_lower = (paper_content + " " + title + " " + " ".join(keywords)).lower()
        
        # Define field-specific keywords
        field_keywords = {
            "computer_science": ["algorithm", "programming", "software", "data", "machine learning", "artificial intelligence", "computer", "computing"],
            "medicine": ["clinical", "patient", "treatment", "medical", "healthcare", "diagnosis", "therapy", "disease"],
            "social_sciences": ["social", "behavior", "society", "human", "sociology", "psychology", "anthropology"],
            "engineering": ["engineering", "technical", "system", "design", "mechanical", "electrical", "civil"],
            "humanities": ["philosophy", "literature", "history", "culture", "art", "language", "humanities"],
            "physics": ["physics", "quantum", "particle", "energy", "force", "matter", "universe"],
            "chemistry": ["chemistry", "chemical", "molecule", "reaction", "compound", "element"],
            "biology": ["biology", "biological", "organism", "cell", "gene", "evolution", "species"],
            "economics": ["economics", "economic", "market", "finance", "trade", "business", "economy"],
            "psychology": ["psychology", "psychological", "behavior", "cognitive", "mental", "brain", "mind"]
        }
        
        # Count keyword matches for each field
        field_scores = {}
        for field, keywords in field_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            field_scores[field] = score
        
        # Return the field with highest score
        return max(field_scores, key=field_scores.get)
    
    def clean_paper_content(self, content: str) -> str:
        """Clean and normalize paper content"""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters but keep basic punctuation
        content = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]]', '', content)
        
        # Normalize line breaks
        content = content.replace('\n', ' ').replace('\r', ' ')
        
        # Remove page numbers and headers
        content = re.sub(r'Page \d+', '', content)
        content = re.sub(r'^\d+\s*', '', content, flags=re.MULTILINE)
        
        return content.strip()
    
    def process_single_paper(self, paper_content: str, filename: str = "") -> Dict[str, Any]:
        """Process a single paper and extract all necessary information"""
        # Clean content
        cleaned_content = self.clean_paper_content(paper_content)
        
        # Extract metadata
        metadata = self.extract_paper_metadata(cleaned_content)
        
        # Classify paper type and field
        paper_type = self.classify_paper_type(cleaned_content, metadata["title"])
        field = self.classify_field(cleaned_content, metadata["title"], metadata["keywords"])
        
        # Update metadata
        metadata["paper_type"] = paper_type
        metadata["field"] = field
        metadata["filename"] = filename
        metadata["word_count"] = len(cleaned_content.split())
        
        return metadata
    
    def create_training_dataset(self, processed_papers: List[Dict]) -> Dict[str, Any]:
        """Create structured training dataset"""
        dataset = {
            "papers": processed_papers,
            "statistics": {
                "total_papers": len(processed_papers),
                "paper_types": {},
                "fields": {},
                "avg_word_count": 0
            }
        }
        
        # Calculate statistics
        word_counts = []
        for paper in processed_papers:
            # Count paper types
            paper_type = paper["paper_type"]
            dataset["statistics"]["paper_types"][paper_type] = dataset["statistics"]["paper_types"].get(paper_type, 0) + 1
            
            # Count fields
            field = paper["field"]
            dataset["statistics"]["fields"][field] = dataset["statistics"]["fields"].get(field, 0) + 1
            
            # Collect word counts
            word_counts.append(paper["word_count"])
        
        dataset["statistics"]["avg_word_count"] = sum(word_counts) / len(word_counts) if word_counts else 0
        
        return dataset
    
    def save_processed_data(self, dataset: Dict, filename: str):
        """Save processed dataset to JSON file"""
        output_path = os.path.join(self.output_dir, "training_data", filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved processed dataset to: {output_path}")
    
    def generate_plagiarism_test_cases(self, papers: List[Dict]) -> List[Dict]:
        """Generate plagiarism test cases from processed papers"""
        test_cases = []
        
        for i, paper in enumerate(papers):
            # Create original text
            original_text = paper["abstract"] if paper["abstract"] else paper["content"][:500]
            
            # Create different variations
            variations = [
                {
                    "type": "exact_copy",
                    "text": original_text,
                    "expected_similarity": 1.0,
                    "category": "high"
                },
                {
                    "type": "paraphrased",
                    "text": self.paraphrase_text(original_text),
                    "expected_similarity": 0.7,
                    "category": "medium"
                },
                {
                    "type": "similar_topic",
                    "text": self.create_similar_topic_text(paper),
                    "expected_similarity": 0.4,
                    "category": "low"
                },
                {
                    "type": "different_topic",
                    "text": self.create_different_topic_text(),
                    "expected_similarity": 0.1,
                    "category": "none"
                }
            ]
            
            for variation in variations:
                test_cases.append({
                    "original_paper_id": i,
                    "original_text": original_text,
                    "test_text": variation["text"],
                    "variation_type": variation["type"],
                    "expected_similarity": variation["expected_similarity"],
                    "expected_category": variation["category"]
                })
        
        return test_cases
    
    def paraphrase_text(self, text: str) -> str:
        """Simple paraphrasing for test cases"""
        # This is a simplified version - in practice, you might use more sophisticated methods
        paraphrases = {
            "study": "research",
            "analysis": "examination",
            "results": "findings",
            "method": "approach",
            "data": "information",
            "shows": "demonstrates",
            "found": "discovered",
            "important": "significant"
        }
        
        paraphrased = text
        for original, replacement in paraphrases.items():
            paraphrased = paraphrased.replace(original, replacement)
        
        return paraphrased
    
    def create_similar_topic_text(self, paper: Dict) -> str:
        """Create text on similar topic"""
        field = paper["field"]
        paper_type = paper["paper_type"]
        
        # Generate similar topic text based on field and type
        similar_texts = {
            "computer_science": "This research investigates computational methods and algorithmic approaches in the field of computer science.",
            "medicine": "This study examines clinical outcomes and patient responses in medical treatment protocols.",
            "social_sciences": "This analysis explores social behaviors and human interactions in various societal contexts."
        }
        
        return similar_texts.get(field, "This research examines related topics in the academic field.")
    
    def create_different_topic_text(self) -> str:
        """Create text on completely different topic"""
        different_topics = [
            "The weather patterns in tropical regions are influenced by ocean currents and atmospheric pressure systems.",
            "Cooking techniques vary significantly across different cultures and geographical regions.",
            "Sports performance is affected by various factors including training, nutrition, and psychological preparation."
        ]
        
        import random
        return random.choice(different_topics)

def main():
    """Main processing function"""
    processor = PaperProcessor()
    
    print("📚 Academic Paper Processing Guide")
    print("=" * 50)
    
    # Create directory structure
    processor.create_directory_structure()
    
    print("\n📋 Next Steps:")
    print("1. Place your collected papers in the 'processed_papers/raw_papers' directory")
    print("2. Run the processing script to extract metadata and classify papers")
    print("3. Review and validate the classifications")
    print("4. Generate training datasets for ML models")
    
    print("\n📁 Directory Structure Created:")
    print("processed_papers/")
    print("├── raw_papers/          # Place your collected papers here")
    print("├── processed_papers/    # Processed individual papers")
    print("├── training_data/       # Final training datasets")
    print("├── validation_data/     # Validation datasets")
    print("└── test_data/          # Test datasets")

if __name__ == "__main__":
    main() 