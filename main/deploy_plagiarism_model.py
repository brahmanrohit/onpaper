#!/usr/bin/env python3
"""
Deployment script for the ML-based Plagiarism Detection System
This script demonstrates how to deploy and use the plagiarism detection model.
"""

import os
import sys
import json
from datetime import datetime
from utils.text_analyzer import PlagiarismDetector, check_plagiarism, add_reference_document

def create_sample_dataset():
    """Create a sample dataset for testing the plagiarism detection system."""
    sample_documents = [
        {
            "id": "research_paper_1",
            "title": "Machine Learning in Healthcare",
            "content": """
            Machine learning has revolutionized healthcare by enabling predictive analytics 
            and personalized medicine. Deep learning algorithms can analyze medical images 
            with remarkable accuracy, helping doctors diagnose diseases earlier and more 
            accurately. The integration of AI in healthcare systems has shown significant 
            improvements in patient outcomes and treatment efficiency.
            """
        },
        {
            "id": "research_paper_2", 
            "title": "Climate Change Impact Analysis",
            "content": """
            Climate change represents one of the most critical challenges facing humanity. 
            Rising global temperatures have led to melting polar ice caps and rising sea 
            levels. Scientists have documented unprecedented changes in weather patterns, 
            with increased frequency of extreme weather events worldwide.
            """
        },
        {
            "id": "research_paper_3",
            "title": "Blockchain Technology Applications",
            "content": """
            Blockchain technology has emerged as a revolutionary force in digital 
            transactions and data security. Its decentralized nature provides enhanced 
            security and transparency for various applications including cryptocurrency, 
            supply chain management, and digital identity verification.
            """
        }
    ]
    return sample_documents

def test_plagiarism_detection():
    """Test the plagiarism detection system with various scenarios."""
    print("🚀 Testing ML-based Plagiarism Detection System")
    print("=" * 60)
    
    # Initialize detector
    detector = PlagiarismDetector()
    
    # Add sample documents
    sample_docs = create_sample_dataset()
    for doc in sample_docs:
        add_reference_document(doc["content"], doc["id"])
        print(f"✅ Added reference document: {doc['title']}")
    
    # Test cases
    test_cases = [
        {
            "name": "Original Text",
            "text": "Quantum computing represents a paradigm shift in computational power. Unlike classical computers that use bits, quantum computers use quantum bits or qubits that can exist in multiple states simultaneously.",
            "expected": "low"
        },
        {
            "name": "Slightly Modified Text",
            "text": "Machine learning has transformed healthcare by allowing predictive analytics and personalized medicine. Deep learning algorithms can examine medical images with impressive accuracy, assisting physicians in diagnosing diseases earlier and more precisely.",
            "expected": "moderate"
        },
        {
            "name": "Highly Similar Text",
            "text": "Machine learning has revolutionized healthcare by enabling predictive analytics and personalized medicine. Deep learning algorithms can analyze medical images with remarkable accuracy, helping doctors diagnose diseases earlier and more accurately.",
            "expected": "high"
        }
    ]
    
    print("\n📊 Running Plagiarism Detection Tests")
    print("-" * 40)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Text: {test_case['text'][:100]}...")
        
        result = check_plagiarism(test_case['text'])
        
        print(f"Plagiarism Score: {result['plagiarism_score']}%")
        print(f"Status: {result['message']}")
        
        if result['similar_sentences']:
            print(f"Similar sentences found: {len(result['similar_sentences'])}")
            for similar in result['similar_sentences'][:2]:
                print(f"  - Similarity: {similar['similarity']:.2%} with {similar['reference_doc']}")
        else:
            print("  - No similar sentences found")
    
    # Save the model
    detector.save_model()
    print(f"\n💾 Model saved to: {detector.model_path}")
    
    # Display statistics
    stats = detector.get_statistics()
    print(f"\n📈 System Statistics:")
    print(f"  - Reference Documents: {stats['total_reference_documents']}")
    print(f"  - Total Sentences: {stats['total_reference_sentences']}")
    print(f"  - Model Trained: {stats['model_trained']}")

def deploy_model():
    """Deploy the model for production use."""
    print("🚀 Deploying Plagiarism Detection Model")
    print("=" * 50)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize and train model
    detector = PlagiarismDetector("models/plagiarism_model.pkl")
    
    # Add sample data if no model exists
    if not os.path.exists("models/plagiarism_model.pkl"):
        print("📚 Initializing with sample dataset...")
        sample_docs = create_sample_dataset()
        for doc in sample_docs:
            add_reference_document(doc["content"], doc["id"])
        detector.save_model()
        print("✅ Model initialized and saved")
    
    # Create deployment configuration
    config = {
        "model_path": "models/plagiarism_model.pkl",
        "threshold": 0.7,
        "max_features": 5000,
        "deployment_date": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    
    with open("models/deployment_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Deployment configuration saved")
    print("🎉 Model deployed successfully!")
    
    return detector

def main():
    """Main deployment function."""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_plagiarism_detection()
    else:
        deploy_model()

if __name__ == "__main__":
    main() 