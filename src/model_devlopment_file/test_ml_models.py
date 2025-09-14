#!/usr/bin/env python3
"""
Comprehensive ML Model Testing Script
Tests both plagiarism detection and paper type detection models for accuracy and performance.
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Tuple

# Add the project path to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
main_path = os.path.join(project_root, 'main')
sys.path.append(main_path)

from utils.text_analyzer import PlagiarismDetector, check_plagiarism, add_reference_document
from utils.paper_type_detector import ResearchPaperTypeDetector

def test_plagiarism_detection_accuracy():
    """Test plagiarism detection model with known test cases."""
    print("🔍 Testing Plagiarism Detection Model Accuracy")
    print("=" * 60)
    
    # Initialize detector
    detector = PlagiarismDetector()
    
    # Test dataset with known plagiarism levels
    test_cases = [
        {
            "name": "Original Text (No Plagiarism)",
            "text": "Quantum computing represents a paradigm shift in computational power. Unlike classical computers that use bits, quantum computers use quantum bits or qubits that can exist in multiple states simultaneously.",
            "expected_score": 0,
            "expected_category": "low"
        },
        {
            "name": "Slightly Modified Text (Low Plagiarism)",
            "text": "Machine learning has transformed healthcare by allowing predictive analytics and personalized medicine. Deep learning algorithms can examine medical images with impressive accuracy, assisting physicians in diagnosing diseases earlier and more precisely.",
            "expected_score": 30,
            "expected_category": "low"
        },
        {
            "name": "Moderately Similar Text (Medium Plagiarism)",
            "text": "Machine learning has revolutionized healthcare by enabling predictive analytics and personalized medicine. Deep learning algorithms can analyze medical images with remarkable accuracy, helping doctors diagnose diseases earlier and more accurately.",
            "expected_score": 60,
            "expected_category": "medium"
        },
        {
            "name": "Highly Similar Text (High Plagiarism)",
            "text": "Machine learning has revolutionized healthcare by enabling predictive analytics and personalized medicine. Deep learning algorithms can analyze medical images with remarkable accuracy, helping doctors diagnose diseases earlier and more accurately. The integration of AI in healthcare systems has shown significant improvements in patient outcomes and treatment efficiency.",
            "expected_score": 80,
            "expected_category": "high"
        },
        {
            "name": "Exact Copy (Very High Plagiarism)",
            "text": "Machine learning has revolutionized healthcare by enabling predictive analytics and personalized medicine. Deep learning algorithms can analyze medical images with remarkable accuracy, helping doctors diagnose diseases earlier and more accurately.",
            "expected_score": 90,
            "expected_category": "very_high"
        }
    ]
    
    # Add reference documents
    reference_docs = [
        "Machine learning has revolutionized healthcare by enabling predictive analytics and personalized medicine. Deep learning algorithms can analyze medical images with remarkable accuracy, helping doctors diagnose diseases earlier and more accurately.",
        "Climate change represents one of the most critical challenges facing humanity. Rising global temperatures have led to melting polar ice caps and rising sea levels.",
        "Blockchain technology has emerged as a revolutionary force in digital transactions and data security. Its decentralized nature provides enhanced security and transparency."
    ]
    
    for i, doc in enumerate(reference_docs):
        add_reference_document(doc, f"ref_doc_{i}")
    
    # Test each case
    results = []
    start_time = time.time()
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['name']}")
        
        # Run plagiarism detection
        result = check_plagiarism(test_case['text'], threshold=0.7)
        score = result['plagiarism_score']
        
        # Categorize result
        if score < 30:
            detected_category = "low"
        elif score < 60:
            detected_category = "medium"
        elif score < 80:
            detected_category = "high"
        else:
            detected_category = "very_high"
        
        # Calculate accuracy
        category_accuracy = 1 if detected_category == test_case['expected_category'] else 0
        score_accuracy = 1 - min(abs(score - test_case['expected_score']) / 100, 1)
        
        results.append({
            "test_name": test_case['name'],
            "expected_score": test_case['expected_score'],
            "actual_score": score,
            "expected_category": test_case['expected_category'],
            "detected_category": detected_category,
            "category_accuracy": category_accuracy,
            "score_accuracy": score_accuracy,
            "similar_sentences": len(result['similar_sentences'])
        })
        
        print(f"  Expected Score: {test_case['expected_score']}%")
        print(f"  Actual Score: {score}%")
        print(f"  Expected Category: {test_case['expected_category']}")
        print(f"  Detected Category: {detected_category}")
        print(f"  Similar Sentences Found: {len(result['similar_sentences'])}")
    
    end_time = time.time()
    
    # Calculate overall accuracy
    total_tests = len(results)
    category_accuracy = sum(r['category_accuracy'] for r in results) / total_tests
    score_accuracy = sum(r['score_accuracy'] for r in results) / total_tests
    avg_processing_time = (end_time - start_time) / total_tests
    
    print(f"\n📊 Plagiarism Detection Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Category Accuracy: {category_accuracy:.2%}")
    print(f"  Score Accuracy: {score_accuracy:.2%}")
    print(f"  Average Processing Time: {avg_processing_time:.3f} seconds")
    
    return {
        "model": "plagiarism_detection",
        "category_accuracy": category_accuracy,
        "score_accuracy": score_accuracy,
        "processing_time": avg_processing_time,
        "total_tests": total_tests
    }

def test_paper_type_detection_accuracy():
    """Test paper type detection model with known test cases."""
    print("\n📄 Testing Paper Type Detection Model Accuracy")
    print("=" * 60)
    
    detector = ResearchPaperTypeDetector()
    
    # Test cases with known paper types
    test_cases = [
        {
            "topic": "Statistical analysis of student performance data using machine learning algorithms",
            "expected_type": "empirical",
            "expected_confidence": 0.8
        },
        {
            "topic": "Theoretical framework for understanding social media influence on behavior",
            "expected_type": "theoretical",
            "expected_confidence": 0.7
        },
        {
            "topic": "Systematic review of literature on artificial intelligence in healthcare",
            "expected_type": "review",
            "expected_confidence": 0.9
        },
        {
            "topic": "Comparative analysis of traditional vs modern teaching methods",
            "expected_type": "comparative",
            "expected_confidence": 0.8
        },
        {
            "topic": "Case study of successful digital transformation in retail industry",
            "expected_type": "case_study",
            "expected_confidence": 0.8
        },
        {
            "topic": "Critical analysis of blockchain technology applications",
            "expected_type": "analytical",
            "expected_confidence": 0.7
        },
        {
            "topic": "Development and validation of new research methodology for social sciences",
            "expected_type": "methodological",
            "expected_confidence": 0.8
        },
        {
            "topic": "Position paper on the future of remote work policies",
            "expected_type": "position",
            "expected_confidence": 0.8
        },
        {
            "topic": "Technical report on implementation of cloud computing infrastructure",
            "expected_type": "technical",
            "expected_confidence": 0.8
        },
        {
            "topic": "Interdisciplinary approach combining psychology and computer science for AI ethics",
            "expected_type": "interdisciplinary",
            "expected_confidence": 0.8
        }
    ]
    
    results = []
    start_time = time.time()
    
    for test_case in test_cases:
        print(f"\n📝 Testing: {test_case['topic'][:50]}...")
        
        # Run paper type detection
        result = detector.detect_paper_type(test_case['topic'])
        detected_type = result['detected_type']
        confidence = result['confidence']
        
        # Calculate accuracy
        type_accuracy = 1 if detected_type == test_case['expected_type'] else 0
        confidence_accuracy = 1 - min(abs(confidence - test_case['expected_confidence']), 1)
        
        results.append({
            "topic": test_case['topic'],
            "expected_type": test_case['expected_type'],
            "detected_type": detected_type,
            "expected_confidence": test_case['expected_confidence'],
            "actual_confidence": confidence,
            "type_accuracy": type_accuracy,
            "confidence_accuracy": confidence_accuracy
        })
        
        print(f"  Expected Type: {test_case['expected_type']}")
        print(f"  Detected Type: {detected_type}")
        print(f"  Expected Confidence: {test_case['expected_confidence']:.2f}")
        print(f"  Actual Confidence: {confidence:.2f}")
    
    end_time = time.time()
    
    # Calculate overall accuracy
    total_tests = len(results)
    type_accuracy = sum(r['type_accuracy'] for r in results) / total_tests
    confidence_accuracy = sum(r['confidence_accuracy'] for r in results) / total_tests
    avg_processing_time = (end_time - start_time) / total_tests
    
    print(f"\n📊 Paper Type Detection Results:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Type Accuracy: {type_accuracy:.2%}")
    print(f"  Confidence Accuracy: {confidence_accuracy:.2%}")
    print(f"  Average Processing Time: {avg_processing_time:.3f} seconds")
    
    return {
        "model": "paper_type_detection",
        "type_accuracy": type_accuracy,
        "confidence_accuracy": confidence_accuracy,
        "processing_time": avg_processing_time,
        "total_tests": total_tests
    }

def test_model_performance():
    """Test model performance with larger datasets."""
    print("\n⚡ Testing Model Performance")
    print("=" * 60)
    
    # Performance test for plagiarism detection
    print("\n🔍 Plagiarism Detection Performance Test:")
    
    # Create larger test dataset
    large_text = "This is a test document for performance evaluation. " * 1000
    start_time = time.time()
    
    for i in range(10):
        result = check_plagiarism(large_text[:1000 + i*100])
    
    end_time = time.time()
    plagiarism_avg_time = (end_time - start_time) / 10
    
    print(f"  Average processing time for large texts: {plagiarism_avg_time:.3f} seconds")
    
    # Performance test for paper type detection
    print("\n📄 Paper Type Detection Performance Test:")
    
    detector = ResearchPaperTypeDetector()
    start_time = time.time()
    
    for i in range(50):
        detector.detect_paper_type(f"Test topic number {i} for performance evaluation")
    
    end_time = time.time()
    paper_type_avg_time = (end_time - start_time) / 50
    
    print(f"  Average processing time per detection: {paper_type_avg_time:.3f} seconds")
    
    return {
        "plagiarism_avg_time": plagiarism_avg_time,
        "paper_type_avg_time": paper_type_avg_time
    }

def main():
    """Run all tests and generate comprehensive report."""
    print("🚀 Starting Comprehensive ML Model Testing")
    print("=" * 80)
    
    # Run all tests
    plagiarism_results = test_plagiarism_detection_accuracy()
    paper_type_results = test_paper_type_detection_accuracy()
    performance_results = test_model_performance()
    
    # Generate final report
    print("\n" + "=" * 80)
    print("📋 COMPREHENSIVE ML MODEL TESTING REPORT")
    print("=" * 80)
    
    print(f"\n🔍 Plagiarism Detection Model:")
    print(f"  Category Accuracy: {plagiarism_results['category_accuracy']:.2%}")
    print(f"  Score Accuracy: {plagiarism_results['score_accuracy']:.2%}")
    print(f"  Processing Time: {plagiarism_results['processing_time']:.3f} seconds")
    print(f"  Tests Run: {plagiarism_results['total_tests']}")
    
    print(f"\n📄 Paper Type Detection Model:")
    print(f"  Type Accuracy: {paper_type_results['type_accuracy']:.2%}")
    print(f"  Confidence Accuracy: {paper_type_results['confidence_accuracy']:.2%}")
    print(f"  Processing Time: {paper_type_results['processing_time']:.3f} seconds")
    print(f"  Tests Run: {paper_type_results['total_tests']}")
    
    print(f"\n⚡ Performance Metrics:")
    print(f"  Plagiarism Detection (Large Text): {performance_results['plagiarism_avg_time']:.3f} seconds")
    print(f"  Paper Type Detection (Per Detection): {performance_results['paper_type_avg_time']:.3f} seconds")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    if plagiarism_results['category_accuracy'] > 0.8:
        print(f"  ✅ Plagiarism Detection: EXCELLENT ({plagiarism_results['category_accuracy']:.2%})")
    elif plagiarism_results['category_accuracy'] > 0.6:
        print(f"  ⚠️ Plagiarism Detection: GOOD ({plagiarism_results['category_accuracy']:.2%})")
    else:
        print(f"  ❌ Plagiarism Detection: NEEDS IMPROVEMENT ({plagiarism_results['category_accuracy']:.2%})")
    
    if paper_type_results['type_accuracy'] > 0.8:
        print(f"  ✅ Paper Type Detection: EXCELLENT ({paper_type_results['type_accuracy']:.2%})")
    elif paper_type_results['type_accuracy'] > 0.6:
        print(f"  ⚠️ Paper Type Detection: GOOD ({paper_type_results['type_accuracy']:.2%})")
    else:
        print(f"  ❌ Paper Type Detection: NEEDS IMPROVEMENT ({paper_type_results['type_accuracy']:.2%})")

if __name__ == "__main__":
    main() 