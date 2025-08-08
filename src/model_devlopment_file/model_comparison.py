"""
Model Comparison and Testing
Comprehensive testing of enhanced ML models with performance metrics
"""

import sys
import os
from pathlib import Path

# Get the absolute path to the project root
project_root = Path(__file__).parent.parent.parent  # Go up to onpaperfixed root
utils_path = project_root / "main" / "utils"

# Add paths to sys.path
sys.path.append(str(project_root))
sys.path.append(str(utils_path))

# Local imports
from src.model_devlopment_file.enhanced_ml_models import EnhancedPlagiarismDetector, EnhancedPaperTypeDetector
from main.utils.text_analyzer import PlagiarismDetector
from main.utils.paper_type_detector import ResearchPaperTypeDetector as PaperTypeDetector
import numpy as np
import time
from typing import Dict, List, Tuple

def create_test_dataset():
    """Create comprehensive test dataset"""
    
    # Plagiarism test cases
    plagiarism_test_cases = [
        ("Machine learning algorithms have revolutionized the field of artificial intelligence.", "High Plagiarism"),
        ("AI has been transformed by machine learning algorithms.", "Medium Plagiarism"),
        ("Deep learning techniques are advancing computer vision applications.", "Low Plagiarism"),
        ("The weather patterns in tropical regions are influenced by ocean currents.", "No Plagiarism"),
        ("The implementation of convolutional neural networks demonstrates superior performance in image classification tasks.", "No Plagiarism"),
        ("AI is useful.", "No Plagiarism"),
        ("According to Smith et al. (2020), machine learning shows promise. Johnson (2019) argues that AI will transform industries.", "No Plagiarism"),
        ("Neural network implementations exhibit exceptional efficacy in visual recognition applications.", "Medium Plagiarism"),
    ]
    
    # Paper type test cases
    paper_type_test_cases = [
        ("Experimental study on machine learning algorithms with 200 participants and statistical analysis", "empirical"),
        ("Theoretical framework for understanding neural networks and conceptual models", "theoretical"),
        ("Systematic review of deep learning applications in healthcare and medical imaging", "review"),
        ("Comparative analysis of different optimization techniques and their performance evaluation", "comparative"),
        ("Case study of implementing AI systems in a manufacturing company", "case_study"),
        ("Analytical examination of data preprocessing methods and their impact on model accuracy", "analytical"),
        ("Methodological approach to developing new algorithms for natural language processing", "methodological"),
        ("Position paper advocating for ethical considerations in artificial intelligence development", "position"),
        ("Technical implementation of computer vision algorithms for autonomous vehicles", "technical"),
        ("Interdisciplinary research combining psychology and computer science for human-AI interaction", "interdisciplinary")
    ]
    
    return {
        'plagiarism': plagiarism_test_cases,
        'paper_type': paper_type_test_cases
    }

def test_models(test_data: Dict, model_type: str = "enhanced") -> Dict:
    """Test models (enhanced or original)"""
    print(f"🔍 Testing {'Enhanced' if model_type == 'enhanced' else 'Original'} Models...")
    
    results = {'plagiarism': [], 'paper_type': []}
    
    # Test plagiarism detection
    try:
        if model_type == "enhanced":
            detector = EnhancedPlagiarismDetector()
        else:
            detector = PlagiarismDetector()
        
        # Add reference documents
        reference_docs = [
            "Machine learning algorithms have revolutionized the field of artificial intelligence.",
            "The study conducted experiments with 100 participants to evaluate the effectiveness.",
            "Previous research has demonstrated the importance of data preprocessing in ML.",
            "This paper presents a novel approach to natural language processing.",
            "Neural networks have shown remarkable performance in various applications."
        ]
        
        if model_type == "enhanced":
            detector.add_reference_documents(reference_docs)
        else:
            for i, doc in enumerate(reference_docs):
                detector.add_reference_document(doc, f"ref_{i}")
        
        for text, expected in test_data['plagiarism']:
            start_time = time.time()
            
            if model_type == "enhanced":
                result = detector.detect_plagiarism(text)
                category = result['category']
                score = result['plagiarism_score']
                confidence = result['confidence']
            else:
                result = detector.check_plagiarism(text)
                score = result['plagiarism_score'] / 100
                confidence = min(score * 1.2, 0.95)
                if score > 0.7:
                    category = 'High Plagiarism'
                elif score > 0.4:
                    category = 'Medium Plagiarism'
                elif score > 0.2:
                    category = 'Low Plagiarism'
                else:
                    category = 'No Plagiarism'
            
            execution_time = time.time() - start_time
            
            results['plagiarism'].append({
                'text': text,
                'predicted': category,
                'expected': expected,
                'score': score,
                'confidence': confidence,
                'execution_time': execution_time
            })
            
    except Exception as e:
        print(f"Error testing plagiarism detection: {e}")
        results['plagiarism'] = []
    
    # Test paper type detection
    try:
        if model_type == "enhanced":
            detector = EnhancedPaperTypeDetector()
        else:
            detector = PaperTypeDetector()
        
        for text, expected in test_data['paper_type']:
            start_time = time.time()
            
            if model_type == "enhanced":
                result = detector.detect_paper_type(text)
                predicted = result['type']
                confidence = result['confidence']
            else:
                result = detector.detect_paper_type(text)
                predicted = result.get('detected_type', 'unknown')
                confidence = result.get('confidence', 0.0)
            
            execution_time = time.time() - start_time
            
            results['paper_type'].append({
                'text': text,
                'predicted': predicted,
                'expected': expected,
                'confidence': confidence,
                'execution_time': execution_time
            })
            
    except Exception as e:
        print(f"Error testing paper type detection: {e}")
        results['paper_type'] = []
    
    return results

def calculate_metrics(results: Dict) -> Dict:
    """Calculate performance metrics"""
    metrics = {}
    
    for task in ['plagiarism', 'paper_type']:
        if not results[task]:
            continue
            
        predictions = [r['predicted'] for r in results[task]]
        expected = [r['expected'] for r in results[task]]
        
        # Basic metrics
        correct = sum(1 for p, e in zip(predictions, expected) if p == e)
        accuracy = correct / len(predictions) if predictions else 0
        confidences = [r['confidence'] for r in results[task]]
        execution_times = [r['execution_time'] for r in results[task]]
        
        metrics[task] = {
            'accuracy': accuracy,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'total_tests': len(predictions)
        }
        
        # Task-specific metrics
        if task == 'plagiarism':
            scores = [r['score'] for r in results[task]]
            score_accuracy = sum(1 for r in results[task] if 
                (r['expected'] == 'High Plagiarism' and r['score'] > 0.7) or
                (r['expected'] == 'Medium Plagiarism' and 0.4 <= r['score'] <= 0.7) or
                (r['expected'] == 'Low Plagiarism' and 0.2 <= r['score'] < 0.4) or
                (r['expected'] == 'No Plagiarism' and r['score'] < 0.2)
            ) / len(scores) if scores else 0
            metrics[task]['score_accuracy'] = score_accuracy
    
    return metrics

def print_comparison(original_metrics: Dict, enhanced_metrics: Dict):
    """Print comparison between original and enhanced models"""
    print("\n📊 MODEL COMPARISON RESULTS")
    print("="*60)
    
    for task in ['plagiarism', 'paper_type']:
        if task in original_metrics and task in enhanced_metrics:
            print(f"\n{task.upper()} DETECTION")
            print("-"*40)
            
            metrics = [
                ('Accuracy', 'accuracy', '0.85'),
                ('Avg Confidence', 'avg_confidence', '0.80'),
                ('Execution Time', 'avg_execution_time', '<0.1s')
            ]
            
            if task == 'plagiarism':
                metrics.append(('Score Accuracy', 'score_accuracy', '0.85'))
            
            for name, key, target in metrics:
                orig = original_metrics[task].get(key, 0)
                enhanced = enhanced_metrics[task].get(key, 0)
                improvement = ((enhanced - orig) / orig * 100) if orig > 0 else float('inf')
                
                print(f"{name:15} Original: {orig:.3f}, Enhanced: {enhanced:.3f} ({improvement:+.1f}%)")

def main():
    """Main function"""
    print("🎯 MODEL COMPARISON ANALYSIS")
    print("="*60)
    
    # Create test dataset
    test_data = create_test_dataset()
    
    # Test both model versions
    original_results = test_models(test_data, "original")
    enhanced_results = test_models(test_data, "enhanced")
    
    # Calculate metrics
    original_metrics = calculate_metrics(original_results)
    enhanced_metrics = calculate_metrics(enhanced_results)
    
    # Print comparison
    print_comparison(original_metrics, enhanced_metrics)
    
    print("\n✅ Analysis completed!")

if __name__ == "__main__":
    main() 