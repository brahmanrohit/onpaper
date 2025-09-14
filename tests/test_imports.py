#!/usr/bin/env python3
"""
Test script to verify that all imports work correctly with the new structure.
"""

import sys
import os

def test_imports():
    """Test all critical imports."""
    print("Testing imports...")
    
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(__file__))
        sys.path.insert(0, project_root)
        sys.path.insert(0, os.path.join(project_root, 'main'))
        
        # Test main application imports
        print("✓ Testing main application imports...")
        
        # Test utils imports
        print("✓ Testing utils imports...")
        from src.utils.pdf_processor import extract_text_from_pdf, summarize_text
        from src.utils.text_analyzer import PlagiarismDetector, check_plagiarism
        from src.utils.citation_manager import suggest_citations, format_citation
        from src.utils.content_generator import generate_comprehensive_paper, get_available_paper_types
        from src.utils.default_paper_generator import generate_default_paper, create_word_document
        from src.utils.topic_type_predictor import TopicTypePredictor
        from src.utils.paper_type_detector import ResearchPaperTypeDetector
        from src.utils.gemini_helper import generate_text
        from src.utils.nlp_utils import generate_content, check_grammar
        
        print("✓ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    
    try:
        # Add project root to path
        project_root = os.path.dirname(os.path.dirname(__file__))
        sys.path.insert(0, project_root)
        
        from src.utils.config import (
            PLAGIARISM_MODEL_PATH, 
            PAPER_TYPE_MODEL_PATH, 
            TOPIC_TYPE_MODEL_PATH,
            PAPER_TYPES,
            CITATION_STYLES
        )
        print("✓ Configuration loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

if __name__ == "__main__":
    print("=== Import Test Results ===")
    
    imports_ok = test_imports()
    config_ok = test_config()
    
    if imports_ok and config_ok:
        print("\n🎉 All tests passed! The new structure is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the import paths.")
