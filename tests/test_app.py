#!/usr/bin/env python3
"""
Test script to verify the application can start without errors.
"""

import sys
import os

def test_app_startup():
    """Test that the main application can be imported and started."""
    print("Testing application startup...")
    
    try:
        # Add the main directory to path
        main_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main')
        sys.path.insert(0, main_dir)
        
        # Try to import the main module
        import main
        
        print("✓ Main application imported successfully!")
        
        # Test that key functions exist
        if hasattr(main, 'st'):
            print("✓ Streamlit imported successfully!")
        
        print("✓ Application startup test passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_configuration():
    """Test that configuration is accessible."""
    print("Testing configuration access...")
    
    try:
        # Add the project root to path
        project_root = os.path.dirname(os.path.dirname(__file__))
        sys.path.insert(0, project_root)
        
        # Test that config can be imported
        from src.utils.config import (
            PLAGIARISM_MODEL_PATH,
            PAPER_TYPES,
            CITATION_STYLES
        )
        
        # Test that paths exist
        if PLAGIARISM_MODEL_PATH.exists():
            print("✓ Model paths are valid!")
        else:
            print("⚠ Model files not found (this is expected if models haven't been trained)")
        
        print("✓ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

if __name__ == "__main__":
    print("=== Application Test Results ===")
    
    startup_ok = test_app_startup()
    config_ok = test_configuration()
    
    if startup_ok and config_ok:
        print("\n🎉 Application is ready to run!")
        print("Run: python run.py")
    else:
        print("\n❌ Some tests failed. Please check the setup.")
