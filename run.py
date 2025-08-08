#!/usr/bin/env python3
"""
Entry point for the Research Paper Assistant application.
This script runs the Streamlit application from the organized structure.
"""

import os
import sys
import subprocess

def main():
    """Run the Streamlit application."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the main directory where main.py is located
    main_dir = os.path.join(script_dir, 'main')
    
    if not os.path.exists(os.path.join(main_dir, 'main.py')):
        print("Error: main.py not found in main directory")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    print("🚀 Starting Research Paper Assistant...")
    print(f"📁 Project root: {script_dir}")
    print(f"📄 Main app: {main_dir}/main.py")
    
    # Change to the main directory
    os.chdir(main_dir)
    
    # Run streamlit
    try:
        subprocess.run(['streamlit', 'run', 'main.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: streamlit not found. Please install streamlit first.")
        print("Run: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
