# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# # Get API Key from .env file
# API_KEY = os.getenv("GOOGLE_API_KEY")

# if not API_KEY:
#     raise ValueError("⚠️ Missing GOOGLE_API_KEY in .env file!")

# # Initialize Gemini AI
# try:
#     genai.configure(api_key=API_KEY)
# except Exception as e:
#     raise RuntimeError(f"Error initializing Gemini AI: {e}")

# def generate_text(prompt):
#     """Generate text using Google Gemini AI."""
#     try:
#         response = genai.generate_text(model="models/gemini-pro", prompt=prompt)

#         return response.text
#     except Exception as e:
#         return f"Error generating text: {str(e)}"


import os
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from multiple possible locations
def load_env_file():
    """Load .env file from multiple possible locations."""
    # Get the current file's directory
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir / '.env',  # In utils directory
        current_dir.parent.parent / '.env',  # In project root (onpaperfixed)
        current_dir.parent.parent.parent / '.env',  # In parent directory
        Path.cwd() / '.env',  # In current working directory
        Path.home() / '.env',  # In user home directory
    ]
    
    for env_path in possible_paths:
        if env_path.exists():
            print(f"Loading .env from: {env_path}")
            load_dotenv(env_path, override=True)
            return True
    
    print("Warning: No .env file found in any of the expected locations")
    print("Expected locations:")
    for path in possible_paths:
        print(f"  - {path}")
    return False

# Load environment variables
load_env_file()

# Get API Key from .env file
API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini AI only if API key is available
model = None
if API_KEY and API_KEY != "your_api_key_here" and API_KEY.strip():
    try:
        genai.configure(api_key=API_KEY)
        MODEL_NAME = "models/gemini-1.5-flash-latest"  # or "models/gemini-1.5-flash-latest" for speed
        model = genai.GenerativeModel(MODEL_NAME)  # Correct AI Studio Model
        print("✓ Gemini AI initialized successfully!")
    except Exception as e:
        print(f"Warning: Error initializing Gemini AI: {e}")
        model = None
else:
    print("Warning: GOOGLE_API_KEY not found or is placeholder in .env file. AI features will be limited.")
    print("To enable AI features, add your Google API key to the .env file:")
    print("1. Get your API key from: https://makersuite.google.com/app/apikey")
    print("2. Replace 'your_api_key_here' with your actual API key in the .env file")

def generate_text(prompt):
    """Generate text using Google AI Studio Gemini model."""
    if model is None:
        return "AI features are not available. Please set up GOOGLE_API_KEY in .env file."
    
    try:
        response = model.generate_content(prompt)  # Correct method for AI Studio
        return response.text if response else "Error: Empty response from AI"
    except Exception as e:
        return f"Error generating text: {str(e)}"

