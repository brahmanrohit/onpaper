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
from .ollama_helper import get_ollama_helper, is_ollama_available

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
        MODEL_NAME = "models/gemini-flash-latest"  # Updated to correct model name
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

def generate_text(prompt, backend="auto"):
    """Generate text using available AI backend (Gemini or Ollama)."""
    
    # Get backend preference from environment or parameter
    if backend == "auto":
        backend = os.getenv("AI_BACKEND", "auto").lower()
    
    # Try Ollama first if available and preferred
    if backend in ["auto", "ollama"] and is_ollama_available():
        try:
            ollama_helper = get_ollama_helper()
            result = ollama_helper.generate_text(prompt)
            if not result.startswith("Error") and not result.startswith("Ollama is not available"):
                return result
            elif backend == "ollama":
                return result
        except Exception as e:
            if backend == "ollama":
                return f"Ollama error: {str(e)}"
    
    # Fall back to Gemini
    if backend in ["auto", "gemini"] and model is not None:
        try:
            response = model.generate_content(prompt)
            return response.text if response else "Error: Empty response from Gemini"
        except Exception as e:
            if backend == "gemini":
                return f"Gemini error: {str(e)}"
    
    # If we get here, no backend is available
    if backend == "auto":
        return "AI features are not available. Please set up GOOGLE_API_KEY or ensure Ollama is running."
    else:
        return f"Backend '{backend}' is not available. Please check your configuration."

def generate_text_with_gemini(prompt):
    """Generate text using only Gemini AI."""
    if model is None:
        return "Gemini AI is not available. Please set up GOOGLE_API_KEY in .env file."
    
    try:
        response = model.generate_content(prompt)
        return response.text if response else "Error: Empty response from AI"
    except Exception as e:
        return f"Error generating text with Gemini: {str(e)}"

def generate_text_with_ollama(prompt):
    """Generate text using only Ollama."""
    if not is_ollama_available():
        return "Ollama is not available. Please ensure Ollama is running."
    
    try:
        ollama_helper = get_ollama_helper()
        return ollama_helper.generate_text(prompt)
    except Exception as e:
        return f"Error generating text with Ollama: {str(e)}"

def get_available_backends():
    """Get list of available AI backends."""
    backends = []
    
    if is_ollama_available():
        backends.append("ollama")
    
    if model is not None:
        backends.append("gemini")
    
    return backends

def get_backend_status():
    """Get status of all AI backends."""
    status = {
        "gemini": {
            "available": model is not None,
            "model": "models/gemini-flash-latest" if model else None
        },
        "ollama": {
            "available": is_ollama_available(),
            "models": get_ollama_helper().list_models() if is_ollama_available() else []
        }
    }
    return status

