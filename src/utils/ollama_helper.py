"""
Ollama AI Helper for Research Paper Assistant
Provides local AI capabilities using Ollama models.
"""

import os
import requests
from typing import Optional, Dict, Any

class OllamaHelper:
    """Helper class for Ollama AI integration."""
    
    def __init__(self):
        self.base_url = self._get_ollama_url()
        self.available = self._check_ollama_available()
        self.model = self._get_default_model()
        
    def _get_ollama_url(self) -> str:
        """Get Ollama server URL from environment or use default."""
        # Try to get from environment first
        url = os.getenv("OLLAMA_BASE_URL")
        if url:
            return url
            
        # Default URLs to try (in order of preference). For a non-default host,
        # set OLLAMA_BASE_URL in .env rather than hardcoding an IP here.
        default_urls = [
            "http://localhost:11434",  # Local
            "http://127.0.0.1:11434",  # Local alternative
            "http://host.docker.internal:11434",  # WSL/Docker bridge
        ]
        
        for url in default_urls:
            try:
                response = requests.get(f"{url}/api/tags", timeout=2)
                if response.status_code == 200:
                    print(f"✓ Ollama found at: {url}")
                    return url
            except:
                continue
                
        print("⚠️ Ollama not found at default URLs")
        print("Please ensure Ollama is running or set OLLAMA_BASE_URL in .env")
        return "http://localhost:11434"  # Default fallback
    
    def _get_default_model(self) -> str:
        """Get default model from environment or use sensible defaults."""
        model = os.getenv("OLLAMA_MODEL")
        if model:
            return model
            
        # Preferred models for academic writing
        preferred_models = [
            "llama3.2",  # Latest Llama
            "llama3.1",  # Previous Llama
            "llama3",    # Stable Llama
            "mistral",   # Good for academic writing
            "qwen2.5",   # Strong academic model
            "codellama", # For technical papers
        ]
        
        available_models = self.list_models()
        for model in preferred_models:
            if model in available_models:
                print(f"✓ Using model: {model}")
                return model
                
        # Fallback to first available model
        if available_models:
            print(f"✓ Using available model: {available_models[0]}")
            return available_models[0]
            
        print("⚠️ No models found. Please pull a model with: ollama pull llama3.2")
        return "llama3.2"  # Default fallback
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> list:
        """List all available Ollama models."""
        if not self.available:
            return []
            
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
            return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate_text(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate text using Ollama."""
        if not self.available:
            return "Ollama is not available. Please ensure Ollama is running."
        
        model_to_use = model or self.model
        
        try:
            payload = {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,  # Good for academic writing
                    "top_p": 0.9,
                    "max_tokens": 2048,
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Error: Empty response from Ollama')
            else:
                return f"Error: Ollama returned status {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Error: Request to Ollama timed out"
        except Exception as e:
            return f"Error generating text with Ollama: {str(e)}"
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a new model from Ollama."""
        if not self.available:
            print("Ollama is not available")
            return False
        
        try:
            print(f"Pulling model {model_name}... This may take a while.")
            payload = {"name": model_name}
            
            response = requests.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code == 200:
                print(f"✓ Successfully pulled model: {model_name}")
                return True
            else:
                print(f"Error pulling model: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        if not self.available:
            return {}
        
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{self.base_url}/api/show",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            return {}
            
        except Exception as e:
            print(f"Error getting model info: {e}")
            return {}

# Global instance
ollama_helper = None

def get_ollama_helper():
    """Get or create the global Ollama helper instance."""
    global ollama_helper
    if ollama_helper is None:
        ollama_helper = OllamaHelper()
    return ollama_helper

def generate_text_with_ollama(prompt: str, model: Optional[str] = None) -> str:
    """Convenience function to generate text with Ollama."""
    helper = get_ollama_helper()
    return helper.generate_text(prompt, model)

def is_ollama_available() -> bool:
    """Check if Ollama is available."""
    helper = get_ollama_helper()
    return helper.available

def list_available_models() -> list:
    """List all available Ollama models."""
    helper = get_ollama_helper()
    return helper.list_models()

# Initialize on import
if __name__ == "__main__":
    # Test Ollama connection
    helper = get_ollama_helper()
    print(f"Ollama available: {helper.available}")
    print(f"Base URL: {helper.base_url}")
    print(f"Default model: {helper.model}")
    print(f"Available models: {helper.list_models()}")
    
    # Test generation
    if helper.available:
        test_prompt = "Write a brief summary of machine learning."
        result = helper.generate_text(test_prompt)
        print(f"\nTest generation result:\n{result}")
