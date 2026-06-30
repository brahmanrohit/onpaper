"""
Ollama AI Helper for Research Paper Assistant
Provides local AI capabilities using Ollama models.
"""

import os
import requests
from typing import Optional, Dict, Any

# Keep server-discovery probes short so a deploy with no local Ollama (e.g.
# Streamlit Community Cloud) isn't blocked for long the first time the backend is
# checked. Overridable via env for slower local setups.
_PROBE_TIMEOUT = float(os.getenv("OLLAMA_PROBE_TIMEOUT", "1.5"))
_GEN_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "60"))


class OllamaHelper:
    """Helper class for Ollama AI integration."""

    def __init__(self):
        # Single discovery pass: find a reachable server (if any) and record
        # availability in one shot, instead of probing the network three+ times.
        self.base_url, self.available = self._discover()
        self.model = (self._get_default_model() if self.available
                      else (os.getenv("OLLAMA_MODEL") or "llama3.2"))

    def _candidate_urls(self) -> list:
        """Ollama URLs to try. An explicit OLLAMA_BASE_URL wins (and is the only
        one tried); otherwise probe the usual local addresses."""
        url = os.getenv("OLLAMA_BASE_URL")
        if url:
            return [url]
        # For a non-default host, set OLLAMA_BASE_URL in .env rather than
        # hardcoding an IP here.
        return [
            "http://localhost:11434",             # Local
            "http://127.0.0.1:11434",             # Local alternative
            "http://host.docker.internal:11434",  # WSL/Docker bridge
        ]

    def _discover(self):
        """Return (base_url, available). One short-timeout probe per candidate;
        on a host with no Ollama this costs at most len(candidates)*_PROBE_TIMEOUT
        and never raises."""
        candidates = self._candidate_urls()
        for url in candidates:
            try:
                resp = requests.get(f"{url}/api/tags", timeout=_PROBE_TIMEOUT)
                if resp.status_code == 200:
                    return url, True
            except requests.RequestException:
                continue
        # Nothing reachable — fall back to the first candidate URL, marked down.
        return candidates[0], False

    def _get_ollama_url(self) -> str:
        """Backward-compat shim: the URL discovered at construction."""
        return self.base_url


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
        """Whether the server is reachable (determined once during discovery)."""
        return self.available

    def list_models(self) -> list:
        """List all available Ollama models."""
        if not self.available:
            return []

        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m.get('name') for m in data.get('models', []) if m.get('name')]
                return models
            return []
        except requests.RequestException as e:
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
                timeout=_GEN_TIMEOUT
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
