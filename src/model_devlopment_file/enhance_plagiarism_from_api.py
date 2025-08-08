import requests
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

API_KEY = input('Enter your CORE API key: ').strip()
BASE_URL = "https://api.core.ac.uk/v3"
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

def fetch_papers_from_api(topics, papers_per_topic=30):
    all_refs = []
    for topic in topics:
        print(f"Fetching papers for topic: {topic}")
        url = f"{BASE_URL}/search/works"
        payload = {"q": topic, "limit": papers_per_topic}
        try:
            response = requests.post(url, headers=HEADERS, json=payload)
            response.raise_for_status()
            papers = response.json().get('results', [])
            for paper in papers:
                abstract = paper.get('abstract', '')
                if len(abstract) > 50:
                    all_refs.append(abstract)
        except Exception as e:
            print(f"Error fetching papers for '{topic}': {e}")
    print(f"Total reference documents fetched: {len(all_refs)}")
    return all_refs

def save_enhanced_plagiarism_model(reference_docs, model_path='ML/enhanced_models_plagiarism.pkl'):
    vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words='english', min_df=1, max_df=0.95)
    if reference_docs:
        vectorizer.fit(reference_docs)
    enhanced_model = {
        'vectorizer': vectorizer,
        'reference_documents': reference_docs
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(enhanced_model, f)
    print(f"Enhanced plagiarism model saved with {len(reference_docs)} reference documents at {model_path}")

def main():
    topics = [
        "machine learning healthcare",
        "artificial intelligence education",
        "deep learning applications",
        "natural language processing",
        "computer vision",
        "data science business",
        "blockchain technology",
        "cybersecurity machine learning",
        "quantum computing",
        "robotics automation",
        "systematic review methodology",
        "theoretical framework development",
        "case study research",
        "comparative analysis methods",
        "analytical research methods"
    ]
    reference_docs = fetch_papers_from_api(topics, papers_per_topic=30)
    save_enhanced_plagiarism_model(reference_docs)

if __name__ == "__main__":
    main() 