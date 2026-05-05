# OnPaper Architecture - Comprehensive Explanation

## Overview

OnPaper is a **layered, modular architecture** built on Python, designed to separate concerns across multiple tiers: **Presentation (UI)**, **Business Logic (Processing)**, **ML/AI Services**, and **Utilities**. This architecture enables scalability, maintainability, and independent testing of components.

---

## 1. MODULAR STRUCTURE

### Directory Hierarchy

```
onpaperfixed/
├── main/                          # 🎨 PRESENTATION LAYER
│   ├── main.py                    # Streamlit UI - Feature orchestration
│   ├── deploy_plagiarism_model.py # Model deployment utilities
│   └── assets/                    # UI resources
│
├── src/                           # 🔧 BUSINESS LOGIC & SERVICES LAYER
│   ├── utils/                     # Core utilities (text processing, analysis)
│   │   ├── config.py              # 📋 Centralized configuration
│   │   ├── gemini_helper.py       # AI API integration
│   │   ├── pdf_processor.py       # Document extraction
│   │   ├── text_analyzer.py       # Plagiarism detection (ML)
│   │   ├── content_generator.py   # Content templates & generation
│   │   ├── topic_type_predictor.py# ML-based paper type classification
│   │   ├── grammar_checker.py     # Grammar validation (AI-powered)
│   │   ├── citation_manager.py    # Citation formatting
│   │   ├── nlp_utils.py           # NLP processing utilities
│   │   └── __pycache__/           # Python cache
│   │
│   └── ML/                        # 🤖 MACHINE LEARNING MODELS LAYER
│       ├── train_topic_type_classifier.py  # Model training script
│       ├── topic_type_classifier.pkl       # Trained classifier model
│       ├── topic_type_vectorizer.pkl       # Feature vectorizer
│       ├── enhanced_models_plagiarism.pkl  # Plagiarism detection model
│       ├── enhanced_models_paper_type.pkl  # Paper type classification
│       ├── plagiarism_model.pkl            # Alternative plagiarism model
│       └── training_report.json            # Training metrics & results
│
├── models/                        # 📦 MODEL ARTIFACTS (Persistent Storage)
│   └── __init__.py
│
├── processed_papers/              # 📚 DATA LAYER
│   ├── training_data/
│   │   ├── training_dataset.json  # Training data for ML models
│   │   ├── plagiarism_test_cases.json
│   │   └── core_training/
│   ├── test_data/                 # Test datasets
│   ├── validation_data/           # Validation datasets
│   └── raw_papers/                # Raw input documents
│
├── tests/                         # ✅ TESTING LAYER
│   ├── test_app.py                # Streamlit app tests
│   ├── test_imports.py            # Import validation
│   └── __init__.py
│
├── scripts/                       # 🚀 EXECUTION LAYER
│   ├── run_app.py                 # Alternative app launcher
│   └── __init__.py
│
├── docs/                          # 📖 DOCUMENTATION
│   └── MODEL_TUNING.md
│
├── requirements.txt               # Dependencies
├── run.py                         # 🔑 Entry Point
├── README.md                      # Project documentation
└── .env                           # Environment variables (secrets)
```

---

## 2. COMPONENT SEPARATION & RESPONSIBILITIES

### Layer 1: **PRESENTATION LAYER** (`main/`)
**Responsibility**: User Interface & Request Orchestration

```python
# main/main.py - Streamlit Application
import streamlit as st
from src.utils.pdf_processor import extract_text_from_pdf
from src.utils.text_analyzer import PlagiarismDetector
from src.utils.content_generator import generate_comprehensive_paper
from src.utils.grammar_checker import check_grammar_text

st.set_page_config(page_title="Research Paper Assistant")
st.title("📚 Research Paper Assistant")

# Menu-driven architecture
menu = ["Content Generation", "Paper Analysis", "Citation Assistant", "Grammar Check", "Plagiarism Detection"]
choice = st.sidebar.selectbox("Select a Feature", menu)

# Component interaction example:
if choice == "Content Generation":
    topic = st.text_input("Enter your research topic")
    paper_type = st.selectbox("Select paper type")
    
    if st.button("Generate Paper"):
        # Calls business logic layer
        paper = generate_comprehensive_paper(topic, paper_type)
        st.success("Paper generated!")
```

**Key Characteristics**:
- Pure UI orchestration (no business logic)
- Calls utility functions and services
- Handles user input/output
- Manages state with Streamlit session

---

### Layer 2: **BUSINESS LOGIC & SERVICES LAYER** (`src/utils/`)
**Responsibility**: Core functionality, text processing, ML integration

#### **2A. Data Processing & Analysis** (`text_analyzer.py`)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class PlagiarismDetector:
    """Detects plagiarism using TF-IDF + Cosine Similarity"""
    
    def __init__(self, model_path=None, vectorizer_path=None):
        self.vectorizer = TfidfVectorizer()  # Feature extraction
        self.reference_documents = []         # Knowledge base
        self.model = None
        self.load_model()
    
    def preprocess_text(self, text: str) -> str:
        """Normalize text: lowercase, remove special chars"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def add_reference_document(self, text: str, doc_id: str):
        """Add comparison baseline"""
        processed = self.preprocess_text(text)
        self.reference_documents.append({
            'id': doc_id,
            'text': processed,
            'sentences': self.extract_sentences(text)
        })
    
    def check_plagiarism(self, document: str) -> Dict:
        """Compare document against reference base"""
        # 1. Vectorize both documents
        # 2. Calculate cosine similarity
        # 3. Identify matching segments
        # 4. Return similarity score + matches
```

**Data Flow**:
```
User Input (Text) 
    ↓
preprocess_text() [Normalization]
    ↓
TfidfVectorizer [Feature Extraction]
    ↓
cosine_similarity() [Comparison]
    ↓
Dict {score, matches, segments} [Output]
```

---

#### **2B. AI-Powered Services** (`gemini_helper.py`)

```python
import google.generativeai as genai

# Centralized API configuration
def load_env_file():
    """Locate and load .env from multiple paths"""
    # Tries: current dir → parent → project root → home
    
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-flash-latest")

def generate_text(prompt: str) -> str:
    """Unified interface for Gemini API calls"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"
```

**Design Pattern**: **Adapter Pattern**
- Single point of API integration
- Encapsulates API complexity
- Enables easy switching to alternative providers

---

#### **2C. Content Generation** (`content_generator.py`)

```python
class ResearchContentGenerator:
    """Generates content using templates + AI"""
    
    def __init__(self):
        self.research_templates = self._load_research_templates()
        # Template structure:
        # {
        #     "empirical": {
        #         "structure": ["Title", "Abstract", "Introduction", ...],
        #         "word_limits": {"Abstract": "150-250", ...},
        #         "focus": "Data analysis, statistics..."
        #     },
        #     ...
        # }
    
    def generate_comprehensive_paper(self, topic, paper_type, suggestions):
        """
        1. Lookup template for paper_type
        2. For each section:
           - Use template + topic as context
           - Call Gemini API to generate content
        3. Aggregate sections + citations
        4. Return complete paper structure
        """
        template = self.research_templates[paper_type]
        paper = {}
        
        for section in template['structure']:
            prompt = f"Write {section} for: {topic}"
            content = generate_text(prompt)  # Calls gemini_helper
            paper[section] = content
        
        return paper
```

---

#### **2D. ML Classification** (`topic_type_predictor.py`)

```python
class TopicTypePredictor:
    """Maps research topics to paper types"""
    
    def __init__(self):
        # Load pre-trained models
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)  # e.g., LogisticRegression
        with open(VECTORIZER_PATH, 'rb') as f:
            self.vectorizer = pickle.load(f)  # e.g., TfidfVectorizer
    
    def predict(self, topic: str) -> Tuple[str, float]:
        """
        Input: "Machine learning in healthcare"
        Process:
            1. topic → vectorizer → feature vector
            2. feature vector → model.predict() → class label
            3. model.predict_proba() → confidence score
        Output: ("empirical", 0.87)
        """
        X = self.vectorizer.transform([topic])
        prediction = self.model.predict(X)[0]
        confidence = max(self.model.predict_proba(X)[0])
        return prediction, confidence
```

---

#### **2E. Grammar & Language Processing** (`grammar_checker.py`)

```python
class GrammarChecker:
    """AI-powered grammar checking"""
    
    def check_grammar(self, text: str) -> Dict:
        """
        1. Validate text is not empty
        2. Create detailed prompt for Gemini
        3. Call gemini_helper.generate_text()
        4. Parse JSON response
        5. Return structured corrections
        """
        prompt = self._create_grammar_check_prompt(text)
        response = generate_text(prompt)  # Calls gemini_helper
        result = self._parse_gemini_response(text, response)
        return result
```

**Response Structure**:
```json
{
    "corrected_text": "...",
    "changes": [
        {
            "type": "Grammar|Spelling|Style",
            "message": "description",
            "original": "...",
            "corrected": "...",
            "severity": "error|warning|info"
        }
    ],
    "statistics": {
        "total_errors": 3,
        "grammar_errors": 1,
        "spelling_errors": 2
    }
}
```

---

#### **2F. Citation Management** (`citation_manager.py`)

```python
def suggest_citations(query: str) -> List[Dict]:
    """
    Could be:
    - Mock data (current)
    - API call to research DB (Semantic Scholar, CrossRef)
    - Database lookup
    """
    return [
        {"title": "...", "author": "...", "year": 2023},
        ...
    ]

def format_citation(paper: Dict, style: str) -> str:
    """Multi-format support: APA, IEEE, MLA"""
    if style == "APA":
        return f"{paper['author']} ({paper['year']}). {paper['title']}."
    elif style == "IEEE":
        return f"[{paper['year']}] {paper['author']}, \"{paper['title']}\"."
    elif style == "MLA":
        return f"{paper['author']}. \"{paper['title']}.\" {paper['year']}."
```

---

### Layer 3: **CONFIGURATION & SETTINGS** (`src/utils/config.py`)
**Responsibility**: Centralized configuration management

```python
"""
Configuration hub - Single source of truth for all settings
"""
import os
from pathlib import Path

# BASE PATHS
BASE_DIR = Path(__file__).parent.parent.parent
ML_DIR = BASE_DIR / 'src' / 'ML'

# ML MODEL PATHS
PLAGIARISM_MODEL_PATH = ML_DIR / 'enhanced_models_plagiarism.pkl'
PLAGIARISM_VECTORIZER_PATH = ML_DIR / 'enhanced_models_vectorizer.pkl'
PAPER_TYPE_MODEL_PATH = ML_DIR / 'enhanced_models_paper_type.pkl'
TOPIC_TYPE_MODEL_PATH = ML_DIR / 'topic_type_classifier.pkl'
TOPIC_TYPE_VECTORIZER_PATH = ML_DIR / 'topic_type_vectorizer.pkl'

# APPLICATION SETTINGS
DEFAULT_PLAGIARISM_THRESHOLD = 0.7  # Similarity threshold
DEFAULT_SUMMARY_SENTENCES = 3
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# PAPER TYPES REGISTRY
PAPER_TYPES = {
    "empirical": "Empirical Research Paper (Data-based)",
    "theoretical": "Theoretical Research Paper",
    "review": "Review Paper (Literature or Systematic Review)",
    "comparative": "Comparative Research Paper",
    "case_study": "Case Study",
    "analytical": "Analytical Research Paper",
    "methodological": "Methodological Research Paper",
    "position": "Position Paper / Opinion Paper",
    "technical": "Technical Report",
    "interdisciplinary": "Interdisciplinary Research Paper"
}

# CITATION STYLES
CITATION_STYLES = ["APA", "IEEE", "MLA"]

# FILE HANDLING
ALLOWED_FILE_TYPES = ["pdf", "txt"]

# ✅ BENEFITS OF CENTRALIZED CONFIG:
# 1. Change thresholds in one place
# 2. Add/remove paper types globally
# 3. Update model paths without touching code
# 4. Environment-specific overrides (dev vs prod)
# 5. Consistent settings across all modules
```

**Why This Matters**:
```python
# ❌ BAD: Hardcoded in multiple files
# text_analyzer.py
THRESHOLD = 0.7
# content_generator.py
THRESHOLD = 0.7  # When to update? Inconsistency!

# ✅ GOOD: Single source of truth
from src.utils.config import DEFAULT_PLAGIARISM_THRESHOLD
THRESHOLD = DEFAULT_PLAGIARISM_THRESHOLD  # Always synchronized
```

---

### Layer 4: **ML MODELS & ARTIFACTS** (`src/ML/`)
**Responsibility**: Trained models and training infrastructure

```
src/ML/
├── topic_type_classifier.pkl         # Model for topic → paper type
├── topic_type_vectorizer.pkl         # Feature extractor for topics
├── enhanced_models_plagiarism.pkl    # Plagiarism detection model
├── enhanced_models_paper_type.pkl    # Paper type classifier
├── plagiarism_model.pkl              # Alternative implementation
├── training_report.json              # Metrics: accuracy, precision, recall
└── train_topic_type_classifier.py    # Training pipeline
```

**Model Loading Pattern**:
```python
# In config.py
TOPIC_TYPE_MODEL_PATH = ML_DIR / 'topic_type_classifier.pkl'
TOPIC_TYPE_VECTORIZER_PATH = ML_DIR / 'topic_type_vectorizer.pkl'

# In topic_type_predictor.py
class TopicTypePredictor:
    def __init__(self):
        with open(TOPIC_TYPE_MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        with open(TOPIC_TYPE_VECTORIZER_PATH, 'rb') as f:
            self.vectorizer = pickle.load(f)
```

---

### Layer 5: **DATA LAYER** (`processed_papers/`)
**Responsibility**: Training, validation, and test data

```
processed_papers/
├── training_data/
│   ├── training_dataset.json         # ~1000 examples for training
│   ├── plagiarism_test_cases.json    # Plagiarism scenarios
│   └── core_training/
│       └── core_dataset.json         # Core examples
├── test_data/                        # Holdout test set (20% of data)
├── validation_data/                  # Validation set (10% of data)
└── raw_papers/
    ├── sample_healthcare_paper.txt   # Reference documents
    └── core_papers/
```

**Data Split Example**:
```
Total: 1000 examples
├── Training: 800 (80%) → Used to train models
├── Validation: 100 (10%) → Used during training to tune hyperparameters
└── Testing: 100 (10%) → Used to evaluate final performance
```

---

### Layer 6: **TESTING LAYER** (`tests/`)
**Responsibility**: Validation & quality assurance

```python
# tests/test_imports.py
"""Validate all modules can be imported"""
def test_imports():
    from src.utils.text_analyzer import PlagiarismDetector
    from src.utils.content_generator import ResearchContentGenerator
    from src.utils.topic_type_predictor import TopicTypePredictor
    from src.utils.grammar_checker import GrammarChecker
    assert True  # All imports successful

# tests/test_app.py
"""Test Streamlit application flows"""
def test_plagiarism_detection():
    detector = PlagiarismDetector()
    detector.add_reference_document("sample text", "doc1")
    result = detector.check_plagiarism("similar text")
    assert 'score' in result
    assert 0 <= result['score'] <= 1
```

---

## 3. DATA FLOW THROUGH THE SYSTEM

### Flow 1: Content Generation Pipeline

```
USER INPUT (main.py)
    ↓
topic = "Machine Learning in Healthcare"
paper_type = "empirical"
suggestions = "focus on healthcare applications"
    ↓
[PRESENTATION LAYER]
st.button("Generate Paper") triggers:
    ↓
generate_comprehensive_paper(topic, paper_type, suggestions)
    ↓
[BUSINESS LOGIC LAYER - content_generator.py]
1. Load template: PAPER_TYPES[paper_type]
2. Get structure: ["Title", "Abstract", "Introduction", ...]
3. For each section:
   - Create prompt: f"Write {section} for {topic}..."
   - Call gemini_helper.generate_text(prompt)
   ↓
[AI SERVICE LAYER - gemini_helper.py]
generate_text(prompt)
    ↓
    gemini.GenerativeModel("gemini-flash-latest")
    model.generate_content(prompt)
    ↓
[GOOGLE CLOUD - Gemini API]
    ↓
[RESPONSE]
Generated content for section
    ↓
[BUSINESS LOGIC LAYER - content_generator.py]
4. Format response: {"section": content}
5. Add citations: suggest_citations(topic)
6. Return complete paper:
   {
       "title": "...",
       "sections": {...},
       "citations": [...]
   }
    ↓
[PRESENTATION LAYER - main.py]
Display paper sections
Download as Word/PDF
    ↓
USER OUTPUT
```

---

### Flow 2: Plagiarism Detection Pipeline

```
USER INPUT (main.py)
    ↓
uploaded_file = user_document.pdf
    ↓
[PRESENTATION LAYER]
st.file_uploader() → st.button("Check Plagiarism")
    ↓
text = extract_text_from_pdf(uploaded_file)
    ↓
[DATA PROCESSING LAYER - pdf_processor.py]
Read PDF → Extract text → Return string
    ↓
detector = PlagiarismDetector()
detector.add_reference_document(reference_text, "reference1")
result = detector.check_plagiarism(text)
    ↓
[BUSINESS LOGIC LAYER - text_analyzer.py]

PlagiarismDetector.check_plagiarism():
    1. Preprocess text:
       - Lowercase
       - Remove special characters
       - Normalize whitespace
    ↓
    2. Extract sentences using NLTK
    ↓
    3. Vectorize using TF-IDF:
       text → TfidfVectorizer → feature_vector
    ↓
    4. Compare against reference documents:
       similarity = cosine_similarity(
           user_vector,
           reference_vectors
       )
    ↓
    5. Identify matching segments:
       For each sentence pair with similarity > THRESHOLD:
           - Record matching text
           - Calculate segment score
    ↓
    6. Aggregate results:
       overall_score = average(segment_scores)
       matching_segments = [...]
    ↓
    Return: {
        'score': 0.65,
        'matches': [...],
        'plagiarism_detected': True,
        'threshold': 0.7
    }
    ↓
[PRESENTATION LAYER - main.py]
Display:
    - Overall plagiarism score
    - Highlighted matching segments
    - Recommendation: "Check sources"
    ↓
USER OUTPUT
```

---

### Flow 3: Paper Type Classification Pipeline

```
USER INPUT (main.py)
    ↓
topic = "Sentiment analysis in social media"
    ↓
[PRESENTATION LAYER]
st.text_input("Enter topic")
predictor = TopicTypePredictor()
paper_type, confidence = predictor.predict(topic)
    ↓
[ML LAYER - topic_type_predictor.py]

TopicTypePredictor.predict(topic):
    1. Load pre-trained components:
       - self.model (e.g., LogisticRegression)
       - self.vectorizer (TfidfVectorizer)
    ↓
    2. Vectorize topic:
       topic_vector = self.vectorizer.transform([topic])
       → sparse matrix (1, num_features)
    ↓
    3. Predict class:
       prediction = self.model.predict(topic_vector)[0]
       → "empirical"
    ↓
    4. Get confidence:
       probabilities = self.model.predict_proba(topic_vector)[0]
       confidence = max(probabilities)
       → 0.87
    ↓
    Return: ("empirical", 0.87)
    ↓
[CONFIGURATION LAYER - config.py]
paper_type_name = PAPER_TYPES["empirical"]
→ "Empirical Research Paper (Data-based)"
    ↓
[PRESENTATION LAYER - main.py]
Display: "Suggested type: Empirical (87% confidence)"
Allow user to override or accept
    ↓
USER OUTPUT
```

---

### Flow 4: Grammar Checking Pipeline

```
USER INPUT (main.py)
    ↓
text = "The papers was submitted in 2023."
    ↓
[PRESENTATION LAYER]
st.text_area("Paste text")
st.button("Check Grammar")
result = check_grammar_text(text)
    ↓
[BUSINESS LOGIC LAYER - grammar_checker.py]

GrammarChecker.check_grammar(text):
    1. Validate input:
       if not text.strip(): return {errors: 0}
    ↓
    2. Create detailed prompt for Gemini:
       prompt = f"""
           You are a professional grammar checker.
           Text to check: "{text}"
           Respond with JSON: {{
               "corrected_text": "...",
               "changes": [{{
                   "type": "Grammar|Spelling",
                   "message": "...",
                   "original": "The papers was",
                   "corrected": "The paper was",
                   "severity": "error"
               }}],
               "statistics": {{
                   "total_errors": 1,
                   "grammar_errors": 1
               }}
           }}
       """
    ↓
    3. Call Gemini API:
       response = generate_text(prompt)
       ↓
       [AI SERVICE LAYER - gemini_helper.py]
       model.generate_content(prompt)
       ↓
       Returns: Corrected text + structured changes
    ↓
    4. Parse JSON response:
       result = json.loads(response)
    ↓
    5. Create output dict:
       {
           'corrected_text': 'The paper was submitted in 2023.',
           'changes': [{...}],
           'statistics': {
               'total_errors': 1,
               'grammar_errors': 1,
               'spelling_errors': 0,
               'style_errors': 0
           }
       }
    ↓
[PRESENTATION LAYER - main.py]
Display:
    - Side-by-side comparison
    - Highlighted corrections
    - Error statistics
    ↓
USER OUTPUT
```

---

## 4. CONFIGURATION ROLE (config.py)

### Purpose: Centralized Settings Hub

```python
# config.py acts as a single source of truth

# ✅ BENEFIT 1: Model Path Management
from src.utils.config import PLAGIARISM_MODEL_PATH

# If you need to update model location:
# Change only in config.py
# All other files automatically use new path

# ✅ BENEFIT 2: Feature Flags
DEFAULT_PLAGIARISM_THRESHOLD = 0.7  # Change once, affects entire system

# ✅ BENEFIT 3: Paper Type Registry
PAPER_TYPES = {
    "empirical": "...",
    "theoretical": "...",
    # Add new type here, automatically available everywhere
}

# ✅ BENEFIT 4: Multi-Environment Support
# Can load different configs for dev/staging/prod:
if ENVIRONMENT == "production":
    DEFAULT_PLAGIARISM_THRESHOLD = 0.75
    MAX_FILE_SIZE = 5 * 1024 * 1024
else:
    DEFAULT_PLAGIARISM_THRESHOLD = 0.7
    MAX_FILE_SIZE = 100 * 1024 * 1024

# ✅ BENEFIT 5: API Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")  # Read from .env
ALLOWED_FILE_TYPES = ["pdf", "txt"]     # File constraints
CITATION_STYLES = ["APA", "IEEE", "MLA"]  # Available options
```

### Architecture Benefits

```
WITHOUT Centralized Config:
┌─────────────────┐
│  text_analyzer  │
│  THRESHOLD=0.7  │
└─────────────────┘

┌─────────────────┐
│ content_gen     │
│ THRESHOLD=0.7   │
└─────────────────┘

┌─────────────────┐
│  grammar_check  │
│ MODEL_PATH=...  │
└─────────────────┘
❌ Hard to maintain, inconsistencies


WITH Centralized Config:
┌─────────────────────────────────────────┐
│  config.py (SINGLE SOURCE OF TRUTH)     │
│  ├─ THRESHOLD = 0.7                     │
│  ├─ MODEL_PATH = ...                    │
│  ├─ PAPER_TYPES = {...}                 │
│  ├─ API_KEY = ...                       │
│  └─ MAX_FILE_SIZE = ...                 │
└─────────────────────────────────────────┘
       ↑         ↑         ↑
       │         │         │
┌──────┴──┐ ┌────┴────┐ ┌──┴───────┐
│  text   │ │ content │ │ grammar  │
│analyzer │ │  gen    │ │  check   │
└─────────┘ └─────────┘ └──────────┘
✅ Consistency, maintainability, scalability
```

---

## 5. COMPONENT INTERACTION EXAMPLE: Complete Workflow

### Scenario: User generates a paper on "AI in Healthcare"

```
┌────────────────────────────────────────────────────────────────┐
│ 1. USER SUBMITS REQUEST (Presentation Layer)                  │
│    Topic: "AI in Healthcare"                                  │
│    Paper Type: Auto-detect                                    │
│    Suggestions: "Focus on clinical applications"              │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. PAPER TYPE CLASSIFICATION (ML Layer)                        │
│    TopicTypePredictor.predict("AI in Healthcare")              │
│    └─ Vectorize topic                                          │
│    └─ Load model from config.TOPIC_TYPE_MODEL_PATH             │
│    └─ Predict: ("empirical", 0.92)                             │
│    └─ Lookup: PAPER_TYPES["empirical"]                         │
│       = "Empirical Research Paper (Data-based)"                │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. LOAD TEMPLATE (Business Logic Layer)                        │
│    ResearchContentGenerator._load_research_templates()         │
│    └─ Get structure for "empirical" paper                      │
│    └─ Structure:                                               │
│       ["Title", "Abstract", "Introduction",                    │
│        "Literature Review", "Methodology",                     │
│        "Results", "Discussion", "Conclusion"]                  │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. GENERATE EACH SECTION (AI Service + Config)                 │
│    For each section in structure:                              │
│    ├─ Create prompt: "Write Methodology for AI in Healthcare.."│
│    ├─ Call gemini_helper.generate_text(prompt)                 │
│    │  └─ Uses API_KEY from config                              │
│    │  └─ Calls Gemini API                                      │
│    │  └─ Returns content                                       │
│    ├─ Validate word limit from config.WORD_LIMITS              │
│    └─ Aggregate: paper["Methodology"] = content                │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. ADD CITATIONS (Citation Manager)                            │
│    suggest_citations("AI in Healthcare")                       │
│    └─ Returns relevant papers                                  │
│    └─ format_citation(paper, style="APA")                      │
│       Uses CITATION_STYLES from config                         │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 6. CHECK GRAMMAR (Grammar Checker + AI)                        │
│    check_grammar_text(paper_text)                              │
│    └─ Create detailed grammar check prompt                     │
│    └─ Call gemini_helper.generate_text()                       │
│    └─ Parse corrections                                        │
│    └─ Return corrected paper with change tracking              │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 7. OPTIONAL: PLAGIARISM CHECK (Text Analyzer + ML)             │
│    detector = PlagiarismDetector()                             │
│    └─ Load models from config.PLAGIARISM_MODEL_PATH            │
│    └─ Add reference docs                                       │
│    └─ Run check_plagiarism(paper_text)                         │
│    └─ Uses threshold from config.DEFAULT_PLAGIARISM_THRESHOLD  │
│    └─ Returns: {score, matches, recommendation}                │
└────────────────────────────────────────────────────────────────┘
                          ↓
┌────────────────────────────────────────────────────────────────┐
│ 8. RETURN TO USER (Presentation Layer)                         │
│    ✅ Display complete paper                                   │
│    ✅ Show quality metrics                                     │
│    ✅ Download options (Word, PDF, TXT)                        │
│    ✅ Export as zipfile with assets                            │
└────────────────────────────────────────────────────────────────┘
```

---

## 6. KEY ARCHITECTURAL PATTERNS

| Pattern | Implementation | Benefit |
|---------|---|---|
| **Layered Architecture** | main/ → src/utils/ → src/ML/ | Separation of concerns |
| **Adapter Pattern** | gemini_helper wraps Google API | Easy API switching |
| **Factory Pattern** | Paper type selection from PAPER_TYPES | Extensible paper types |
| **Template Method** | content_generator uses templates | Consistent structure |
| **Singleton Pattern** | PlagiarismDetector loaded once | Efficient resource use |
| **Pipeline Pattern** | text → preprocess → vectorize → compare | Clear processing flow |
| **Configuration Pattern** | config.py centralization | Single source of truth |

---

## 7. STRENGTHS & LIMITATIONS

### ✅ Strengths
- **Modular**: Each component has single responsibility
- **Testable**: Utils can be tested independently
- **Maintainable**: Clear folder structure and imports
- **Scalable**: Easy to add new paper types or models
- **Configurable**: Centralized settings management
- **Flexible**: Can switch APIs or ML models easily

### ⚠️ Limitations
- **No Database**: Data persisted only as files/pickles
- **Synchronous**: No async for long-running tasks
- **Single User**: No multi-user support or auth
- **Memory Intensive**: Full models loaded at startup
- **No Caching**: Every request recomputes
- **Limited Error Handling**: Some components lack fallbacks

---

## Summary

The OnPaper architecture follows a **layered, modular design** where:
1. **UI Layer** (Streamlit) orchestrates user requests
2. **Business Logic Layer** (utils/) implements core features
3. **ML Layer** (src/ML/) provides trained models
4. **Config Layer** centralizes all settings
5. **Data Layer** stores training/test data

This design enables **independent development**, **easy testing**, and **flexible deployment** while maintaining clear data flow and component boundaries.

