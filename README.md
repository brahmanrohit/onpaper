# OnPaper - Research Paper Writing Assistant

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Features](#features)
3. [Research Paper Types](#research-paper-types)
4. [Perfect Paper Guide](#perfect-paper-guide)
5. [Plagiarism Detection](#plagiarism-detection)
6. [Research Improvements](#research-improvements)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/onpaper.git
cd onpaper
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
# Option 1: Using the entry point script (recommended)
python run.py

# Option 2: Direct streamlit run
cd main
streamlit run main.py
```

## Project Structure

The project has been optimized with a clear, maintainable structure:

```
onpaperfixed/
├── app/                          # Main application (main.py)
├── core/                         # Core functionality
│   ├── models/                   # ML models
│   ├── utils/                    # Utility functions
│   └── config/                   # Configuration
├── training/                     # Model training scripts
├── data/                         # Data files
├── docs/                         # Documentation
├── tests/                        # Test files
├── requirements.txt              # Dependencies
└── run.py                       # Entry point
```

## Features
- **Topic-to-Type Prediction**: Enter any research topic and the app will automatically suggest the most appropriate research paper type (empirical, review, theoretical, etc.) using a trained AI classifier.
- **Content Generation**: Generate sections like Abstract, Introduction, Literature Review, and Methodology, or a full paper, tailored to the predicted or selected type.
- **Paper Analysis**: Extract text from PDFs.
- **Citation Assistant**: Get citation recommendations in APA, IEEE, or MLA format.
- **Grammar Check**: Detect and correct grammatical errors.
- **Plagiarism Detection**: Check for plagiarism using AI-based detection.

## Research Paper Types

OnPaper supports 10 main research paper types, each with customized structures:

1. **Empirical Research Paper**
   - Focus: Data analysis, statistics, empirical findings
   - Best for: Original research with data collection

2. **Theoretical Research Paper**
   - Focus: Theoretical concepts, frameworks
   - Best for: Developing new theories

3. **Review Paper**
   - Focus: Past studies, research gaps
   - Best for: Literature reviews and gap analysis

4. **Comparative Research Paper**
   - Focus: Comparison between subjects
   - Best for: Comparing approaches/methods

5. **Case Study**
   - Focus: Deep dive on specific subject
   - Best for: In-depth analysis of cases

6. **Analytical Research Paper**
   - Focus: Critical analysis, logical reasoning
   - Best for: Critical examination of topics

7. **Methodological Research Paper**
   - Focus: Method development, validation
   - Best for: New research methods

8. **Position Paper**
   - Focus: Clear position, arguments
   - Best for: Taking stances on issues

9. **Technical Report**
   - Focus: Technical details, implementation
   - Best for: Technical projects

10. **Interdisciplinary Research Paper**
    - Focus: Multiple disciplines, integration
    - Best for: Cross-disciplinary research

## Perfect Paper Guide

### The 10 Perfect Research Paper Rules

1. **Title** — Short, clear, keywords (under 15 words)
2. **Abstract** — Purpose, methods, results, conclusion (150–250 words)
3. **Introduction** — Topic, question, importance (500-800 words)
4. **Literature Review** — Past work, gaps (800-1200 words)
5. **Methodology** — Data collection/analysis (400-600 words)
6. **Results** — Clear data presentation (400-600 words)
7. **Discussion** — Interpretation, comparison (600-800 words)
8. **Conclusion** — Main takeaway, future work (300-500 words)
9. **References** — 5-8 relevant sources
10. **General** — Clean language, proper structure

### Word Count Guidelines

| Section | Standard Paper | Simple Paper |
|---------|----------------|--------------|
| Abstract | 150-250 words | 150-200 words |
| Introduction | 500-800 words | 400-600 words |
| Literature Review | 800-1200 words | - |
| Methodology | 400-600 words | 300-500 words |
| Results | 400-600 words | 300-500 words |
| Discussion | 600-800 words | 400-600 words |
| Conclusion | 300-500 words | 200-400 words |
| Total | ~3,500-5,000 words | ~2,000-3,000 words |

## Plagiarism Detection

### Features
- TF-IDF Vectorization
- Cosine Similarity Analysis
- Sentence-Level Detection
- Model Persistence
- Configurable Thresholds
- Reference Document Management
- Real-time Analysis
- Web Interface

### Usage
```python
from utils.text_analyzer import check_plagiarism, add_reference_document

# Add reference documents
add_reference_document("Your reference text here", "doc_id")

# Check for plagiarism
result = check_plagiarism("Text to check for plagiarism")
print(f"Plagiarism Score: {result['plagiarism_score']}%")
```

### Threshold Guidelines
- **0.1-0.3**: Very strict (high false positives)
- **0.4-0.6**: Moderate (balanced)
- **0.7-0.8**: Standard (recommended)
- **0.9-1.0**: Lenient (high false negatives)

## Research Improvements

### Key Improvements Made

1. **Verified Data Sources**
   - Integration with academic databases
   - Statistical data sources
   - Industry reports

2. **Defined Research Periods**
   - Clear timeframes
   - Historical context
   - Future projections

3. **Comprehensive Citations**
   - 15-20 high-quality sources
   - Multiple citation styles
   - Recent and classic papers

4. **Methodology Framework**
   - Qualitative and quantitative methods
   - Bias mitigation
   - Expert opinion integration

5. **Sample Specifications**
   - Statistical power analysis
   - Sample size calculations
   - Representativeness assessment

6. **Ethics Framework**
   - Informed consent
   - Data protection
   - IRB considerations
   - Conflict of interest disclosure

## 🚀 How to Use

### Option 1: Complete Paper
1. Go to "Content Generation" → "Complete Paper"
2. Enter your research topic
3. Choose paper type (or use AI suggestion)
4. Enable citations (recommended)
5. Click "Generate Paper"

### Option 2: Section Generator
1. Go to "Content Generation" → "Section Generator"
2. Enter topic and select section
3. Choose paper type
4. Click "Generate Section"

### Option 3: Quick Draft
1. Go to "Content Generation" → "Quick Draft"
2. Enter topic
3. Click "Generate Quick Draft"

## Developer Notes

### Retraining the Topic-to-Type Classifier
1. Add new papers to your dataset (see `core_api_collector.py`)
2. Run:
   ```bash
   python train_topic_type_classifier.py
   ```
3. This will update `topic_type_classifier.pkl` and `topic_type_vectorizer.pkl`

Enjoy writing research papers with AI assistance! 🚀
