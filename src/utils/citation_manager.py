import random

def suggest_citations(query):
    """Returns a list of mock research paper citations based on a query."""
    mock_papers = [
        {"title": "Deep Learning in Healthcare", "author": "Smith et al.", "year": 2021},
        {"title": "Blockchain for Data Security", "author": "Doe et al.", "year": 2020},
        {"title": "Advancements in AI", "author": "Brown et al.", "year": 2022},
        {"title": "Natural Language Processing Trends", "author": "Clark et al.", "year": 2023},
        {"title": "Quantum Computing and Cryptography", "author": "Taylor et al.", "year": 2019}
    ]
    return random.sample(mock_papers, min(len(mock_papers), 3))  # Return 3 random papers

def format_citation(paper, style="APA"):
    """Formats a citation according to the selected style."""
    if style == "APA":
        return f"{paper['author']} ({paper['year']}). {paper['title']}."
    elif style == "IEEE":
        return f"[{paper['year']}] {paper['author']}, \"{paper['title']}\"."
    elif style == "MLA":
        return f"{paper['author']}. \"{paper['title']}.\" {paper['year']}."
    return "Invalid citation style."
