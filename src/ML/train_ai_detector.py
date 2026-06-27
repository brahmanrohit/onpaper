"""
Train a real AI-vs-Human text classifier on the HC3 dataset
(Human ChatGPT Comparison Corpus).

Uses only scikit-learn (already a project dependency). Produces:
    src/ML/ai_detector_model.pkl  -> {'pipeline': ..., 'metrics': {...}}

Run:  python src/ML/train_ai_detector.py
"""

import json
import pickle
import requests
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

HC3_URL = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl"
OUT_PATH = Path(__file__).parent / "ai_detector_model.pkl"
MIN_CHARS = 200  # skip very short answers that carry little signal


def download_hc3() -> list:
    """Download HC3 and return a list of (text, label) where label 1 = AI."""
    print("Downloading HC3 dataset...")
    resp = requests.get(HC3_URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    samples = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        for ans in obj.get("human_answers", []):
            if ans and len(ans) >= MIN_CHARS:
                samples.append((ans, 0))
        for ans in obj.get("chatgpt_answers", []):
            if ans and len(ans) >= MIN_CHARS:
                samples.append((ans, 1))
    print(f"Collected {len(samples)} samples "
          f"({sum(1 for _, y in samples if y == 0)} human, "
          f"{sum(1 for _, y in samples if y == 1)} AI).")
    return samples


def train():
    samples = download_hc3()
    if len(samples) < 1000:
        raise SystemExit("Not enough samples downloaded; aborting.")

    texts = [t for t, _ in samples]
    labels = [y for _, y in samples]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(max_iter=1000, C=4.0)),
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["human", "ai"])
    print(f"\nTest accuracy: {acc:.4f}\n")
    print(report)

    with open(OUT_PATH, "wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "metrics": {"accuracy": acc, "n_samples": len(samples)},
        }, f)
    print(f"Saved model -> {OUT_PATH}")


if __name__ == "__main__":
    train()
