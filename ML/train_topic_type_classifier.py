import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Load data
with open('processed_papers/training_data/core_training/core_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

papers = data['papers']

# Prepare data: use title + abstract as input, paper_type as label
texts = []
labels = []
for paper in papers:
    topic = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    paper_type = paper.get('paper_type', 'empirical')
    if topic.strip() and paper_type:
        texts.append(topic)
        labels.append(paper_type)

# Vectorize
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(texts)
y = np.array(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Topic-to-Type Classifier Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
with open('topic_type_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('topic_type_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Topic-to-type classifier and vectorizer saved!") 