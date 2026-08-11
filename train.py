import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

# Load datasets
with open("datasets/positives.txt", "r") as f:
    positives = f.read().splitlines()
with open("datasets/negatives.txt", "r") as f:
    negatives = f.read().splitlines()

# Embedder + classifier
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# data (positives + diverse negatives)
texts = positives + negatives
labels = [1]*len(positives) + [0]*len(negatives)

X_emb = embedder.encode(texts, show_progress_bar=True)

clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_emb, labels)

# Save model
joblib.dump((embedder, clf), 'detector_model.pkl')
print("Done")
