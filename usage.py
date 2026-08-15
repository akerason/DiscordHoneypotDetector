import joblib

# Load the trained classifier
embedder, clf = joblib.load('detector_model.pkl')

def prediction(message):
    """
    0.0 - 0.2 -> Low chance
    0.2 - 0.4 -> Mid chance
    0.4 - 0.6 -> High chance
    0.6 - 1.0 -> Almost certain

    This return a number between 0 and 1, so that you can use a heuristic, based on score, to avoid false positives
    """
    emb = embedder.encode([message])
    prob = clf.predict_proba(emb)[0][1]
    return prob

# Example
msg = "wdym bro"
print(prediction(msg))
