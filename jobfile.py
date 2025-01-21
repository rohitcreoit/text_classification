import joblib
import json

# Convert vectorizer
vectorizer = joblib.load("text_vectorizer.pkl")
vectorizer_dict = vectorizer.vocabulary_
with open("text_vectorizer.json", "w") as f:
    json.dump(vectorizer_dict, f)

# Convert label encoder
label_encoder = joblib.load("label_encoder.pkl")
label_classes = label_encoder.classes_.tolist()
with open("label_encoder.json", "w") as f:
    json.dump(label_classes, f)