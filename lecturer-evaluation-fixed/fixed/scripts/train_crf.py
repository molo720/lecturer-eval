import pandas as pd
from ast import literal_eval
import sklearn_crfsuite

def word_to_features(tokens, i):
    word = tokens[i]
    
    shape = ""
    for c in word:
        if c.isupper():
            shape += "X"
        elif c.islower():
            shape += "x"
        elif c.isdigit():
            shape += "d"
        else:
            shape += c
    
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:2]': word[:2],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.shape': shape,
        'BOS': i == 0,
        'EOS': i == len(tokens) - 1,
    }
    
    if i > 0:
        prev = tokens[i-1]
        features['prev_word.lower()'] = prev.lower()
        features['prev_word.istitle()'] = prev.istitle()
    else:
        features['BOS'] = True

    if i < len(tokens) - 1:
        nxt = tokens[i+1]
        features['next_word.lower()'] = nxt.lower()
        features['next_word.istitle()'] = nxt.istitle()
    else:
        features['EOS'] = True

    return features

def sentence_to_features(tokens):
    return [word_to_features(tokens, i) for i in range(len(tokens))]

df = pd.read_csv("data/crf_training_data.csv")

df["tokens"] = df["tokens"].apply(literal_eval)
df["bio_tags"] = df["bio_tags"].apply(literal_eval)

X = [sentence_to_features(tokens) for tokens in df["tokens"]]
y = df["bio_tags"].tolist()

from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100)
crf.fit(X_train, y_train)

y_pred = crf.predict(X_test)

from sklearn_crfsuite import metrics

print("NOTE: F1 ≈ 1.00 expected on synthetic template data. See Chapter 5: Limitations.")

print(metrics.flat_classification_report(y_test, y_pred))

import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(crf, "models/crf_aspect_extractor.pkl")
print("CRF model saved to models/crf_aspect_extractor.pkl")
