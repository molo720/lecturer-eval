import pandas as pd
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/preprocessed_evaluations.csv")

df["cleaned_text"] = df["cleaned_comment"].apply(lambda x: " ".join(literal_eval(x)))

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["cleaned_text"])

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

try:
    from gensim.models import Word2Vec
    tokenized_comments = df["cleaned_comment"].apply(literal_eval).tolist()
    w2v_model = Word2Vec(sentences=tokenized_comments, vector_size=100, window=5, min_count=1, workers=4)
    print(w2v_model.wv.most_similar("explain", topn=5))
except ModuleNotFoundError:
    print("Word2Vec skipped: gensim not installed (Python 3.14 compatibility issue).")