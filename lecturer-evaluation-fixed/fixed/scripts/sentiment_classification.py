import pandas as pd
import numpy as np
import ast
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

df = pd.read_csv("data/preprocessed_evaluations.csv")

df["cleaned_text"] = df["cleaned_comment"].apply(lambda x: " ".join(ast.literal_eval(x)))
df["labels"] = df["labels"].apply(ast.literal_eval)

def build_aspect_dataset(df):
    rows = []
    for _, row in df.iterrows():
        text = row["cleaned_text"]
        for aspect, sentiment in row["labels"]:
            rows.append({
                "aspect": aspect,
                "text": text,
                "sentiment": sentiment
            })
    return pd.DataFrame(rows)

aspect_df = build_aspect_dataset(df)

aspects = [
    "Clarity of Instruction",
    "Subject Knowledge",
    "Engagement",
    "Assessment",
    "Availability"
]

os.makedirs("models", exist_ok=True)
results_summary = []

print("=" * 70)
print("LAYER 5: SVM vs NAIVE BAYES — ASPECT-LEVEL SENTIMENT CLASSIFICATION")
print("=" * 70)

for aspect in aspects:
    print(f"\n{'='*70}")
    print(f"ASPECT: {aspect.upper()}")
    print(f"{'='*70}")

    data = aspect_df[aspect_df["aspect"] == aspect].copy()

    if len(data) < 10:
        print(f"  [SKIP] Only {len(data)} samples — insufficient.")
        continue

    print(f"  Total samples: {len(data)}")
    print(f"  Class distribution:\n{data['sentiment'].value_counts().to_string().replace(chr(10), chr(10) + '    ')}")

    X = data["text"].values
    y = data["sentiment"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    svm = SVC(kernel="linear", C=1.0)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    svm_acc = accuracy_score(y_test, svm_pred)
    svm_precision, svm_recall, svm_f1, _ = precision_recall_fscore_support(
        y_test, svm_pred, average="weighted", zero_division=0
    )

    print(f"\n  --- SVM (Linear Kernel) ---")
    print(f"  Accuracy : {svm_acc:.4f}")
    print(f"  Precision: {svm_precision:.4f}")
    print(f"  Recall   : {svm_recall:.4f}")
    print(f"  F1-Score : {svm_f1:.4f}")
    print(f"\n  Detailed Report:")
    print(classification_report(y_test, svm_pred, zero_division=0))

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)

    nb_acc = accuracy_score(y_test, nb_pred)
    nb_precision, nb_recall, nb_f1, _ = precision_recall_fscore_support(
        y_test, nb_pred, average="weighted", zero_division=0
    )

    print(f"\n  --- Naive Bayes ---")
    print(f"  Accuracy : {nb_acc:.4f}")
    print(f"  Precision: {nb_precision:.4f}")
    print(f"  Recall   : {nb_recall:.4f}")
    print(f"  F1-Score : {nb_f1:.4f}")
    print(f"\n  Detailed Report:")
    print(classification_report(y_test, nb_pred, zero_division=0))

    safe_name = aspect.replace(" ", "_").lower()
    joblib.dump(svm, f"models/svm_{safe_name}.pkl")
    joblib.dump(nb, f"models/nb_{safe_name}.pkl")
    joblib.dump(vectorizer, f"models/vectorizer_{safe_name}.pkl")

    results_summary.append({
        "Aspect": aspect,
        "SVM_Acc": round(svm_acc, 4),
        "SVM_F1": round(svm_f1, 4),
        "NB_Acc": round(nb_acc, 4),
        "NB_F1": round(nb_f1, 4),
        "Winner_by_F1": "SVM" if svm_f1 > nb_f1 else "Naive Bayes" if nb_f1 > svm_f1 else "Tie"
    })

print("\n" + "=" * 70)
print("OVERALL COMPARISON: SVM vs NAIVE BAYES")
print("=" * 70)

summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

summary_df.to_csv("data/classification_results_summary.csv", index=False)
print(f"\nModels saved to /models/")
print(f"Summary saved to data/classification_results_summary.csv")