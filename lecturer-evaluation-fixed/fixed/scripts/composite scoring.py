import pandas as pd
import ast
import joblib
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/preprocessed_evaluations.csv")
df["cleaned_text"] = df["cleaned_comment"].apply(lambda x: " ".join(ast.literal_eval(x)))
df["labels"] = df["labels"].apply(ast.literal_eval)

aspects = [
    "Clarity of Instruction",
    "Subject Knowledge",
    "Engagement",
    "Assessment",
    "Availability"
]

aspect_scores = []

for aspect in aspects:
    safe_name = aspect.replace(" ", "_").lower()
    model_path = f"models/svm_{safe_name}.pkl"
    vectorizer_path = f"models/vectorizer_{safe_name}.pkl"
    
    if not os.path.exists(model_path):
        continue
    
    svm = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    mask = df["labels"].apply(lambda x: any(l[0] == aspect for l in x))
    aspect_df = df[mask].copy()
    
    if len(aspect_df) == 0:
        continue
    
    X = vectorizer.transform(aspect_df["cleaned_text"])
    preds = svm.predict(X)
    
    for idx, pred in zip(aspect_df.index, preds):
        aspect_scores.append({
            "index": idx,
            "aspect": aspect,
            "sentiment": pred,
            "sentiment_score": 5 if pred == "positive" else 1
        })

scores_df = pd.DataFrame(aspect_scores)
pivot = scores_df.pivot_table(index="index", columns="aspect", values="sentiment_score", aggfunc="first")

df = df.join(pivot, how="left")

ratings = df["rating"].values
for aspect in aspects:
    if aspect in df.columns:
        df[f"{aspect}_cs"] = 0.4 * ratings + 0.6 * df[aspect].fillna(3)

cs_cols = [c for c in df.columns if c.endswith("_cs")]
df["overall_cs"] = df[cs_cols].mean(axis=1)

lecturer_summary = df.groupby("lecturer_name")[cs_cols + ["overall_cs", "rating"]].mean().round(2)
lecturer_summary.to_csv("data/lecturer_composite_scores.csv")

print(lecturer_summary)
print(f"\nSaved to data/lecturer_composite_scores.csv")