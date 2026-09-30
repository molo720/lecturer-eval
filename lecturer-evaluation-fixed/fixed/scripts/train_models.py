import os
import random
import json
import ast
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

# Import preprocessing
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import clean_text, preprocess_for_vectorizer

random.seed(42)
np.random.seed(42)

# ==========================================
# 1. DEFINE ALL 8 TEACHING ASPECTS & PHRASES
# (Section 3.5.2 & 3.6.2 of Thesis)
# ==========================================
aspect_templates = {
    "Teaching Clarity": {
        "positive": [
            "Explains complex concepts with clarity and precision.",
            "Breaks down difficult topics into digestible steps.",
            "Makes abstract ideas very easy to understand.",
            "Every lecture explanation is logical and clear.",
            "Uses practical analogies that simplify technical concepts.",
            "Delivers structured and comprehensible lectures every time.",
            "Clarifies every question thoroughly until we grasp the topic.",
            "The explanations are articulate and straightforward to follow.",
            "Brilliant teaching clarity; makes challenging algorithms look simple."
        ],
        "negative": [
            "Explanations are confusing and lack structure.",
            "Rushes through difficult slides without proper explanation.",
            "Complicates straightforward concepts unnecessarily.",
            "Speaks too fast and jumps between topics erratically.",
            "Difficult to follow what the lecturer is trying to convey.",
            "Lacks pedagogical clarity; students leave the hall confused.",
            "Uses dense jargon without breaking down fundamentals.",
            "Explanations are vague and leave too many unanswered gaps.",
            "Fails to explain core principles clearly."
        ],
        "neutral": [
            "Lectures follow the textbook explanations adequately.",
            "The explanations are standard and neither exciting nor unclear.",
            "Coverage of the syllabus was delivered as outlined.",
            "Explanations are moderate and aligned with the course guide.",
            "Lecture delivery is acceptable for introductory topics."
        ]
    },
    "Course Organisation": {
        "positive": [
            "Course is exceptionally organized and follows the syllabus perfectly.",
            "The lecture schedule and weekly milestones are well planned.",
            "Maintains a clear course outline from week one to final exams.",
            "Course structure is cohesive and modules transition smoothly.",
            "Outstanding planning; topics progress logically and systematically.",
            "The timetable and course pace are managed impeccably."
        ],
        "negative": [
            "The course is poorly organized with no clear syllabus.",
            "Course outline was never provided and topics were disjointed.",
            "Disorganized scheduling; classes were shifted erratically.",
            "The syllabus pace was rushed heavily in the final weeks.",
            "Lacks structure; we jumped from chapter five back to chapter two.",
            "Total lack of course organization throughout the entire semester."
        ],
        "neutral": [
            "The course follows the institutional standard syllabus.",
            "Course topics were addressed according to the departmental calendar.",
            "The syllabus pacing was fairly typical of university modules.",
            "Module sequence followed the assigned curriculum."
        ]
    },
    "Assessment and Grading": {
        "positive": [
            "Grading is fair, objective, and transparently communicated.",
            "Continuous assessments directly test what was taught in class.",
            "Provides detailed rubrics and constructive feedback on scripts.",
            "Exams were well-balanced and marked with complete fairness.",
            "Returns test scores quickly with helpful correction notes.",
            "Grading criteria are explicit and applied equitably to all students."
        ],
        "negative": [
            "Grading is arbitrary, inconsistent, and unfairly penalized.",
            "Test questions covered material that was never taught in class.",
            "Continuous assessment scripts were never returned all semester.",
            "Grading rubrics are completely absent and marks appear random.",
            "Exams were unreasonably difficult and far beyond the syllabus.",
            "Vague feedback on assignments; nobody knows why they lost marks."
        ],
        "neutral": [
            "Assessments adhered to standard departmental grading scales.",
            "Two tests and one exam were administered as scheduled.",
            "Exam difficulty reflected the average level of past papers.",
            "Marks were computed according to normal university regulations."
        ]
    },
    "Lecturer Punctuality": {
        "positive": [
            "Always punctual and starts every class right on time.",
            "Consistently arrives early and utilizes the full lecture period.",
            "Exemplary punctuality and regular attendance all semester.",
            "Never misses a lecture and values students' scheduled time.",
            "Punctual arrival to 8 AM lectures without exception."
        ],
        "negative": [
            "Habitually late to lectures and wastes valuable class time.",
            "Arrives thirty to forty minutes behind schedule repeatedly.",
            "Frequently cancels lectures without giving prior notice.",
            "Missed multiple lecture sessions without making up the hours.",
            "Very poor punctuality; classes start whenever he feels like it."
        ],
        "neutral": [
            "Lecturer arrives roughly on time for most scheduled classes.",
            "Attendance was generally consistent with occasional slight delays.",
            "Lectures started within reasonable university grace periods.",
            "Session timing was generally maintained."
        ]
    },
    "Availability": {
        "positive": [
            "Always available during office hours for student consultation.",
            "Very approachable and willing to assist students after hours.",
            "Welcoming to students seeking help with difficult course projects.",
            "Office doors are always open for academic guidance.",
            "Readily accessible and makes time to mentor struggling learners."
        ],
        "negative": [
            "Virtually impossible to locate outside scheduled lecture hours.",
            "Never available during posted office hours; doors always locked.",
            "Dismissive and impatient when students approach for assistance.",
            "Unavailable to provide guidance on term papers and projects.",
            "Students cannot reach the lecturer for urgent academic inquiries."
        ],
        "neutral": [
            "Available primarily during designated departmental consultation hours.",
            "Office hours are kept when requested in advance.",
            "Consultation is available within standard university hours.",
            "Meeting the lecturer requires scheduling an appointment."
        ]
    },
    "Communication": {
        "positive": [
            "Communicates announcements clearly, proactively, and timely.",
            "Responds promptly to student emails with helpful guidance.",
            "Keeps students well informed about assignment changes and events.",
            "Professional and courteous communication across all channels.",
            "Clear verbal and written communication throughout the semester."
        ],
        "negative": [
            "Fails to respond to student emails and academic inquiries.",
            "Announcements are confusing, contradictory, and last-minute.",
            "Poor communication; students were uninformed about test venues.",
            "Ignores emails regarding coursework clarifications.",
            "Communication channel is disorganized and unreliable."
        ],
        "neutral": [
            "Class announcements were relayed through the class representative.",
            "Communication was limited to official notice boards and portals.",
            "Email responses take two to three business days on average.",
            "Standard course notices were shared periodically."
        ]
    },
    "Student Interaction": {
        "positive": [
            "Encourages active classroom participation and welcoming debates.",
            "Treats every student with dignity, patience, and respect.",
            "Creates an engaging, dynamic, and interactive learning environment.",
            "Listens attentively to student viewpoints and questions.",
            "Fosters classroom interaction where everyone feels safe to speak."
        ],
        "negative": [
            "Intimidates students and ridicules genuine questions in class.",
            "Class is a dull one-way monologue with zero student interaction.",
            "Discourages questions and dismisses student contributions rudely.",
            "Hostile classroom environment where students fear participating.",
            "Completely unengaging; ignores students raising hands."
        ],
        "neutral": [
            "Class interaction was typical with occasional question sessions.",
            "Questions are entertained towards the final ten minutes of class.",
            "Interaction level was standard for a large lecture hall.",
            "Lectures are predominantly instructional with some Q&A."
        ]
    },
    "Use of Teaching Materials": {
        "positive": [
            "Provides comprehensive slides, reference materials, and handouts.",
            "Uses modern multimedia, projectors, and code examples effectively.",
            "Shares lecture notes well ahead of time on the student portal.",
            "Teaching materials are rich, up-to-date, and beautifully designed.",
            "Curated readings and textbook recommendations are extremely helpful."
        ],
        "negative": [
            "Does not share lecture slides or study materials with the class.",
            "Uses outdated, blurry, and handwritten materials from a decade ago.",
            "Refuses to upload lecture notes or reference documents.",
            "Teaching materials are riddled with typos, errors, and missing pages.",
            "Zero supplementary learning materials provided throughout the course."
        ],
        "neutral": [
            "Standard slide presentations were projected during lecture.",
            "Recommended textbooks are available in the university library.",
            "Course materials were shared according to standard faculty policy.",
            "Lecture slides cover only basic bullet points."
        ]
    }
}

CONNECTORS_CONTRAST = [
    ", but ", ", however, ", ", though ", ", yet ", ", on the other hand, ", " although "
]
CONNECTORS_AND = [
    " and ", ". Furthermore, ", ". Also, ", ". Additionally, ", "; moreover, "
]

def generate_synthetic_dataset(num_samples=1800):
    """
    Generates a rich, domain-specific evaluation dataset covering:
    - All 8 teaching aspects
    - Document-level sentiment (positive, negative, neutral)
    - Multi-clause aspect combinations with realistic contrastive and additive connectors
    - Nigerian university courses and lecturers
    """
    courses = [
        ("CSC410", "Special Computing"),
        ("CSC412", "Data Science & Big Data"),
        ("CSC406", "Cloud Computing Architectures"),
        ("CSC101", "Introduction to Computer Science"),
        ("CSC408", "Machine Learning & Neural Nets"),
        ("CSC302", "Database Design & Management"),
        ("CSC304", "Operating Systems Principles"),
        ("CSC201", "Data Structures and Algorithms")
    ]
    lecturers = [
        "Dr. Okafor", "Dr. Adeyemi", "Prof. Martins",
        "Dr. Faith", "Dr. Pomele", "Prof. Balogun",
        "Dr. Chukwu", "Dr. (Mrs) Adeleke"
    ]
    
    aspect_list = list(aspect_templates.keys())
    records = []

    for i in range(num_samples):
        ccode, cname = random.choice(courses)
        lecturer = random.choice(lecturers)
        
        # Decide number of aspects mentioned in this comment (1, 2, or 3)
        k = random.choices([1, 2, 3], weights=[0.25, 0.50, 0.25])[0]
        selected_aspects = random.sample(aspect_list, k)
        
        # Decide document sentiment bias
        target_doc_sentiment = random.choices(["positive", "negative", "neutral"], weights=[0.42, 0.38, 0.20])[0]
        
        clauses = []
        clause_labels = []
        
        if target_doc_sentiment == "neutral":
            # Mostly neutral statements
            for asp in selected_aspects:
                phrase = random.choice(aspect_templates[asp]["neutral"])
                clauses.append(phrase)
                clause_labels.append((asp, "neutral"))
            comment_text = " ".join(clauses)
            doc_sentiment = "neutral"
            rating = random.choice([3, 3, 3, 2, 4])
        elif target_doc_sentiment == "positive":
            # Mostly positive, may have 1 neutral or 1 contrastive negative
            for idx, asp in enumerate(selected_aspects):
                if idx > 0 and random.random() < 0.25:
                    sent = random.choice(["neutral", "negative"])
                else:
                    sent = "positive"
                phrase = random.choice(aspect_templates[asp][sent])
                clauses.append(phrase)
                clause_labels.append((asp, sent))
            
            # Combine clauses
            if len(clauses) == 1:
                comment_text = clauses[0]
            elif len(clauses) == 2:
                if clause_labels[0][1] != clause_labels[1][1]:
                    connector = random.choice(CONNECTORS_CONTRAST)
                else:
                    connector = random.choice(CONNECTORS_AND)
                comment_text = clauses[0] + connector + clauses[1]
            else:
                comment_text = clauses[0] + random.choice(CONNECTORS_AND) + clauses[1] + random.choice(CONNECTORS_CONTRAST) + clauses[2]
            
            pos_c = sum(1 for _, s in clause_labels if s == "positive")
            neg_c = sum(1 for _, s in clause_labels if s == "negative")
            if pos_c > neg_c:
                doc_sentiment = "positive"
                rating = random.choice([4, 5, 5, 4])
            elif neg_c > pos_c:
                doc_sentiment = "negative"
                rating = random.choice([1, 2])
            else:
                doc_sentiment = "neutral"
                rating = 3
        else: # negative
            for idx, asp in enumerate(selected_aspects):
                if idx > 0 and random.random() < 0.25:
                    sent = random.choice(["neutral", "positive"])
                else:
                    sent = "negative"
                phrase = random.choice(aspect_templates[asp][sent])
                clauses.append(phrase)
                clause_labels.append((asp, sent))
            
            if len(clauses) == 1:
                comment_text = clauses[0]
            elif len(clauses) == 2:
                if clause_labels[0][1] != clause_labels[1][1]:
                    connector = random.choice(CONNECTORS_CONTRAST)
                else:
                    connector = random.choice(CONNECTORS_AND)
                comment_text = clauses[0] + connector + clauses[1]
            else:
                comment_text = clauses[0] + random.choice(CONNECTORS_AND) + clauses[1] + random.choice(CONNECTORS_CONTRAST) + clauses[2]
            
            pos_c = sum(1 for _, s in clause_labels if s == "positive")
            neg_c = sum(1 for _, s in clause_labels if s == "negative")
            if neg_c > pos_c:
                doc_sentiment = "negative"
                rating = random.choice([1, 2, 1, 2])
            elif pos_c > neg_c:
                doc_sentiment = "positive"
                rating = random.choice([4, 5])
            else:
                doc_sentiment = "neutral"
                rating = 3

        records.append({
            "id": i + 1,
            "lecturer_name": lecturer,
            "course": cname,
            "course_code": ccode,
            "rating": rating,
            "comment": comment_text,
            "document_sentiment": doc_sentiment,
            "aspect_labels": json.dumps(clause_labels)
        })

    df = pd.DataFrame(records)
    return df

def main():
    print("========================================================================", flush=True)
    print("   AI-BASED LECTURER EVALUATION SYSTEM: CHAPTER THREE MODEL TRAINING", flush=True)
    print("========================================================================", flush=True)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("\n[STEP 1/5] Generating domain-specific student evaluation dataset...", flush=True)
    df = generate_synthetic_dataset(num_samples=1600)
    df.to_csv("data/synthetic_evaluations.csv", index=False)
    print(f"Generated {len(df)} realistic evaluation records across 8 teaching aspects.", flush=True)
    print("Document Sentiment Distribution:\n", df["document_sentiment"].value_counts(), flush=True)

    print("\n[STEP 2/5] Running preprocessing pipeline (cleaning, contractions, stemming)...", flush=True)
    df["preprocessed_comment"] = df["comment"].apply(preprocess_for_vectorizer)
    df.to_csv("data/preprocessed_evaluations.csv", index=False)
    print("Preprocessing completed and saved to data/preprocessed_evaluations.csv.", flush=True)

    # ==========================================
    # 2. DOCUMENT-LEVEL SENTIMENT CLASSIFICATION
    # (Section 3.6.1 & 3.6.3: SVM vs Naive Bayes)
    # ==========================================
    print("\n[STEP 3/5] Training Document-Level Classifiers (SVM vs Naive Bayes)...", flush=True)
    X_doc = df["preprocessed_comment"].values
    y_doc = df["document_sentiment"].values

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_doc, y_doc, test_size=0.2, random_state=42, stratify=y_doc
    )

    doc_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X_train_d_vec = doc_vectorizer.fit_transform(X_train_d)
    X_test_d_vec = doc_vectorizer.transform(X_test_d)

    # 1. Baseline Majority Classifier
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_d_vec, y_train_d)
    dummy_pred = dummy.predict(X_test_d_vec)
    dummy_acc = accuracy_score(y_test_d, dummy_pred)
    print(f"  [Baseline] Majority Class Accuracy: {dummy_acc:.4f}", flush=True)

    # 2. Linear SVM
    svm_doc = LinearSVC(C=1.0, max_iter=3000, random_state=42)
    svm_doc.fit(X_train_d_vec, y_train_d)
    svm_doc_pred = svm_doc.predict(X_test_d_vec)

    svm_doc_acc = accuracy_score(y_test_d, svm_doc_pred)
    svm_p_macro, svm_r_macro, svm_f1_macro, _ = precision_recall_fscore_support(
        y_test_d, svm_doc_pred, average="macro", zero_division=0
    )
    svm_p_weight, svm_r_weight, svm_f1_weight, _ = precision_recall_fscore_support(
        y_test_d, svm_doc_pred, average="weighted", zero_division=0
    )
    svm_doc_cm = confusion_matrix(y_test_d, svm_doc_pred, labels=["positive", "neutral", "negative"])

    print(f"\n  --- Document-Level SVM (LinearSVC) ---", flush=True)
    print(f"  Accuracy       : {svm_doc_acc:.4f}", flush=True)
    print(f"  Macro-F1       : {svm_f1_macro:.4f}", flush=True)
    print(f"  Weighted-F1    : {svm_f1_weight:.4f}", flush=True)
    print(f"  Macro-Precision: {svm_p_macro:.4f}", flush=True)
    print(f"  Macro-Recall   : {svm_r_macro:.4f}", flush=True)
    print(f"  Confusion Matrix (pos, neu, neg):\n{svm_doc_cm}", flush=True)

    # 3. Multinomial Naive Bayes
    nb_doc = MultinomialNB(alpha=0.5)
    nb_doc.fit(X_train_d_vec, y_train_d)
    nb_doc_pred = nb_doc.predict(X_test_d_vec)

    nb_doc_acc = accuracy_score(y_test_d, nb_doc_pred)
    nb_p_macro, nb_r_macro, nb_f1_macro, _ = precision_recall_fscore_support(
        y_test_d, nb_doc_pred, average="macro", zero_division=0
    )
    nb_p_weight, nb_r_weight, nb_f1_weight, _ = precision_recall_fscore_support(
        y_test_d, nb_doc_pred, average="weighted", zero_division=0
    )
    nb_doc_cm = confusion_matrix(y_test_d, nb_doc_pred, labels=["positive", "neutral", "negative"])

    print(f"\n  --- Document-Level Naive Bayes (MultinomialNB) ---", flush=True)
    print(f"  Accuracy       : {nb_doc_acc:.4f}", flush=True)
    print(f"  Macro-F1       : {nb_f1_macro:.4f}", flush=True)
    print(f"  Weighted-F1    : {nb_f1_weight:.4f}", flush=True)
    print(f"  Macro-Precision: {nb_p_macro:.4f}", flush=True)
    print(f"  Macro-Recall   : {nb_r_macro:.4f}", flush=True)
    print(f"  Confusion Matrix (pos, neu, neg):\n{nb_doc_cm}", flush=True)

    joblib.dump(svm_doc, "models/svm_document.pkl")
    joblib.dump(nb_doc, "models/nb_document.pkl")
    joblib.dump(doc_vectorizer, "models/vectorizer_document.pkl")
    print("\nSaved document-level models to models/", flush=True)

    # ==========================================
    # 3. ASPECT-LEVEL SENTIMENT CLASSIFICATION
    # (Section 3.6.2 & 3.6.3 for all 8 Aspects)
    # ==========================================
    print("\n[STEP 4/5] Training Aspect-Level Classifiers for all 8 Aspects...", flush=True)
    aspect_records = []
    for _, row in df.iterrows():
        c_text = row["preprocessed_comment"]
        labels = json.loads(row["aspect_labels"])
        for asp, sent in labels:
            aspect_records.append({
                "aspect": asp,
                "text": c_text,
                "sentiment": sent
            })
    
    aspect_df = pd.DataFrame(aspect_records)
    aspect_metrics = []
    
    for aspect in aspect_templates.keys():
        sub_data = aspect_df[aspect_df["aspect"] == aspect]
        if len(sub_data) < 20:
            continue

        X_asp = sub_data["text"].values
        y_asp = sub_data["sentiment"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_asp, y_asp, test_size=0.2, random_state=42, stratify=y_asp
        )

        asp_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        X_tr_v = asp_vec.fit_transform(X_tr)
        X_te_v = asp_vec.transform(X_te)

        # Train SVM
        svm_asp = LinearSVC(C=1.0, max_iter=3000, random_state=42)
        svm_asp.fit(X_tr_v, y_tr)
        svm_asp_pred = svm_asp.predict(X_te_v)
        svm_acc = accuracy_score(y_te, svm_asp_pred)
        _, _, svm_macro_f1, _ = precision_recall_fscore_support(y_te, svm_asp_pred, average="macro", zero_division=0)
        _, _, svm_weight_f1, _ = precision_recall_fscore_support(y_te, svm_asp_pred, average="weighted", zero_division=0)

        # Train NB
        nb_asp = MultinomialNB(alpha=0.5)
        nb_asp.fit(X_tr_v, y_tr)
        nb_asp_pred = nb_asp.predict(X_te_v)
        nb_acc = accuracy_score(y_te, nb_asp_pred)
        _, _, nb_macro_f1, _ = precision_recall_fscore_support(y_te, nb_asp_pred, average="macro", zero_division=0)
        _, _, nb_weight_f1, _ = precision_recall_fscore_support(y_te, nb_asp_pred, average="weighted", zero_division=0)

        safe_name = aspect.replace(" ", "_").lower()
        joblib.dump(svm_asp, f"models/svm_{safe_name}.pkl")
        joblib.dump(nb_asp, f"models/nb_{safe_name}.pkl")
        joblib.dump(asp_vec, f"models/vectorizer_{safe_name}.pkl")

        aspect_metrics.append({
            "Aspect": aspect,
            "Samples": len(sub_data),
            "SVM_Acc": round(svm_acc, 4),
            "SVM_Macro_F1": round(svm_macro_f1, 4),
            "SVM_Weighted_F1": round(svm_weight_f1, 4),
            "NB_Acc": round(nb_acc, 4),
            "NB_Macro_F1": round(nb_macro_f1, 4),
            "NB_Weighted_F1": round(nb_weight_f1, 4),
            "Best_Model": "SVM" if svm_macro_f1 >= nb_macro_f1 else "Naive Bayes"
        })

    asp_results_df = pd.DataFrame(aspect_metrics)
    print("\nAspect-Level Model Evaluation Summary:", flush=True)
    print(asp_results_df.to_string(index=False), flush=True)
    asp_results_df.to_csv("data/classification_results_summary.csv", index=False)

    # ==========================================
    # 4. EXPORT COMPREHENSIVE EVALUATION METRICS
    # ==========================================
    print("\n[STEP 5/5] Saving detailed metrics for FR-04 comparative dashboard...", flush=True)
    metrics_data = {
        "document_level": {
            "majority_baseline_accuracy": round(dummy_acc, 4),
            "svm": {
                "accuracy": round(svm_doc_acc, 4),
                "macro_precision": round(svm_p_macro, 4),
                "macro_recall": round(svm_r_macro, 4),
                "macro_f1": round(svm_f1_macro, 4),
                "weighted_f1": round(svm_f1_weight, 4),
                "confusion_matrix": svm_doc_cm.tolist(),
                "classes": ["positive", "neutral", "negative"]
            },
            "naive_bayes": {
                "accuracy": round(nb_doc_acc, 4),
                "macro_precision": round(nb_p_macro, 4),
                "macro_recall": round(nb_r_macro, 4),
                "macro_f1": round(nb_f1_macro, 4),
                "weighted_f1": round(nb_f1_weight, 4),
                "confusion_matrix": nb_doc_cm.tolist(),
                "classes": ["positive", "neutral", "negative"]
            }
        },
        "aspect_level": aspect_metrics
    }

    with open("data/model_evaluation_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)

    print("Model metrics saved to data/model_evaluation_metrics.json.", flush=True)
    print("=" * 70, flush=True)
    print("   ALL MODELS TRAINED AND SAVED SUCCESSFULLY!", flush=True)
    print("========================================================================", flush=True)

if __name__ == "__main__":
    main()

