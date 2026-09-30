import os
import sys
import io
import sqlite3
import json
import pandas as pd
import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lecturer-evaluation-fixed", "fixed")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.lexicons import ASPECTS
from core.ml_engine import load_all_models, run_analysis, generate_textual_summary

DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "feedback.db")
SEED_CSV = os.path.join(DATA_DIR, "synthetic_evaluations.csv")

DEMO_USERS = [
    ("admin", "admin123", "administrator", None, "System Administrator"),
    ("okafor", "lecturer123", "lecturer", "Dr. Okafor", "Dr. Okafor"),
    ("adeyemi", "lecturer123", "lecturer", "Dr. Adeyemi", "Dr. Adeyemi"),
    ("martins", "lecturer123", "lecturer", "Prof. Martins", "Prof. Martins"),
    ("faith", "lecturer123", "lecturer", "Dr. Faith", "Dr. Faith"),
    ("pomele", "lecturer123", "lecturer", "Dr. Pomele", "Dr. Pomele"),
    ("balogun", "lecturer123", "lecturer", "Prof. Balogun", "Prof. Balogun"),
    ("chukwu", "lecturer123", "lecturer", "Dr. Chukwu", "Dr. Chukwu"),
    ("adeleke", "lecturer123", "lecturer", "Dr. (Mrs) Adeleke", "Dr. (Mrs) Adeleke"),
]

st.set_page_config(
    page_title="EvalAI — Lecturer Evaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def init_database():
    os.makedirs(DATA_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            matric_number TEXT,
            lecturer_name TEXT NOT NULL,
            course TEXT NOT NULL,
            course_code TEXT,
            rating INTEGER NOT NULL,
            comment TEXT NOT NULL,
            document_sentiment TEXT,
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            aspects_json TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            lecturer_name TEXT,
            full_name TEXT
        )
    """)
    existing = con.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing == 0:
        for username, password, role, lecturer_name, full_name in DEMO_USERS:
            con.execute(
                "INSERT INTO users(username,password_hash,role,lecturer_name,full_name) VALUES(?,?,?,?,?)",
                (username, generate_password_hash(password), role, lecturer_name, full_name),
            )
    feedback_count = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    if feedback_count == 0 and os.path.exists(SEED_CSV):
        try:
            df = pd.read_csv(SEED_CSV)
            for _, row in df.head(150).iterrows():
                con.execute(
                    """INSERT INTO feedback
                    (student_name, matric_number, lecturer_name, course, course_code, rating, comment, document_sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        "Anonymous", "ANON", str(row.get("lecturer_name", "")),
                        str(row.get("course", "")), str(row.get("course_code", "CSC")),
                        int(row.get("rating", 3)), str(row.get("comment", "")),
                        str(row.get("document_sentiment", "neutral")),
                    ),
                )
            con.commit()
        except Exception:
            pass
    con.commit()
    return con

@st.cache_resource
def load_models_once():
    try:
        load_all_models()
        return True
    except Exception:
        return False

db = init_database()
load_models_once()

def query_df(sql, params=()):
    return pd.read_sql_query(sql, db, params=params)

def lecturers():
    rows = query_df("""
        SELECT DISTINCT lecturer_name FROM users
        WHERE role='lecturer' AND lecturer_name IS NOT NULL
        ORDER BY lecturer_name
    """)
    names = rows["lecturer_name"].tolist() if not rows.empty else []
    fb = query_df("SELECT DISTINCT lecturer_name FROM feedback ORDER BY lecturer_name")
    for name in fb["lecturer_name"].dropna().tolist():
        if name and name not in names:
            names.append(name)
    return sorted(names)

def login(username, password):
    row = db.execute(
        "SELECT * FROM users WHERE lower(username)=lower(?)", (username.strip(),)
    ).fetchone()
    if row and check_password_hash(row["password_hash"], password):
        return dict(row)
    return None

def logout():
    for key in ["authenticated", "user", "page"]:
        st.session_state.pop(key, None)

def feedback_analysis(df):
    sentiment = {"positive": 0, "neutral": 0, "negative": 0}
    aspect_counts = {a: {"positive": 0, "neutral": 0, "negative": 0} for a in ASPECTS}
    for _, row in df.iterrows():
        s = str(row.get("document_sentiment") or "neutral").lower()
        if s not in sentiment:
            s = "neutral"
        sentiment[s] += 1
        raw = row.get("aspects_json")
        if raw:
            try:
                obj = json.loads(raw)
                for item in obj.get("aspects", []):
                    a = item.get("aspect")
                    pol = item.get("sentiment", "neutral")
                    if a in aspect_counts and pol in aspect_counts[a]:
                        aspect_counts[a][pol] += 1
            except Exception:
                pass
    return sentiment, aspect_counts

def report_for(lecturer_name):
    df = query_df(
        "SELECT * FROM feedback WHERE lecturer_name=? ORDER BY submitted_at DESC",
        (lecturer_name,),
    )
    if df.empty:
        return df, {"count": 0, "avg_rating": 0, "pos_pct": 0, "neu_pct": 0, "neg_pct": 0}
    sentiments = df["document_sentiment"].fillna("neutral").str.lower()
    counts = sentiments.value_counts()
    total = len(df)
    stats = {
        "count": total,
        "avg_rating": round(float(df["rating"].mean()), 2),
        "pos_pct": round(100 * counts.get("positive", 0) / total, 1),
        "neu_pct": round(100 * counts.get("neutral", 0) / total, 1),
        "neg_pct": round(100 * counts.get("negative", 0) / total, 1),
    }
    return df, stats

def show_login():
    st.markdown("# EvalAI")
    st.subheader("Staff Login")
    st.caption("Students do not need an account. Staff use this area to view evaluation analytics.")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)
    if submitted:
        user = login(username, password)
        if user:
            st.session_state.authenticated = True
            st.session_state.user = user
            st.rerun()
        else:
            st.error("Invalid username or password.")

def show_student_form():
    st.title("Anonymous Lecturer Evaluation")
    st.write("Share your rating and comments. No student name or matric number is required.")
    lec_list = lecturers()
    with st.form("evaluation_form"):
        lecturer = st.selectbox("Lecturer", lec_list)
        course = st.text_input("Course")
        code = st.text_input("Course Code")
        rating = st.slider("Overall Rating", 1, 5, 3)
        comment = st.text_area("Comment", height=150, placeholder="Describe your experience with the lecturer...")
        submitted = st.form_submit_button("Submit Anonymous Evaluation", use_container_width=True)
    if submitted:
        if not course.strip() or not comment.strip():
            st.error("Please enter the course and comment.")
            return
        analysis = run_analysis(comment.strip(), rating)
        db.execute(
            """INSERT INTO feedback
            (student_name, matric_number, lecturer_name, course, course_code, rating, comment, document_sentiment, aspects_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("Anonymous", "ANON", lecturer, course.strip(), code.strip(),
             rating, comment.strip(), analysis.get("document_sentiment", "neutral"),
             json.dumps(analysis)),
        )
        db.commit()
        st.success("Thank you. Your anonymous evaluation has been submitted.")
        st.rerun()

def show_admin_dashboard():
    st.title("Institutional Performance Dashboard")
    df = query_df("SELECT * FROM feedback")
    if df.empty:
        st.info("No evaluation records yet.")
        return
    total = len(df)
    mean_rating = round(float(df["rating"].mean()), 2)
    sentiments = df["document_sentiment"].fillna("neutral").str.lower()
    positive_ratio = round(100 * (sentiments == "positive").sum() / total, 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Evaluations", total)
    c2.metric("Mean SET Rating", f"{mean_rating} / 5")
    c3.metric("Positive Feedback", f"{positive_ratio}%")
    c4.metric("Teaching Aspects", len(ASPECTS))

    sentiment_counts = sentiments.value_counts().reindex(["positive", "neutral", "negative"], fill_value=0)
    st.subheader("Overall Sentiment Distribution")
    st.bar_chart(sentiment_counts)

    rows = []
    for lec in lecturers():
        ldf, stats = report_for(lec)
        rows.append({
            "Lecturer": lec,
            "Evaluations": stats["count"],
            "Mean Rating": stats["avg_rating"],
            "Positive %": stats["pos_pct"],
            "Neutral %": stats["neu_pct"],
            "Negative %": stats["neg_pct"],
        })
    table = pd.DataFrame(rows)
    st.subheader("Lecturer Performance Summary")
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Export")
    st.download_button(
        "Download All Evaluations (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "lecturer_evaluations.csv",
        "text/csv",
    )

def show_lecturer_report(lecturer_name):
    df, stats = report_for(lecturer_name)
    st.title(lecturer_name)
    st.caption("Comprehensive Evaluation Report")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Evaluations", stats["count"])
    c2.metric("Mean Rating", f"{stats['avg_rating']} / 5")
    c3.metric("Positive", f"{stats['pos_pct']}%")
    c4.metric("Neutral", f"{stats['neu_pct']}%")
    c5.metric("Negative", f"{stats['neg_pct']}%")
    if df.empty:
        st.info("No comments submitted for this lecturer yet.")
        return

    sentiment, aspect_counts = feedback_analysis(df)
    overall = max(sentiment, key=sentiment.get) if sentiment else "neutral"
    summary = generate_textual_summary(aspect_counts, overall)
    st.info(summary)

    chart_df = pd.DataFrame(aspect_counts).T
    st.subheader("Aspect-Level Sentiment Breakdown")
    st.bar_chart(chart_df)

    st.subheader("Student Comments")
    for _, row in df.head(30).iterrows():
        with st.container(border=True):
            st.write(f"**{row['course']}** • Rating: **{row['rating']}/5**")
            st.write(f"_{row['comment']}_")
            st.caption(f"Sentiment: {str(row.get('document_sentiment') or 'neutral').capitalize()}")

    st.download_button(
        "Download Lecturer Report (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        f"{lecturer_name.replace(' ', '_')}_report.csv",
        "text/csv",
    )

def show_analysis_tool():
    st.title("Sentiment Analysis")
    st.caption("Analyze a sample student comment using the project's trained NLP models.")
    with st.form("analysis_form"):
        comment = st.text_area("Student comment", height=180)
        rating = st.slider("Rating", 1, 5, 3)
        submitted = st.form_submit_button("Analyze", use_container_width=True)
    if submitted and comment.strip():
        result = run_analysis(comment.strip(), rating)
        c1, c2, c3 = st.columns(3)
        c1.metric("Document Sentiment", result["document_sentiment"].capitalize())
        c2.metric("Rating", f"{rating}/5")
        c3.metric("Composite Score", result["overall_cs"])
        if result["aspects"]:
            st.subheader("Detected Teaching Aspects")
            st.dataframe(pd.DataFrame(result["aspects"]), use_container_width=True, hide_index=True)
        else:
            st.info("No specific teaching aspect was detected in this comment.")

def main():
    if not st.session_state.get("authenticated"):
        st.sidebar.title("EvalAI")
        st.sidebar.info("Anonymous student evaluation portal")
        if st.sidebar.button("Staff Login", use_container_width=True):
            st.session_state.show_staff_login = True
        if st.session_state.get("show_staff_login"):
            show_login()
        else:
            show_student_form()
        return

    user = st.session_state.user
    st.sidebar.title("EvalAI")
    st.sidebar.write(f"Signed in as **{user.get('full_name') or user.get('username')}**")
    if st.sidebar.button("Student Evaluation", use_container_width=True):
        st.session_state.page = "student"
    if user["role"] == "administrator":
        if st.sidebar.button("Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
        if st.sidebar.button("Sentiment Analysis", use_container_width=True):
            st.session_state.page = "analysis"
        if st.sidebar.button("Lecturer Report", use_container_width=True):
            st.session_state.page = "report"
    else:
        if st.sidebar.button("My Report", use_container_width=True):
            st.session_state.page = "report"
    if st.sidebar.button("Log out", use_container_width=True):
        logout()
        st.rerun()

    page = st.session_state.get("page", "dashboard" if user["role"] == "administrator" else "report")
    if page == "student":
        show_student_form()
    elif page == "dashboard" and user["role"] == "administrator":
        show_admin_dashboard()
    elif page == "analysis" and user["role"] == "administrator":
        show_analysis_tool()
    elif page == "report":
        if user["role"] == "lecturer":
            show_lecturer_report(user["lecturer_name"])
        else:
            names = lecturers()
            if names:
                selected = st.selectbox("Select Lecturer", names)
                show_lecturer_report(selected)
            else:
                st.info("No lecturers available.")
    else:
        show_admin_dashboard()

if __name__ == "__main__":
    main()
