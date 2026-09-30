import os
import sqlite3
import pandas as pd
from flask import g
from werkzeug.security import generate_password_hash

DEMO_LECTURERS = [
    "Dr. Okafor",
    "Dr. Adeyemi",
    "Prof. Martins",
    "Dr. Faith",
    "Dr. Pomele",
    "Prof. Balogun",
    "Dr. Chukwu",
    "Dr. (Mrs) Adeleke",
]

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _resolve_db_path():
    """
    Order of preference:
    1. DATABASE_PATH env var (set this to /data/feedback.db when a Railway Volume is mounted at /data)
    2. RAILWAY_VOLUME_MOUNT_PATH (Railway sets this automatically when a Volume is attached)
    3. Local fallback: <project>/data/feedback.db
    """
    explicit = os.environ.get('DATABASE_PATH')
    if explicit:
        return explicit
    volume = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH')
    if volume:
        return os.path.join(volume, 'feedback.db')
    return os.path.join(BASE_DIR, 'data', 'feedback.db')

DATABASE_PATH = _resolve_db_path()
SEED_CSV = os.path.join(BASE_DIR, 'data', 'synthetic_evaluations.csv')

DEFAULT_SETTINGS = {
    "announcement_banner": "2025/2026 Academic Session — Anonymous Student Evaluation of Teaching (SET) Portal Active",
    "show_announcement": "true",
    "landing_title": "Anonymous Lecturer Evaluation",
    "landing_subtitle": "Share numerical ratings and free-text comments. Your name and student identity are never stored.",
    "privacy_notice": "This form does not collect student name, matric number, or login details. Only the lecturer, course, rating, and comment are saved for analysis.",
    "custom_guidelines": "Please evaluate objectively based on course engagement, syllabus delivery, and instructional clarity."
}

def get_connection(db_path=DATABASE_PATH):
    """
    Creates an optimized SQLite connection with:
    - WAL journal mode for concurrent read/write throughput
    - In-memory temporary storage
    - 64MB cache size
    - Row factory for dictionary-like column access
    """
    con = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA synchronous = NORMAL;")
    con.execute("PRAGMA cache_size = -64000;")
    con.execute("PRAGMA temp_store = MEMORY;")
    con.execute("PRAGMA mmap_size = 268435456;")
    return con

def get_db():
    """Returns thread-local cached database connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = get_connection(DATABASE_PATH)
    return db

def init_db(db_path=DATABASE_PATH):
    """Initializes tables, auto-runs schema migrations, and builds B-Tree indexes."""
    os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
    db = get_connection(db_path)
    
    # 1. Base Tables
    db.execute('''
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
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            lecturer_name TEXT,
            full_name TEXT
        )
    ''')
    
    # 2. Schema Migrations (backward-compatibility safe)
    for col in ['student_name TEXT', 'matric_number TEXT', 'document_sentiment TEXT', 'aspects_json TEXT']:
        try:
            db.execute(f'ALTER TABLE feedback ADD COLUMN {col}')
        except Exception:
            pass

    db.execute('''
        CREATE TABLE IF NOT EXISTS site_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    ''')

    # Seed default landing page settings if missing
    for key, val in DEFAULT_SETTINGS.items():
        db.execute('INSERT OR IGNORE INTO site_settings (key, value) VALUES (?, ?)', (key, val))

    # 3. High-Performance B-Tree Indexing
    indexes = [
        ("idx_feedback_lecturer", "feedback(lecturer_name)"),
        ("idx_feedback_course", "feedback(course)"),
        ("idx_feedback_submitted_at", "feedback(submitted_at DESC)"),
        ("idx_feedback_rating", "feedback(rating)"),
        ("idx_users_username", "users(username)"),
        ("idx_users_lecturer_name", "users(lecturer_name)")
    ]
    for idx_name, idx_def in indexes:
        try:
            db.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {idx_def};")
        except Exception:
            pass

    # 4. Default Seed Users
    existing = db.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    if existing == 0:
        for username, password, role, lecturer_name, full_name in DEMO_USERS:
            db.execute(
                'INSERT INTO users (username, password_hash, role, lecturer_name, full_name) VALUES (?, ?, ?, ?, ?)',
                (username, generate_password_hash(password), role, lecturer_name, full_name)
            )
    db.commit()
    db.close()

def get_lecturer_names():
    """Returns list of active lecturer names dynamically from the database."""
    names = []
    try:
        db = get_db()
        rows = db.execute(
            "SELECT DISTINCT lecturer_name FROM users WHERE role = 'lecturer' AND lecturer_name IS NOT NULL"
        ).fetchall()
        for row in rows:
            if row['lecturer_name'] and row['lecturer_name'] not in names:
                names.append(row['lecturer_name'])
        fb_rows = db.execute('SELECT DISTINCT lecturer_name FROM feedback WHERE lecturer_name IS NOT NULL ORDER BY lecturer_name').fetchall()
        for row in fb_rows:
            if row['lecturer_name'] and row['lecturer_name'] not in names:
                names.append(row['lecturer_name'])
    except Exception:
        pass
    return sorted(names) if names else list(DEMO_LECTURERS)

def seed_db_if_empty():
    """Seeds realistic evaluation records if the database is currently empty."""
    db = get_db()
    count = db.execute("SELECT COUNT(*) as c FROM feedback").fetchone()['c']
    if count == 0 and os.path.exists(SEED_CSV):
        try:
            df = pd.read_csv(SEED_CSV)
            for _, row in df.head(150).iterrows():
                db.execute('''
                    INSERT INTO feedback (student_name, matric_number, lecturer_name, course, course_code, rating, comment, document_sentiment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    "Anonymous", "ANON",
                    row['lecturer_name'], row['course'], row.get('course_code', 'CSC'),
                    int(row['rating']), row['comment'], row.get('document_sentiment', 'neutral')
                ))
            db.commit()
        except Exception as e:
            print(f"[WARN] Failed to seed DB: {e}")

def get_site_settings():
    """Retrieves dynamic landing page and portal settings from the database."""
    settings = dict(DEFAULT_SETTINGS)
    try:
        db = get_db()
        rows = db.execute('SELECT key, value FROM site_settings').fetchall()
        for r in rows:
            settings[r['key']] = r['value']
    except Exception:
        pass
    return settings

def update_site_settings(new_settings):
    """Updates dynamic landing page settings in the database."""
    db = get_db()
    for key, val in new_settings.items():
        db.execute(
            'INSERT INTO site_settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value',
            (key, str(val))
        )
    db.commit()


