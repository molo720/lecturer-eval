import os
import io
import csv
import json
import time
from collections import defaultdict, deque
from functools import wraps
import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for,
    g, make_response, session, flash
)
from flask_wtf.csrf import CSRFProtect, CSRFError
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

import sys
import nltk

# Standard NLTK download for Render (no /tmp hacks needed)
for resource in ['stopwords', 'punkt', 'punkt_tab']:
    nltk.download(resource, quiet=True)

from core.lexicons import ASPECTS, ASPECT_KEYWORDS, NEGATION_MARKERS
from core.db import (
    get_db, init_db, get_lecturer_names, seed_db_if_empty,
    get_site_settings, update_site_settings, DATABASE_PATH
)
from core.ml_engine import load_all_models, run_analysis, generate_textual_summary
from core.optimizations import (
    setup_compression, get_cached_dashboard, set_cached_dashboard, invalidate_dashboard_cache
)

MIN_PASSWORD_LENGTH = 6
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, 'data', 'model_evaluation_metrics.json')
ON_RAILWAY = bool(os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RAILWAY_PROJECT_ID'))

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or (
    'dev-only-not-for-production' if not os.environ.get('VERCEL') else os.environ.get('SECRET_KEY', '')
)
if not app.secret_key:
    app.secret_key = os.environ.get('SECRET_KEY', 'vercel-set-SECRET_KEY-in-project-env')
app.config['DATABASE'] = DATABASE_PATH
app.config['WTF_CSRF_TIME_LIMIT'] = None
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_NAME'] = 'evalai_session'
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
# Railway terminates TLS at its proxy. The public HTTPS layer can still carry
# an ordinary cookie; disabling Secure here avoids browsers dropping the cookie
# when the upstream WSGI request is seen as HTTP.
app.config['SESSION_COOKIE_SECURE'] = bool(os.environ.get('VERCEL'))
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

csrf = CSRFProtect(app)


setup_compression(app)

evaluation_metrics = {}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please log in to access the staff platform.')
            return redirect(url_for('login', next=request.path))
        return view(*args, **kwargs)
    return wrapped

def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please log in to access the staff platform.')
            return redirect(url_for('login', next=request.path))
        if session.get('user_role') != 'administrator':
            flash('The analysis and administration tools are available only to administrators.')
            return redirect(url_for('staff_home'))
        return view(*args, **kwargs)
    return wrapped

def staff_home_url():
    if session.get('user_role') == 'lecturer':
        name = (session.get('lecturer_name') or '').strip()
        if not name:
            return url_for('change_password')
        return url_for('lecturer_report', lecturer_name=name)
    return url_for('dashboard')

def client_ip():
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or 'unknown'

_rate_buckets = defaultdict(deque)

def too_many_requests(key, limit, window_sec):
    now = time.time()
    bucket = _rate_buckets[key]
    while bucket and now - bucket[0] > window_sec:
        bucket.popleft()
    if len(bucket) >= limit:
        return True
    bucket.append(now)
    return False

def valid_password(password):
    return bool(password) and len(password) >= MIN_PASSWORD_LENGTH

def analysis_from_row(row):
    raw = None
    try:
        raw = row['aspects_json']
    except (IndexError, KeyError):
        raw = None
    if raw:
        try:
            stored = json.loads(raw)
            if isinstance(stored, dict) and 'document_sentiment' in stored:
                return stored, False
        except (TypeError, ValueError):
            pass
    return run_analysis(row['comment'], row['rating']), True

def persist_analysis_updates(updates):
    if not updates:
        return
    db = get_db()
    for feedback_id, ana in updates:
        db.execute(
            'UPDATE feedback SET document_sentiment = ?, aspects_json = ? WHERE id = ?',
            (ana.get('document_sentiment', 'neutral'), json.dumps(ana), feedback_id)
        )
    db.commit()

@app.before_request
def require_password_change():
    if not session.get('user_id') or not session.get('must_change_password'):
        return
    if request.endpoint in ('change_password', 'logout', 'static', None):
        return
    flash('Please set a new password before continuing.')
    return redirect(url_for('change_password'))

@app.errorhandler(CSRFError)
def handle_csrf_error(error):
    flash('Your form session expired or was invalid. Please try again.')
    return redirect(request.referrer or url_for('landing'))


def save_anonymous_evaluation(form):
    lecturer = form.get('lecturer_name', '').strip()
    course = form.get('course', '').strip()
    code = form.get('course_code', '').strip()
    rating = form.get('rating', '3')
    comment = form.get('comment', '').strip()
    if not lecturer or not course or not comment:
        return False, "Please fill in lecturer, course, and comment."
    try:
        r_val = int(rating)
    except Exception:
        r_val = 3
    ana = run_analysis(comment, r_val)
    db = get_db()
    cols = [r[1] for r in db.execute("PRAGMA table_info(feedback)").fetchall()]
    data_dict = {
        'lecturer_name': lecturer,
        'course': course,
        'course_code': code,
        'rating': r_val,
        'comment': comment,
        'document_sentiment': ana.get('document_sentiment', 'neutral'),
        'aspects_json': json.dumps(ana)
    }
    if 'student_name' in cols:
        data_dict['student_name'] = 'Anonymous'
    if 'matric_number' in cols:
        data_dict['matric_number'] = 'ANON'

    insert_cols = [c for c in data_dict.keys() if c in cols]
    placeholders = ', '.join(['?'] * len(insert_cols))
    sql = f"INSERT INTO feedback ({', '.join(insert_cols)}) VALUES ({placeholders})"
    values = [data_dict[c] for c in insert_cols]
    db.execute(sql, values)
    db.commit()

    # Invalidate dashboard cache on new submission
    invalidate_dashboard_cache()
    return True, None

@app.route("/", methods=['GET', 'POST'])
def landing():
    error = None
    success = False
    if request.method == 'POST':
        if too_many_requests(f"submit:{client_ip()}", 8, 600):
            error = 'Too many submissions from this network. Please wait a few minutes and try again.'
        else:
            ok, error = save_anonymous_evaluation(request.form)
            success = ok
    settings = get_site_settings()
    return render_template(
        'landing.html',
        error=error,
        success=success,
        aspects=ASPECTS,
        lecturers=get_lecturer_names(),
        settings=settings
    )

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if too_many_requests(f"submit:{client_ip()}", 8, 600):
            error = 'Too many submissions from this network. Please wait a few minutes and try again.'
            settings = get_site_settings()
            return render_template('landing.html', error=error, success=False, aspects=ASPECTS, lecturers=get_lecturer_names(), settings=settings)
        ok, error = save_anonymous_evaluation(request.form)
        if ok:
            flash('Thank you. Your anonymous evaluation has been submitted.')
            return redirect(url_for('landing'))
        settings = get_site_settings()
        return render_template('landing.html', error=error, success=False, aspects=ASPECTS, lecturers=get_lecturer_names(), settings=settings)
    return redirect(url_for('landing'))

@app.route('/staff')
@login_required
def staff_home():
    return redirect(staff_home_url())

@app.route('/analyze', methods=['GET', 'POST'])
@admin_required
def analyze():
    results = None
    error = None
    prefill = {'comment': '', 'rating': '3', 'lecturer': 'Dr. Okafor', 'course': 'Machine Learning'}

    if request.method == 'POST':
        prefill['comment'] = request.form.get('comment', '').strip()
        prefill['rating'] = request.form.get('rating', '3')
        prefill['lecturer'] = (request.form.get('lecturer_name') or request.form.get('lecturer') or '').strip()
        prefill['course'] = request.form.get('course', '').strip()
        if not prefill['comment']:
            error = "Please enter a comment to analyze."
        else:
            try:
                r_val = int(prefill['rating'])
            except Exception:
                r_val = 3
            results = run_analysis(prefill['comment'], r_val)
            results['lecturer'] = prefill['lecturer']
            results['course'] = prefill['course']
    else:
        prefill['comment'] = request.args.get('comment', '')
        if prefill['comment']:
            try:
                r_val = int(request.args.get('rating', '3'))
            except Exception:
                r_val = 3
            results = run_analysis(prefill['comment'], r_val)
            results['lecturer'] = prefill['lecturer']
            results['course'] = prefill['course']

    return render_template('index.html', results=results, error=error, prefill=prefill, aspects=ASPECTS)

@app.route('/dashboard')
@login_required
def dashboard():
    if session.get('user_role') != 'administrator':
        flash('Institution-wide analytics are available only to administrators.')
        return redirect(staff_home_url())

    cached_data = get_cached_dashboard()
    if cached_data:
        return render_template('dashboard.html', **cached_data)

    db = get_db()
    rows = db.execute('SELECT * FROM feedback').fetchall()
    
    lecturer_map = {}
    sentiment_dist = {"positive": 0, "neutral": 0, "negative": 0}
    aspect_counts = {a: {"positive": 0, "neutral": 0, "negative": 0} for a in ASPECTS}
    total_ratings = 0

    for r in rows:
        lec = r['lecturer_name']
        rating = r['rating']
        total_ratings += rating
        
        if lec not in lecturer_map:
            lecturer_map[lec] = {
                "name": lec,
                "count": 0,
                "ratings": [],
                "sentiments": {"positive": 0, "neutral": 0, "negative": 0},
                "aspects": {a: 0 for a in ASPECTS}
            }
        
        lecturer_map[lec]["count"] += 1
        lecturer_map[lec]["ratings"].append(rating)

        # Run LRU-cached analysis
        ana = run_analysis(r['comment'], rating)
        doc_sent = ana['document_sentiment']
        sentiment_dist[doc_sent] += 1
        lecturer_map[lec]["sentiments"][doc_sent] += 1

        for asp in ana['aspects']:
            a_name = asp['aspect']
            a_sent = asp['sentiment']
            if a_name in aspect_counts:
                aspect_counts[a_name][a_sent] += 1
                if a_sent == "positive":
                    lecturer_map[lec]["aspects"][a_name] += 1

    total = len(rows)
    mean_rating = round(total_ratings / total, 2) if total > 0 else 0
    pos_ratio = round((sentiment_dist["positive"] / total) * 100, 1) if total > 0 else 0

    lecturer_table = []
    for lec, d in lecturer_map.items():
        c = d["count"]
        avg_r = round(sum(d["ratings"]) / c, 2) if c > 0 else 0
        p_pct = round((d["sentiments"]["positive"] / c) * 100) if c > 0 else 0
        u_pct = round((d["sentiments"]["neutral"] / c) * 100) if c > 0 else 0
        n_pct = round((d["sentiments"]["negative"] / c) * 100) if c > 0 else 0
        
        top_asp = max(d["aspects"], key=d["aspects"].get) if d["aspects"] else "Teaching Clarity"
        if d["aspects"][top_asp] == 0:
            top_asp = "Teaching Clarity"

        lecturer_table.append({
            "name": lec,
            "count": c,
            "avg_rating": avg_r,
            "pos_pct": p_pct,
            "neu_pct": u_pct,
            "neg_pct": n_pct,
            "top_aspect": top_asp
        })

    chart_data = {
        "sentiment_dist": sentiment_dist,
        "aspect_names": ASPECTS,
        "aspect_pos": [aspect_counts[a]["positive"] for a in ASPECTS],
        "aspect_neu": [aspect_counts[a]["neutral"] for a in ASPECTS],
        "aspect_neg": [aspect_counts[a]["negative"] for a in ASPECTS]
    }

    stats = {
        "mean_rating": mean_rating,
        "pos_ratio": pos_ratio
    }

    render_context = {
        'total': total,
        'stats': stats,
        'lecturer_table': lecturer_table,
        'chart_data': chart_data
    }
    set_cached_dashboard(render_context)
    return render_template('dashboard.html', **render_context)

@app.route('/model-evaluation')
@admin_required
def model_evaluation():
    """ Comparative Evaluation of SVM and Naive Bayes."""
    global evaluation_metrics
    if not evaluation_metrics and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            evaluation_metrics = json.load(f)
    return render_template('model_evaluation.html', metrics=evaluation_metrics)

@app.route('/admin/retrain', methods=['POST', 'GET'])
@admin_required
def admin_retrain():
    
    try:
        from train_models import main as run_train
        run_train()
        load_all_models()
        invalidate_dashboard_cache()
        flash("Models retrained successfully! Hyperparameters tuned and evaluation metrics updated.")
    except Exception as e:
        flash(f"Retraining error: {e}")
    return redirect(url_for('model_evaluation'))

@app.route('/lecturer/<lecturer_name>')
@login_required
def lecturer_report(lecturer_name):
    """
     Per-lecturer summary report.
    .
    """
    user_role = session.get('user_role', 'administrator')
    current_lec = session.get('lecturer_name', '')

    if user_role == 'lecturer' and current_lec != lecturer_name:
        flash(f"Access restricted: As a Lecturer, you can only view personal reports for courses taught ({current_lec}).")
        return redirect(url_for('lecturer_report', lecturer_name=current_lec))

    db = get_db()
    rows = db.execute('SELECT * FROM feedback WHERE lecturer_name = ? ORDER BY submitted_at DESC', (lecturer_name,)).fetchall()
    
    total = len(rows)
    ratings = [r['rating'] for r in rows]
    avg_rating = round(sum(ratings) / total, 2) if total > 0 else 0

    aspect_counts = {a: {"positive": 0, "neutral": 0, "negative": 0} for a in ASPECTS}
    pos_count, neu_count, neg_count = 0, 0, 0
    sample_comments = []

    for r in rows:
        ana = run_analysis(r['comment'], r['rating'])
        s = ana['document_sentiment']
        if s == 'positive': pos_count += 1
        elif s == 'negative': neg_count += 1
        else: neu_count += 1

        for asp in ana['aspects']:
            a_name = asp['aspect']
            if a_name in aspect_counts:
                aspect_counts[a_name][asp['sentiment']] += 1

        if len(sample_comments) < 8:
            sample_comments.append({
                "comment": r['comment'],
                "rating": r['rating'],
                "course": r['course'],
                "sentiment": s
            })

    pos_pct = round((pos_count / total) * 100) if total > 0 else 0
    neu_pct = round((neu_count / total) * 100) if total > 0 else 0
    neg_pct = round((neg_count / total) * 100) if total > 0 else 0

    textual_summary = generate_textual_summary(aspect_counts, "positive" if pos_count > neg_count else "negative")

    data = {
        "count": total,
        "avg_rating": avg_rating,
        "pos_count": pos_count,
        "neu_count": neu_count,
        "neg_count": neg_count,
        "pos_pct": pos_pct,
        "neu_pct": neu_pct,
        "neg_pct": neg_pct,
        "aspect_counts": aspect_counts,
        "aspect_order": ASPECTS,
        "textual_summary": textual_summary
    }

    all_lecturers = get_lecturer_names()

    return render_template(
        'lecturer_report.html',
        lecturer_name=lecturer_name,
        all_lecturers=all_lecturers,
        data=data,
        feedbacks=rows,
        sample_comments=sample_comments
    )

@app.route('/course/<course_name>')
@admin_required
def course_report(course_name):
    """ Per-course summary report."""
    db = get_db()
    rows = db.execute('SELECT * FROM feedback WHERE course = ? ORDER BY submitted_at DESC', (course_name,)).fetchall()
    
    total = len(rows)
    ratings = [r['rating'] for r in rows]
    avg_rating = round(sum(ratings) / total, 2) if total > 0 else 0

    aspect_counts = {a: {"positive": 0, "neutral": 0, "negative": 0} for a in ASPECTS}
    pos_c, neg_c = 0, 0

    for r in rows:
        ana = run_analysis(r['comment'], r['rating'])
        if ana['document_sentiment'] == 'positive': pos_c += 1
        elif ana['document_sentiment'] == 'negative': neg_c += 1

        for asp in ana['aspects']:
            if asp['aspect'] in aspect_counts:
                aspect_counts[asp['aspect']][asp['sentiment']] += 1

    pos_pct = round((pos_c / total) * 100) if total > 0 else 0
    neg_pct = round((neg_c / total) * 100) if total > 0 else 0

    data = {
        "count": total,
        "avg_rating": avg_rating,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "aspect_counts": aspect_counts,
        "aspect_order": ASPECTS
    }

    return render_template('course_report.html', course_name=course_name, data=data, feedbacks=rows)

@app.route('/feedback')
@admin_required
def feedback_list():
    """Server-side paginated and filtered feedback list (High Performance)."""
    db = get_db()
    page = request.args.get('page', 1, type=int)
    if page < 1:
        page = 1
    per_page = 25
    q = (request.args.get('q') or '').strip()
    lecturer = (request.args.get('lecturer') or '').strip()

    where_clauses = []
    params = []
    if q:
        where_clauses.append("(comment LIKE ? OR course LIKE ? OR course_code LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
    if lecturer:
        where_clauses.append("lecturer_name = ?")
        params.append(lecturer)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    count_sql = f"SELECT COUNT(*) as total FROM feedback {where_sql}"
    total_records = db.execute(count_sql, params).fetchone()['total']
    total_pages = max(1, (total_records + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages

    offset = (page - 1) * per_page
    query_sql = f"SELECT * FROM feedback {where_sql} ORDER BY submitted_at DESC LIMIT ? OFFSET ?"
    rows = db.execute(query_sql, params + [per_page, offset]).fetchall()
    all_lecturers = get_lecturer_names()

    return render_template(
        'feedback.html',
        feedbacks=rows,
        current_page=page,
        total_pages=total_pages,
        total_records=total_records,
        search_query=q,
        selected_lecturer=lecturer,
        all_lecturers=all_lecturers
    )

@app.route('/feedback/<int:id>')
@admin_required
def view_feedback(id):
    db = get_db()
    row = db.execute('SELECT * FROM feedback WHERE id = ?', (id,)).fetchone()
    if not row:
        return redirect(url_for('feedback_list'))
    results = run_analysis(row['comment'], row['rating'])
    results['lecturer'] = row['lecturer_name']
    results['course'] = row['course']
    return render_template('view.html', feedback=row, results=results)

@app.route('/upload', methods=['GET', 'POST'])
@admin_required
def upload():
    """ Upload of student evaluation data in CSV or Excel format."""
    error = None
    success_message = None
    results = []

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            error = "No file selected."
        else:
            fn = file.filename.lower()
            if not (fn.endswith('.csv') or fn.endswith('.xlsx') or fn.endswith('.xls')):
                error = "Unsupported file format. Please upload a CSV or Excel (.xlsx) file."
            else:
                try:
                    if fn.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)

                    req_cols = {'comment', 'rating'}
                    cols_lower = {c.lower(): c for c in df.columns}
                    
                    if not req_cols.issubset(set(cols_lower.keys())):
                        error = f"Uploaded file must contain 'comment' and 'rating' columns. Found: {list(df.columns)}"
                    else:
                        comment_col = cols_lower['comment']
                        rating_col = cols_lower['rating']
                        lec_col = cols_lower.get('lecturer_name') or cols_lower.get('lecturer')
                        course_col = cols_lower.get('course')
                        code_col = cols_lower.get('course_code')

                        db = get_db()
                        count_added = 0

                        for _, row in df.iterrows():
                            c_text = str(row[comment_col]).strip()
                            if not c_text or c_text == 'nan':
                                continue
                            try:
                                r_val = int(row[rating_col])
                            except Exception:
                                r_val = 3
                            
                            lec = str(row[lec_col]).strip() if lec_col and pd.notna(row[lec_col]) else "Dr. Okafor"
                            crs = str(row[course_col]).strip() if course_col and pd.notna(row[course_col]) else "Computer Science"
                            cdc = str(row[code_col]).strip() if code_col and pd.notna(row[code_col]) else "CSC"

                            analysis = run_analysis(c_text, r_val)
                            db.execute('''
                                INSERT INTO feedback (student_name, matric_number, lecturer_name, course, course_code, rating, comment, document_sentiment)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            ''', ("Anonymous", "ANON", lec, crs, cdc, r_val, c_text, analysis['document_sentiment']))
                            
                            if len(results) < 15:
                                analysis['lecturer'] = lec
                                analysis['course'] = crs
                                results.append(analysis)
                            
                            count_added += 1

                        db.commit()
                        invalidate_dashboard_cache()
                        success_message = f"Successfully imported and evaluated {count_added} records!"
                except Exception as e:
                    error = f"Error processing file: {e}"

    return render_template('upload.html', error=error, success_message=success_message, results=results)

@app.route('/export')
@app.route('/export/csv')
@admin_required
def export_csv():
    
    db = get_db()
    rows = db.execute('SELECT * FROM feedback ORDER BY submitted_at DESC').fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Date', 'Lecturer', 'Course', 'Course Code', 'SET Rating', 'Document Sentiment', 'Comment'])
    for r in rows:
        r_dict = dict(r)
        sent = r_dict.get('document_sentiment') or 'neutral'
        writer.writerow([r['id'], r['submitted_at'], r['lecturer_name'], r['course'], r['course_code'], r['rating'], sent, r['comment']])
    output.seek(0)
    res = make_response(output.getvalue())
    res.headers["Content-Disposition"] = "attachment; filename=all_evaluations_export.csv"
    res.headers["Content-Type"] = "text/csv"
    return res

@app.route('/export/lecturer/<lecturer_name>')
@login_required
def export_lecturer_csv(lecturer_name):
    """FR-08: Export specific lecturer data in CSV."""
    user_role = session.get('user_role', '')
    current_lec = session.get('lecturer_name', '')
    if user_role == 'lecturer' and current_lec != lecturer_name:
        flash('You can export only your own evaluation report.')
        return redirect(url_for('lecturer_report', lecturer_name=current_lec))
    db = get_db()
    rows = db.execute('SELECT * FROM feedback WHERE lecturer_name = ? ORDER BY submitted_at DESC', (lecturer_name,)).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Date', 'Lecturer', 'Course', 'Course Code', 'SET Rating', 'Document Sentiment', 'Comment'])
    for r in rows:
        r_dict = dict(r)
        sent = r_dict.get('document_sentiment') or 'neutral'
        writer.writerow([r['id'], r['submitted_at'], r['lecturer_name'], r['course'], r['course_code'], r['rating'], sent, r['comment']])
    output.seek(0)
    safe_name = lecturer_name.replace(' ', '_').lower()
    res = make_response(output.getvalue())
    res.headers["Content-Disposition"] = f"attachment; filename={safe_name}_evaluations.csv"
    res.headers["Content-Type"] = "text/csv"
    return res

@app.route('/login', methods=['GET', 'POST'])
def login():
    
    if session.get('user_id'):
        return redirect(staff_home_url())

    error = None
    next_url = request.args.get('next') or request.form.get('next') or ''
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip().lower()
        password = request.form.get('password') or ''
        db = get_db()
        user = db.execute('SELECT * FROM users WHERE lower(username) = ?', (username,)).fetchone()
        if user and check_password_hash(user['password_hash'], password):
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['user_role'] = user['role']
            session['lecturer_name'] = user['lecturer_name'] or ''
            session['full_name'] = user['full_name'] or user['username']
            if next_url.startswith('/') and not next_url.startswith('//'):
                return redirect(next_url)
            return redirect(staff_home_url())
        error = 'Invalid username or password.'
    return render_template('login.html', error=error, next_url=next_url)

@app.route('/admin/users', methods=['GET', 'POST'])
@admin_required
def admin_users():
    db = get_db()
    error = None
    if request.method == 'POST':
        lecturer_name = (request.form.get('lecturer_name') or '').strip()
        username = (request.form.get('username') or '').strip().lower()
        password = request.form.get('password') or ''
        if not lecturer_name or not username or not password:
            error = 'All account fields are required.'
        else:
            try:
                db.execute(
                    'INSERT INTO users (username, password_hash, role, lecturer_name, full_name) VALUES (?, ?, ?, ?, ?)',
                    (username, generate_password_hash(password), 'lecturer', lecturer_name, lecturer_name)
                )
                db.commit()
                flash(f'Lecturer account created for {lecturer_name}.')
                return redirect(url_for('admin_users'))
            except Exception:
                error = 'That username is already in use.'
    users = db.execute('SELECT id, username, role, lecturer_name FROM users ORDER BY role, username').fetchall()
    all_lecturers = get_lecturer_names()
    return render_template('admin_users.html', users=users, all_lecturers=all_lecturers, error=error)

@app.route('/admin/users/<int:user_id>/password', methods=['POST'])
@admin_required
def admin_set_user_password(user_id):
    """Allows an administrator to reset a user account password."""
    new_password = (request.form.get('new_password') or '').strip()
    if len(new_password) < 4:
        flash('Password must be at least 4 characters.')
        return redirect(url_for('admin_users'))

    db = get_db()
    user = db.execute('SELECT id, username FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        flash('Account not found.')
        return redirect(url_for('admin_users'))

    db.execute(
        'UPDATE users SET password_hash = ? WHERE id = ?',
        (generate_password_hash(new_password), user_id)
    )
    db.commit()
    flash(f"Password updated for '{user['username']}'.")
    return redirect(url_for('admin_users'))

@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    """Allows an administrator to delete a user/lecturer account."""
    if session.get('user_id') == user_id:
        flash('You cannot delete your own active administrator account.')
        return redirect(url_for('admin_users'))

    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        flash('Account not found.')
        return redirect(url_for('admin_users'))

    username = user['username']
    lec_name = user['lecturer_name']
    delete_feedback = request.form.get('delete_feedback') == '1'

    db.execute('DELETE FROM users WHERE id = ?', (user_id,))

    fb_count = 0
    if delete_feedback and lec_name:
        cursor = db.execute('DELETE FROM feedback WHERE lecturer_name = ?', (lec_name,))
        fb_count = cursor.rowcount

    db.commit()
    invalidate_dashboard_cache()

    msg = f"User '{username}' was deleted successfully."
    if fb_count > 0:
        msg += f" Associated {fb_count} feedback evaluations for {lec_name} were also deleted."
    flash(msg)
    return redirect(url_for('admin_users'))

@app.route('/admin/lecturers/delete', methods=['POST'])
@admin_required
def admin_delete_lecturer():
    """Allows an administrator to delete a lecturer and their evaluation records."""
    lecturer_name = (request.form.get('lecturer_name') or '').strip()
    if not lecturer_name:
        flash('Please select a lecturer name to delete.')
        return redirect(request.referrer or url_for('admin_users'))

    db = get_db()
    db.execute('DELETE FROM users WHERE lecturer_name = ?', (lecturer_name,))
    cursor = db.execute('DELETE FROM feedback WHERE lecturer_name = ?', (lecturer_name,))
    fb_count = cursor.rowcount
    db.commit()
    invalidate_dashboard_cache()

    flash(f"Lecturer '{lecturer_name}' and all associated {fb_count} evaluation records have been deleted.")
    return redirect(request.referrer or url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('landing'))

@app.teardown_appcontext
def close_db(error):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# Startup Initialization
with app.app_context():
    init_db()
    load_all_models()
    seed_db_if_empty()

if __name__ == '__main__':
    app.run(debug=True)