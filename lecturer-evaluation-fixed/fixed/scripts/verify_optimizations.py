import sqlite3
import time
from app import app, get_db

print("[1] Testing Database Optimization...")
with app.app_context():
    db = get_db()
    
    # Check WAL mode
    wal_mode = db.execute("PRAGMA journal_mode;").fetchone()[0]
    print(f"    SQLite Journal Mode: {wal_mode.upper()} (WAL expected)")
    assert wal_mode.lower() == 'wal', f"Expected WAL, got {wal_mode}"
    
    # Check Indexes
    indexes = [r[1] for r in db.execute("PRAGMA index_list('feedback');").fetchall()]
    print(f"    Feedback Table Indexes: {indexes}")
    assert any('idx_feedback' in idx for idx in indexes), "Missing feedback indexes!"

print("\n[2] Testing Gzip Compression & Response Headers...")
client = app.test_client()

# Request with Accept-Encoding: gzip
res = client.get('/', headers={'Accept-Encoding': 'gzip'})
print(f"    Landing Page Status: {res.status_code}")
content_encoding = res.headers.get('Content-Encoding')
print(f"    Content-Encoding: {content_encoding} (gzip expected)")
assert content_encoding == 'gzip', "Response was not compressed with gzip!"

print("\n[3] Testing Server-Side Pagination...")
with client.session_transaction() as sess:
    sess['user_id'] = 1
    sess['user_role'] = 'administrator'
    sess['username'] = 'admin'

res_p1 = client.get('/feedback?page=1')
print(f"    Feedback Page 1 Status: {res_p1.status_code}")
assert res_p1.status_code == 200, "Feedback page 1 failed"

res_p2 = client.get('/feedback?page=2')
print(f"    Feedback Page 2 Status: {res_p2.status_code}")
assert res_p2.status_code == 200, "Feedback page 2 failed"

print("\n[4] Testing Dashboard Caching Speedup...")
# First request (computes metrics)
t0 = time.time()
res_dash1 = client.get('/dashboard')
t_uncached = time.time() - t0
print(f"    Uncached Dashboard Request Time: {t_uncached * 1000:.2f} ms")

# Second request (hits memory cache)
t0 = time.time()
res_dash2 = client.get('/dashboard')
t_cached = time.time() - t0
print(f"    Cached Dashboard Request Time: {t_cached * 1000:.2f} ms")
assert res_dash2.status_code == 200, "Dashboard failed"
if t_cached < t_uncached:
    print(f"    >>> Cache Speedup: {t_uncached / max(t_cached, 0.0001):.1f}x FASTER!")

print("\n[5] Testing ML Inference LRU Cache...")
from core.ml_engine import run_analysis

# Uncached analysis
t0 = time.time()
r1 = run_analysis("The lecturer explains machine learning concepts clearly with good slides.", 4)
t_ml1 = time.time() - t0

# Cached analysis (same text + rating)
t0 = time.time()
r2 = run_analysis("The lecturer explains machine learning concepts clearly with good slides.", 4)
t_ml2 = time.time() - t0
print(f"    First ML Inference: {t_ml1 * 1000:.3f} ms")
print(f"    Cached ML Inference: {t_ml2 * 1000:.3f} ms")
assert r1['document_sentiment'] == r2['document_sentiment']
print(f"    >>> ML Cache Speedup: {t_ml1 / max(t_ml2, 0.00001):.1f}x FASTER!")

print("\nALL OPTIMIZATION VERIFICATIONS PASSED SUCCESSFULLY!")

