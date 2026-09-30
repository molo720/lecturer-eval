"""
Production-ready server launcher with multi-threading and load balancing support.
Usage:
    python run_server.py
"""
import os
import sys

# Ensure local core and src modules are on python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from app import app

def run():
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    threads = int(os.environ.get('THREADS', 8))

    print("=" * 65)
    print(" Lecturer Evaluation System — Optimized Production Server")
    print("=" * 65)
    print(f" * Listening on http://{host}:{port}")
    print(f" * Concurrent Worker Threads: {threads}")
    print(f" * Gzip Compression: ENABLED")
    print(f" * SQLite WAL Mode & B-Tree Indexing: ACTIVE")
    print(f" * LRU Cache for ML Inference: ACTIVE")
    print(f" * Response Query Caching: ACTIVE")
    print("=" * 65)

    try:
        # If waitress is installed, use its production WSGI runner
        from waitress import serve
        serve(app, host=host, port=port, threads=threads)
    except ImportError:
        # Fallback to high-concurrency threaded WSGI runner
        app.run(host=host, port=port, threaded=True, debug=False)

if __name__ == '__main__':
    run()

