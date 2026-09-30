import gzip
import io
import time
from functools import wraps
from flask import request, after_this_request

# =========================================================================
# 1. DASHBOARD & EXPENSIVE QUERY MEMORY CACHE
# =========================================================================
_DASHBOARD_CACHE = {}
_CACHE_TIMESTAMP = 0
_CACHE_TTL = 300  # 5 minutes default TTL

def get_cached_dashboard():
    """Retrieves cached dashboard metrics if valid and not expired."""
    global _DASHBOARD_CACHE, _CACHE_TIMESTAMP
    if _DASHBOARD_CACHE and (time.time() - _CACHE_TIMESTAMP < _CACHE_TTL):
        return _DASHBOARD_CACHE
    return None

def set_cached_dashboard(data):
    """Sets cached dashboard metrics."""
    global _DASHBOARD_CACHE, _CACHE_TIMESTAMP
    _DASHBOARD_CACHE = data
    _CACHE_TIMESTAMP = time.time()

def invalidate_dashboard_cache():
    """Invalidates the dashboard cache when evaluations are inserted, uploaded, or deleted."""
    global _DASHBOARD_CACHE, _CACHE_TIMESTAMP
    _DASHBOARD_CACHE = {}
    _CACHE_TIMESTAMP = 0

# =========================================================================
# 2. HTTP PAYLOAD GZIP COMPRESSION MIDDLEWARE
# =========================================================================
def setup_compression(app):
    """
    Registers an after-request handler that automatically compresses text,
    HTML, JSON, and CSV payloads with Gzip when supported by the client.
    Reduces transfer size by 70-85%.
    """
    @app.after_request
    def compress_response(response):
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding.lower():
            return response

        # Only compress successful text/json/csv/html responses
        content_type = response.content_type.lower() if response.content_type else ''
        compressible_types = ('text/', 'application/json', 'application/javascript')
        if not any(content_type.startswith(t) for t in compressible_types):
            return response

        # Don't compress if already encoded or very small (< 400 bytes)
        if response.headers.get('Content-Encoding') or response.status_code < 200 or response.status_code >= 300:
            return response

        data = response.get_data()
        if len(data) < 400:
            return response

        gzip_buffer = io.BytesIO()
        with gzip.GzipFile(mode='wb', fileobj=gzip_buffer, compresslevel=6) as gzip_file:
            gzip_file.write(data)

        compressed_data = gzip_buffer.getvalue()
        response.set_data(compressed_data)
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = len(compressed_data)
        response.headers['Vary'] = 'Accept-Encoding'
        return response

    return app

