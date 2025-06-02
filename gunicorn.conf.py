import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes
workers = 1  # Use only 1 worker to avoid conflicts with async operations
worker_class = "sync"
worker_connections = 1000
timeout = 60  # Increased timeout for long-running queries
keepalive = 2

# Restart workers after handling this many requests (prevents memory leaks)
max_requests = 1000
max_requests_jitter = 50

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'mcp_postgres_api'

# Preload app for better performance
preload_app = True

# Graceful timeout
graceful_timeout = 30

# Memory management
tmp_upload_dir = None
