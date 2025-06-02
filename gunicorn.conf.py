import multiprocessing
import os

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes - Keep at 1 for MCP operations to avoid conflicts
workers = 1
worker_class = "sync"
worker_connections = 100  # Reduced for cloud environment

# Increased timeouts for long-running database operations
timeout = 120  # 2 minutes for worker timeout
keepalive = 5
graceful_timeout = 45  # Time to wait for graceful worker shutdown

# Restart workers more frequently to prevent memory issues
max_requests = 100  # Reduced for cloud environment
max_requests_jitter = 10

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'mcp_postgres_api'

# Don't preload app to avoid issues with async initialization
preload_app = False

# Memory management for cloud deployment
worker_tmp_dir = "/dev/shm"  # Use shared memory for temp files
tmp_upload_dir = None

# Additional settings for cloud deployment
forwarded_allow_ips = "*"  # Allow all forwarded IPs (Render proxy)
secure_scheme_headers = {
    'X-FORWARDED-PROTOCOL': 'https',
    'X-FORWARDED-PROTO': 'https',
    'X-FORWARDED-SSL': 'on'
}

# Resource limits
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

def when_ready(server):
    """Called when the server is ready to start accepting connections"""
    server.log.info("MCP PostgreSQL API server is ready")

def worker_exit(server, worker):
    """Called when a worker exits"""
    server.log.info(f"Worker {worker.pid} exited")

def on_exit(server):
    """Called when the master process exits"""
    server.log.info("MCP PostgreSQL API server shutting down")
