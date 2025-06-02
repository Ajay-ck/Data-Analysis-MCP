import asyncio
import os
import json
import signal
import sys
import subprocess
import shutil
from urllib.parse import quote_plus, urlparse
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import logging
import time
import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Load API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

# PostgreSQL connection URL from .env
postgres_url = os.getenv("DATABASE_URL")
if not postgres_url:
    raise ValueError("DATABASE_URL not found in .env file")

# Test direct PostgreSQL connection
def test_postgres_connection():
    """Test direct PostgreSQL connection"""
    try:
        conn = psycopg2.connect(postgres_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        return {"success": True, "version": version[0] if version else "Unknown"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Test Node.js and npm availability
def test_node_environment():
    """Test if Node.js and npm are available"""
    results = {}
    
    # Test Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=10)
        results['node_version'] = result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
        results['node_available'] = result.returncode == 0
    except Exception as e:
        results['node_version'] = f"Not available: {str(e)}"
        results['node_available'] = False
    
    # Test npm
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True, timeout=10)
        results['npm_version'] = result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
        results['npm_available'] = result.returncode == 0
    except Exception as e:
        results['npm_version'] = f"Not available: {str(e)}"
        results['npm_available'] = False
    
    # Test npx
    try:
        result = subprocess.run(['npx', '--version'], capture_output=True, text=True, timeout=10)
        results['npx_version'] = result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
        results['npx_available'] = result.returncode == 0
    except Exception as e:
        results['npx_version'] = f"Not available: {str(e)}"
        results['npx_available'] = False
    
    return results

# Test MCP server directly
async def test_mcp_server():
    """Test MCP server connection directly"""
    try:
        logger.info("Testing MCP server connection...")
        
        server_params = StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-postgres",
                postgres_url,
            ],
            env={"NODE_OPTIONS": "--max-old-space-size=512"}
        )
        
        async with asyncio.timeout(15):  # 15 second timeout for testing
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    # Test listing resources
                    resources = await session.list_resources()
                    return {
                        "success": True,
                        "resources_count": len(resources.resources) if resources else 0,
                        "message": "MCP server connection successful"
                    }
                    
    except asyncio.TimeoutError:
        return {"success": False, "error": "MCP server connection timed out"}
    except Exception as e:
        return {"success": False, "error": f"MCP server error: {str(e)}"}

# Alternative: Direct SQL execution
def execute_direct_sql(query):
    """Execute SQL directly without MCP (fallback method)"""
    try:
        conn = psycopg2.connect(postgres_url)
        cursor = conn.cursor()
        
        # Handle common queries
        if "how many tables" in query.lower():
            cursor.execute("""
                SELECT COUNT(*) as table_count 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return {
                "success": True,
                "query": query,
                "results": [{
                    "tool": "direct_sql",
                    "result": {
                        "type": "table",
                        "columns": ["table_count"],
                        "rows": [[result[0]]] if result else [[0]],
                        "row_count": 1
                    }
                }]
            }
        
        elif "list tables" in query.lower() or "show tables" in query.lower():
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
                LIMIT 10
            """)
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            return {
                "success": True,
                "query": query,
                "results": [{
                    "tool": "direct_sql",
                    "result": {
                        "type": "table",
                        "columns": ["table_name"],
                        "rows": [[row[0]] for row in results],
                        "row_count": len(results)
                    }
                }]
            }
        
        else:
            # For other queries, try to execute directly (be careful!)
            cursor.execute(query)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
                return {
                    "success": True,
                    "query": query,
                    "results": [{
                        "tool": "direct_sql",
                        "result": {
                            "type": "table",
                            "columns": columns,
                            "rows": [list(row) for row in rows[:10]],  # Limit to 10 rows
                            "row_count": len(rows)
                        }
                    }]
                }
            else:
                cursor.close()
                conn.close()
                return {
                    "success": True,
                    "query": query,
                    "results": [{
                        "tool": "direct_sql",
                        "result": {
                            "type": "string",
                            "data": "Query executed successfully"
                        }
                    }]
                }
                
    except Exception as e:
        return {
            "success": False,
            "query": query,
            "error": f"Direct SQL error: {str(e)}"
        }

model = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-worker")

class DataStructureHandler:
    """Simple handler for formatting MCP results"""
    
    @staticmethod
    def parse_tool_content(content: str):
        """Parse tool content with JSON fallback"""
        if not content:
            return "No content returned"
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    
    @staticmethod
    def format_query_results(data):
        """Simple formatting for query results"""
        if isinstance(data, dict):
            if 'rows' in data and 'columns' in data:
                return {
                    "type": "table",
                    "columns": data['columns'],
                    "rows": data['rows'],
                    "row_count": len(data['rows'])
                }
            elif 'error' in data:
                return {"type": "error", "message": data['error']}
            else:
                return {"type": "dict", "data": data}
        
        elif isinstance(data, list):
            return {"type": "list", "data": data, "count": len(data)}
        
        else:
            return {"type": "string", "data": str(data)}

async def _execute_mcp_query(query):
    """Execute the actual MCP query with better error handling"""
    try:
        logger.info(f"Starting MCP connection for query: {query[:50]}...")
        
        server_params = StdioServerParameters(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-postgres",
                postgres_url,
            ],
            env={"NODE_OPTIONS": "--max-old-space-size=512"}
        )
        
        async with asyncio.timeout(20):  # 20 second timeout
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    logger.info("MCP session established, initializing...")
                    await session.initialize()
                    
                    logger.info("Loading MCP tools...")
                    tools = await load_mcp_tools(session)
                    
                    logger.info(f"Creating agent with {len(tools) if tools else 0} tools...")
                    agent = create_react_agent(model, tools)

                    prompt = f"""You are a PostgreSQL database assistant. Be concise:

1. For schema queries: Use list_tables first
2. For data queries: Write direct SQL with LIMIT 5
3. Keep responses brief

User question: {query}"""

                    logger.info("Executing agent query...")
                    agent_response = await agent.ainvoke({"messages": prompt})
                    messages = agent_response["messages"]

                    tool_results = []
                    handler = DataStructureHandler()
                    
                    for msg in messages:
                        if msg.type == "tool":
                            tool_name = getattr(msg, "name", "unknown_tool")
                            parsed_content = handler.parse_tool_content(msg.content)
                            formatted_content = handler.format_query_results(parsed_content)
                            
                            tool_results.append({
                                "tool": tool_name,
                                "result": formatted_content
                            })

                    logger.info(f"Query completed successfully, {len(tool_results)} results")
                    return {
                        "success": True,
                        "query": query,
                        "results": tool_results,
                        "method": "mcp"
                    }
                    
    except asyncio.TimeoutError:
        logger.error("MCP operation timed out")
        return {
            "success": False,
            "query": query,
            "error": "MCP connection timed out - trying direct SQL fallback",
            "method": "mcp_timeout"
        }
    except Exception as e:
        logger.error(f"MCP execution error: {str(e)}")
        return {
            "success": False,
            "query": query,
            "error": f"MCP error: {str(e)} - trying direct SQL fallback",
            "method": "mcp_error"
        }

def run_query_with_fallback(query):
    """Run query with MCP, fallback to direct SQL if needed"""
    def _run_query():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Try MCP first
            logger.info("Attempting MCP query...")
            result = loop.run_until_complete(_execute_mcp_query(query))
            
            # If MCP failed, try direct SQL
            if not result["success"]:
                logger.info("MCP failed, trying direct SQL...")
                direct_result = execute_direct_sql(query)
                if direct_result["success"]:
                    direct_result["method"] = "direct_sql_fallback"
                    direct_result["mcp_error"] = result["error"]
                    return direct_result
                else:
                    return result  # Return original MCP error
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            # Try direct SQL as last resort
            logger.info("Trying direct SQL as last resort...")
            direct_result = execute_direct_sql(query)
            if direct_result["success"]:
                direct_result["method"] = "direct_sql_emergency"
                direct_result["original_error"] = str(e)
                return direct_result
            else:
                return {
                    "success": False,
                    "query": query,
                    "error": f"All methods failed. Original: {str(e)}, Direct SQL: {direct_result.get('error', 'Unknown')}"
                }
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    try:
                        loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=1.0
                            )
                        )
                    except asyncio.TimeoutError:
                        pass
            except Exception:
                pass
            finally:
                loop.close()
    
    try:
        future = executor.submit(_run_query)
        return future.result(timeout=35)  # 35 second total timeout
    except FutureTimeoutError:
        logger.error("Query thread timed out")
        # Try direct SQL as absolute last resort
        try:
            return execute_direct_sql(query)
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": f"Complete timeout - all methods exhausted: {str(e)}"
            }

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "service": "MCP PostgreSQL API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /query": "Send database queries",
            "GET /health": "Health check",
            "GET /debug": "Debug environment and connections",
            "GET /test": "Test configuration"
        },
        "example": {
            "url": "/query",
            "method": "POST",
            "body": {"query": "How many tables are there?"}
        }
    }), 200

@app.route('/debug', methods=['GET'])
def debug_environment():
    """Debug endpoint to check environment and connections"""
    debug_info = {
        "timestamp": time.time(),
        "environment": {}
    }
    
    # Test Node.js environment
    debug_info["node_environment"] = test_node_environment()
    
    # Test PostgreSQL connection
    debug_info["postgres_connection"] = test_postgres_connection()
    
    # Test MCP server (run in thread to avoid blocking)
    def test_mcp():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(test_mcp_server())
        finally:
            loop.close()
    
    try:
        future = executor.submit(test_mcp)
        debug_info["mcp_server"] = future.result(timeout=20)
    except Exception as e:
        debug_info["mcp_server"] = {"success": False, "error": f"MCP test failed: {str(e)}"}
    
    # Environment variables (safe ones)
    debug_info["environment"] = {
        "openai_key_configured": bool(openai_api_key),
        "database_url_configured": bool(postgres_url),
        "python_version": sys.version,
        "working_directory": os.getcwd()
    }
    
    return jsonify(debug_info), 200

@app.route('/query', methods=['POST'])
def query_database():
    """Single endpoint to handle database queries"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'query' field in request body"
            }), 400

        user_query = data['query'].strip()
        if not user_query:
            return jsonify({
                "success": False,
                "error": "Query cannot be empty"
            }), 400

        logger.info(f"Received query request: {user_query[:100]}...")

        # Process the query with fallback
        result = run_query_with_fallback(user_query)
        
        execution_time = round(time.time() - start_time, 2)
        result["execution_time_seconds"] = execution_time
        
        logger.info(f"Query completed in {execution_time}s, success: {result['success']}, method: {result.get('method', 'unknown')}")
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        execution_time = round(time.time() - start_time, 2)
        logger.error(f"Server error after {execution_time}s: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}",
            "execution_time_seconds": execution_time
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "MCP PostgreSQL API",
        "timestamp": time.time()
    }), 200

@app.route('/test', methods=['GET'])
def test_connection():
    """Simple test endpoint"""
    try:
        parsed_url = urlparse(postgres_url)
        return jsonify({
            "status": "test_ok",
            "database_host": parsed_url.hostname,
            "database_port": parsed_url.port,
            "database_name": parsed_url.path.lstrip('/') if parsed_url.path else None,
            "openai_key_configured": bool(openai_api_key)
        }), 200
    except Exception as e:
        return jsonify({
            "status": "test_failed",
            "error": str(e)
        }), 500

# Graceful shutdown
def signal_handler(signum, frame):
    logger.info("Received shutdown signal, cleaning up...")
    executor.shutdown(wait=True, timeout=10)
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    print("🚀 Starting Flask MCP API server with debug capabilities...")
    
    try:
        parsed_url = urlparse(postgres_url)
        safe_url = f"{parsed_url.scheme}://{parsed_url.username}:***@{parsed_url.hostname}:{parsed_url.port}{parsed_url.path}"
        print(f"🔗 PostgreSQL URL: {safe_url}")
    except:
        print("🔗 PostgreSQL URL: [URL parsed from DATABASE_URL]")
    
    print("📍 Endpoints:")
    print("  GET / - API documentation")
    print("  POST /query - Send database queries")
    print("  GET /health - Health check")
    print("  GET /test - Test configuration")
    print("  GET /debug - Debug environment")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
