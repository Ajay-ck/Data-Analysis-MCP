import asyncio
import os
import json
import signal
import sys
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

# Server setup
server_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-postgres",
        postgres_url,
    ],
    env={"NODE_OPTIONS": "--max-old-space-size=512"}  # Limit Node.js memory usage
)

model = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)

# Thread pool for handling async operations - reduced workers for cloud deployment
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
    session = None
    try:
        logger.info(f"Starting MCP connection for query: {query[:50]}...")
        
        # Add connection timeout
        async with asyncio.timeout(25):  # Leave 5 seconds buffer before outer timeout
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    logger.info("MCP session established, initializing...")
                    await session.initialize()
                    
                    logger.info("Loading MCP tools...")
                    tools = await load_mcp_tools(session)
                    
                    logger.info(f"Creating agent with {len(tools) if tools else 0} tools...")
                    agent = create_react_agent(model, tools)

                    # Simplified prompt for faster execution
                    prompt = f"""You are a PostgreSQL database assistant. Be concise and efficient:

1. For schema exploration: Use list_tables first, then describe_table only if needed
2. For data queries: Write direct SQL using the query tool
3. Limit all results to 10 rows maximum for faster responses
4. Keep responses brief and focused

User question: {query}"""

                    logger.info("Executing agent query...")
                    # Execute the query through the agent
                    agent_response = await agent.ainvoke({"messages": prompt})
                    messages = agent_response["messages"]

                    tool_results = []
                    handler = DataStructureHandler()
                    
                    # Process all tool responses
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
                        "results": tool_results
                    }
                    
    except asyncio.TimeoutError:
        logger.error("MCP operation timed out")
        return {
            "success": False,
            "query": query,
            "error": "Database operation timed out - please try a simpler query"
        }
    except Exception as e:
        logger.error(f"MCP execution error: {str(e)}")
        return {
            "success": False,
            "query": query,
            "error": f"Database connection error: {str(e)}"
        }

def run_query_in_thread(query):
    """Run query in a dedicated thread with proper event loop"""
    def _run_query():
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            logger.info(f"Thread {threading.current_thread().name} processing query")
            return loop.run_until_complete(_execute_mcp_query(query))
        except Exception as e:
            logger.error(f"Thread execution error: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": f"Query execution failed: {str(e)}"
            }
        finally:
            # Cleanup
            try:
                # Cancel any remaining tasks
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    # Wait briefly for cancellation
                    try:
                        loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=1.0
                            )
                        )
                    except asyncio.TimeoutError:
                        pass
            except Exception as cleanup_error:
                logger.warning(f"Cleanup warning: {cleanup_error}")
            finally:
                loop.close()
    
    # Submit to thread pool with timeout
    try:
        future = executor.submit(_run_query)
        return future.result(timeout=45)  # 45 second total timeout
    except FutureTimeoutError:
        logger.error("Query thread timed out")
        return {
            "success": False,
            "query": query,
            "error": "Query timed out - operation took too long"
        }
    except Exception as e:
        logger.error(f"Thread pool error: {str(e)}")
        return {
            "success": False,
            "query": query,
            "error": f"Query processing error: {str(e)}"
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
            "GET /health": "Health check"
        },
        "example": {
            "url": "/query",
            "method": "POST",
            "body": {"query": "How many tables are there?"}
        },
        "tips": [
            "Keep queries simple for faster responses",
            "Results are limited to 10 rows for performance",
            "Complex joins may take longer to execute"
        ]
    }), 200

@app.route('/query', methods=['POST'])
def query_database():
    """Single endpoint to handle database queries"""
    start_time = time.time()
    
    try:
        # Get query from request
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

        # Process the query
        result = run_query_in_thread(user_query)
        
        # Add timing information
        execution_time = round(time.time() - start_time, 2)
        result["execution_time_seconds"] = execution_time
        
        logger.info(f"Query completed in {execution_time}s, success: {result['success']}")
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        execution_time = round(time.time() - start_time, 2)
        logger.error(f"Server error in query_database after {execution_time}s: {str(e)}")
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
        "timestamp": time.time(),
        "thread_pool_active": executor._threads is not None and len(executor._threads) > 0
    }), 200

# Test endpoint for debugging
@app.route('/test', methods=['GET'])
def test_connection():
    """Simple test endpoint to check basic functionality"""
    try:
        # Test database URL parsing
        parsed_url = urlparse(postgres_url)
        return jsonify({
            "status": "test_ok",
            "database_host": parsed_url.hostname,
            "database_port": parsed_url.port,
            "database_name": parsed_url.path.lstrip('/') if parsed_url.path else None,
            "openai_key_configured": bool(openai_api_key),
            "thread_pool_active": executor._threads is not None
        }), 200
    except Exception as e:
        return jsonify({
            "status": "test_failed",
            "error": str(e)
        }), 500

# Graceful shutdown handling
def signal_handler(signum, frame):
    logger.info("Received shutdown signal, cleaning up...")
    executor.shutdown(wait=True, timeout=10)
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    print("🚀 Starting Flask MCP API server...")
    
    # Parse URL to display connection info (without password)
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
    print("\n📋 Example request:")
    print('curl -X POST http://localhost:5000/query -H "Content-Type: application/json" -d \'{"query": "How many tables are there?"}\'\n')
    
    app.run(host='0.0.0.0', port=5000, debug=False)
