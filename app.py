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
from concurrent.futures import ThreadPoolExecutor
import logging

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
)

model = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_api_key)

# Thread pool for handling async operations
executor = ThreadPoolExecutor(max_workers=2)

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

async def process_mcp_query(query):
    """Process database query through MCP and return results"""
    try:
        logger.info(f"Processing query: {query[:100]}...")
        
        # Set shorter timeout for MCP operations
        async with asyncio.timeout(30):  # 30 second timeout
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)
                    agent = create_react_agent(model, tools)

                    # Enhanced prompt for database queries
                    prompt = f"""You are a PostgreSQL database assistant. Follow these instructions:
1. Begin by exploring the database schema using list_tables and describe_table tools
2. Analyze the user's query to understand what data they need
3. Write appropriate SQL queries using the query tool
4. Always use explicit JOINs and proper WHERE clauses
5. Limit results to 50 rows maximum (keep it small for faster responses)
6. For sales/invoice queries, apply appropriate state filters (e.g., state = 'sale' for completed sales)

User question: {query}"""

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
        logger.error(f"Query timed out: {query[:100]}")
        return {
            "success": False,
            "query": query,
            "error": "Query timed out (30s limit exceeded)"
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {
            "success": False,
            "query": query,
            "error": str(e)
        }

def run_async_query_safe(query):
    """Safely run async query with proper event loop handling"""
    try:
        # Check if there's already an event loop in this thread
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to use a different approach
            logger.warning("Already in async context, using thread executor")
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(run_in_new_loop, query)
                return future.result(timeout=45)  # 45 second timeout
        except RuntimeError:
            # No event loop running, safe to create one
            return run_in_new_loop(query)
            
    except Exception as e:
        logger.error(f"Error in run_async_query_safe: {str(e)}")
        return {
            "success": False,
            "query": query,
            "error": f"Query execution error: {str(e)}"
        }

def run_in_new_loop(query):
    """Run query in a new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_mcp_query(query))
    finally:
        try:
            # Clean up any pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except:
            pass
        finally:
            loop.close()

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "service": "MCP PostgreSQL API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Send database queries",
            "GET /health": "Health check"
        },
        "example": {
            "url": "/query",
            "method": "POST",
            "body": {"query": "How many tables are there?"}
        }
    }), 200

@app.route('/query', methods=['POST'])
def query_database():
    """Single endpoint to handle database queries"""
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

        # Process the query with timeout
        try:
            result = run_async_query_safe(user_query)
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Query processing failed: {str(e)}"
            }), 500
        
        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        logger.error(f"Server error in query_database: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "MCP PostgreSQL API",
        "timestamp": asyncio.get_event_loop().time() if asyncio._get_running_loop() else "N/A"
    }), 200

# Graceful shutdown handling
def signal_handler(signum, frame):
    logger.info("Received shutdown signal, cleaning up...")
    executor.shutdown(wait=True)
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
    print("\n📋 Example request:")
    print('curl -X POST http://localhost:5000/query -H "Content-Type: application/json" -d \'{"query": "How many tables are there?"}\'\n')
    
    app.run(host='0.0.0.0', port=5000, debug=False)
