import asyncio
import os
import json
import pandas as pd
from urllib.parse import quote_plus
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Dict, List, Any, Union
import tabulate
import warnings
import tiktoken
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import queue # Not explicitly used now, but good for other async/thread patterns
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

# --- Global variables for MCP session and async management ---
mcp_session = None
agent = None
schema_manager = None
# session_lock = threading.Lock() # Not used with current design

# Async management
async_event_loop = None
async_thread = None
loop_ready_event = threading.Event()
initialization_event = threading.Event()
initialization_successful = False

# Context managers for MCP resources
_mcp_stdio_cm = None
_mcp_session_cm = None

# ThreadPoolExecutor for serializing access to run_async_query from Flask threads
executor = ThreadPoolExecutor(max_workers=1)

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

# Token management constants
MAX_TOKENS = 120000
ENCODING_NAME = "cl100k_base"

def count_tokens(string: str) -> int:
    """Count tokens in a string using tiktoken"""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    return len(encoding.encode(string))

def truncate_to_token_limit(text: str, max_tokens: int) -> str:
    """Truncate text to stay within token limit"""
    encoding = tiktoken.get_encoding(ENCODING_NAME)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)
    return truncated_text + "... [truncated due to token limit]"

class DataStructureHandler:
    """Enhanced handler for complex data structures from PostgreSQL MCP with token limit awareness"""
    
    @staticmethod
    def parse_tool_content(content: str) -> Union[Dict, List, str]:
        if not content:
            return "No content returned"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        import re
        json_pattern = r'\{.*\}|\[.*\]'
        matches = re.findall(json_pattern, content, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return content
    
    @staticmethod
    def safe_numeric_conversion(series):
        try:
            return pd.to_numeric(series)
        except (ValueError, TypeError):
            return series
    
    @staticmethod
    def format_query_results(data: Any, query_id: str = None, max_tokens: int = 10000) -> str:
        prefix = f"[Query {query_id}] " if query_id else ""
        if isinstance(data, dict):
            if 'rows' in data and 'columns' in data:
                rows = data['rows']
                columns = data['columns']
                if len(rows) > 100:
                    sample_size = min(100, len(rows))
                    rows = rows[:sample_size]
                    prefix += f"[SAMPLE of {sample_size} rows from {len(data['rows'])} total] "
                result = prefix + DataStructureHandler.format_tabular_data(rows, columns)
                return truncate_to_token_limit(result, max_tokens)
            elif 'error' in data:
                return f"{prefix}❌ Database Error: {data['error']}"
            else:
                result = prefix + DataStructureHandler.format_dict(data)
                return truncate_to_token_limit(result, max_tokens)
        elif isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                if data:
                    if len(data) > 100:
                        sample_size = min(100, len(data))
                        data = data[:sample_size]
                        prefix += f"[SAMPLE of {sample_size} items from {len(data)} total] "
                    columns = list(data[0].keys())
                    rows = [list(row.values()) for row in data]
                    result = prefix + DataStructureHandler.format_tabular_data(rows, columns)
                    return truncate_to_token_limit(result, max_tokens)
                else:
                    return f"{prefix}No rows returned"
            else:
                result = prefix + DataStructureHandler.format_list(data)
                return truncate_to_token_limit(result, max_tokens)
        elif isinstance(data, str):
            parsed = DataStructureHandler.parse_tool_content(data)
            if parsed != data:
                return DataStructureHandler.format_query_results(parsed, query_id, max_tokens)
            return truncate_to_token_limit(prefix + data, max_tokens)
        else:
            return truncate_to_token_limit(prefix + str(data), max_tokens)
    
    @staticmethod
    def format_tabular_data(rows: List[List], columns: List[str], row_limit: int = 50) -> str:
        if not rows:
            return f"📊 Columns: {', '.join(columns)}\n📭 No data returned"
        original_row_count = len(rows)
        if len(rows) > row_limit:
            rows = rows[:row_limit]
        try:
            df = pd.DataFrame(rows, columns=columns)
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = DataStructureHandler.safe_numeric_conversion(df[col])
            display_df = df.copy()
            for col in display_df.columns:
                if pd.api.types.is_numeric_dtype(display_df[col]):
                    if display_df[col].dropna().apply(lambda x: float(x).is_integer()).all():
                        display_df[col] = display_df[col].astype('Int64')
                    else:
                        display_df[col] = display_df[col].round(4)
            result_prefix = f"📊 Query Results"
            if original_row_count > row_limit:
                result_prefix += f" (showing {row_limit} of {original_row_count} rows, {len(columns)} columns):\n\n"
            else:
                result_prefix += f" ({len(rows)} rows, {len(columns)} columns):\n\n"
            result = result_prefix
            result += tabulate.tabulate(display_df, headers=display_df.columns, tablefmt='grid', showindex=False)
            if len(rows) <= row_limit and len(df.columns) <= 10:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    result += "\n\n📈 Summary Statistics:\n"
                    summary = df[numeric_cols].describe().round(4)
                    result += tabulate.tabulate(summary, headers=summary.columns, tablefmt='grid')
            return result
        except Exception as e:
            logger.warning(f"Error in advanced formatting: {e}")
            result = f"📊 Query Results ({len(rows)} rows, {len(columns)} columns):\n\n"
            result += tabulate.tabulate(rows[:row_limit], headers=columns, tablefmt='grid')
            return result
    
    @staticmethod
    def format_dict(data: Dict, max_items: int = 20) -> str:
        result = "📋 Dictionary Result:\n"
        keys = list(data.keys())[:max_items]
        for key in keys:
            value = data[key]
            if isinstance(value, (dict, list)):
                if isinstance(value, dict) and len(value) > 5:
                    result += f"  {key}: {{{len(value)} items}}\n"
                elif isinstance(value, list) and len(value) > 5:
                    result += f"  {key}: [{len(value)} items]\n"
                else:
                    json_str = json.dumps(value)
                    if len(json_str) > 100:
                        result += f"  {key}: {json_str[:100]}...\n"
                    else:
                        result += f"  {key}: {json_str}\n"
            else:
                result += f"  {key}: {value}\n"
        if len(data) > max_items:
            result += f"  ... and {len(data) - max_items} more items (truncated)\n"
        return result
    
    @staticmethod
    def format_list(data: List, max_items: int = 20) -> str:
        if len(data) == 0:
            return "📭 Empty list returned"
        result = f"📝 List Result ({len(data)} items):\n"
        shown_items = data[:max_items]
        for i, item in enumerate(shown_items):
            if isinstance(item, (dict, list)):
                if isinstance(item, dict) and len(item) > 5:
                    result += f"  {i+1}. {{{len(item)} key-value pairs}}\n"
                elif isinstance(item, list) and len(item) > 5:
                    result += f"  {i+1}. [{len(item)} items]\n"
                else:
                    json_str = json.dumps(item)
                    if len(json_str) > 100:
                        result += f"  {i+1}. {json_str[:100]}...\n"
                    else:
                        result += f"  {i+1}. {json_str}\n"
            else:
                item_str = str(item)
                if len(item_str) > 100:
                    result += f"  {i+1}. {item_str[:100]}...\n"
                else:
                    result += f"  {i+1}. {item_str}\n"
        if len(data) > max_items:
            result += f"  ... and {len(data) - max_items} more items (truncated)\n"
        return result

class SchemaManager:
    def __init__(self):
        self.tables = {}
        self.initialized = False
    
    async def initialize(self, current_agent): # Renamed agent to current_agent to avoid conflict with global
        if self.initialized:
            return
        logger.info("Initializing schema cache...")
        try:
            tables_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name"
            # Use current_agent passed to this method
            response = await process_query(current_agent, f"Execute SQL: {tables_query}", [], "SCHEMA-INIT", skip_prompt=True, schema_manager=self) # Pass self
            
            table_names = []
            if isinstance(response, dict) and "rows" in response:
                for row in response["rows"]:
                    if row and len(row) > 0:
                        table_names.append(row[0])
            
            logger.info(f"Found {len(table_names)} tables")
            
            for table_name in table_names:
                table_query = f"SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema = 'public' AND table_name = '{table_name}' ORDER BY ordinal_position"
                table_schema = await process_query(current_agent, f"Execute SQL: {table_query}", [], f"SCHEMA-{table_name}", skip_prompt=True, schema_manager=self) # Pass self
                
                if isinstance(table_schema, dict) and "rows" in table_schema:
                    columns = []
                    for row in table_schema["rows"]:
                        if row and len(row) >= 3:
                            columns.append({"name": row[0], "type": row[1], "nullable": row[2] == "YES"})
                    self.tables[table_name] = {"columns": columns, "sample_count": 0}
            
            logger.info(f"Schema cache initialized with {len(self.tables)} tables")
            self.initialized = True
        except Exception as e:
            logger.error(f"Error initializing schema: {e}", exc_info=True)
    
    def get_schema_summary(self, max_tokens=8000):
        if not self.tables:
            return "No schema information available"
        summary = [f"DATABASE SCHEMA SUMMARY ({len(self.tables)} tables):\n"]
        for table_name in sorted(self.tables.keys()):
            table_info = self.tables[table_name]
            columns = table_info.get("columns", [])
            summary.append(f"TABLE: {table_name} ({len(columns)} columns)")
            if len(columns) > 20:
                col_sample = columns[:20]
                col_info = ", ".join([f"{c['name']} ({c['type']})" for c in col_sample])
                summary.append(f"COLUMNS: {col_info}... and {len(columns) - 20} more")
            else:
                col_info = ", ".join([f"{c['name']} ({c['type']})" for c in columns])
                summary.append(f"COLUMNS: {col_info}")
            summary.append("")
        result = "\n".join(summary)
        return truncate_to_token_limit(result, max_tokens)

async def process_query(current_agent, query, history, query_counter=None, skip_prompt=False, schema_manager=None):
    logger.info(f"Processing query (skip_prompt={skip_prompt}): {query[:100]}...")
    prompt_tokens = 0
    response_tokens = 0
    query_id = f"Q{query_counter}" if query_counter else "Q"

    if skip_prompt:
        full_prompt = query # For direct SQL execution, query is the SQL
    else:
        # ... (existing prompt construction logic) ...
        initial_prompt = (
            "You are a PostgreSQL database assistant agent working within a LangChain or LangGraph system. Follow these instructions carefully:\n"
            # (Instructions as before)
            "IMPORTANT guidelines when using the `query` tool:\n"
            "- Validate all column names using `describe_table` before referencing them.\n"
            # (More guidelines as before)
            "\n"
            "**Special logic for product cost (`standard_price`)**:\n"
            # (Special logic as before)
        )
        schema_context = ""
        if schema_manager and schema_manager.initialized: # Use the passed schema_manager
            schema_context = "\n\nSCHEMA INFORMATION:\n" + schema_manager.get_schema_summary(max_tokens=8000)
        full_prompt = f"{initial_prompt}{schema_context}\n\nUser question: {query}"
        prompt_tokens = count_tokens(full_prompt)
        # Token adjustments as before

    try:
        # Use current_agent passed to this function
        agent_response = await current_agent.ainvoke({"messages": full_prompt})
        messages = agent_response["messages"]
        tool_results = {}
        sql_queries = []
        handler = DataStructureHandler()
        
        final_ai_message_content = None

        for i, msg in enumerate(messages):
            if msg.type == "ai":
                logger.info(f"LLM Output ({i}): {msg.content[:150]}...")
                final_ai_message_content = msg.content # Capture the last AI message
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        args = tool_call.get("args", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                if tool_name == "query": # If it's a query tool and arg parsing fails, assume args is the SQL
                                    sql_queries.append(args)
                                continue
                        if tool_name == "query" and isinstance(args, dict):
                            sql_query = args.get("sql") or args.get("query")
                            if sql_query:
                                if "LIMIT" not in sql_query.upper() and "SELECT" in sql_query.upper():
                                    sql_query = sql_query.rstrip(";") + " LIMIT 100;"
                                sql_queries.append(sql_query)
                                logger.info(f"SQL Query: {sql_query}")
            elif msg.type == "tool":
                tool_name = getattr(msg, "name", "unknown_tool")
                logger.info(f"Tool Response ({tool_name})")
                parsed_content = handler.parse_tool_content(msg.content)
                
                if skip_prompt and tool_name == "query": # For direct SQL, this is the result
                    return parsed_content # Return raw parsed data for schema init

                formatted_content = handler.format_query_results(parsed_content, f"{query_id}-{tool_name}", max_tokens=10000)
                tool_results[tool_name] = {'raw': parsed_content, 'formatted': formatted_content}
        
        if skip_prompt: # Should have returned earlier if successful
             logger.warning(f"Direct SQL execution for '{query[:50]}...' did not yield expected tool output.")
             return {"error": "No data returned from direct SQL execution or tool call not as expected."}


        if not tool_results and final_ai_message_content:
            # If no tools were called, the AI's last message is the answer
            return final_ai_message_content

        if not tool_results:
             return "❌ No tool actions taken or data returned. The query may have failed or returned empty results."


        response_parts = [f"🔍 Query ID: {query_id}"]
        important_tools = ["query", "list_tables", "describe_table"]
        for tool_name in important_tools:
            if tool_name in tool_results:
                response_parts.append(f"\n🔹 {tool_name.upper()} RESULT:")
                response_parts.append(tool_results[tool_name]['formatted'])
        other_tools = [t for t in tool_results.keys() if t not in important_tools]
        for tool_name in other_tools:
            result = f"\n🔹 {tool_name.upper()} RESULT:\n{tool_results[tool_name]['formatted']}"
            if count_tokens("\n".join(response_parts) + result) < MAX_TOKENS * 0.8:
                response_parts.append(result)
        if sql_queries:
            sql_section = f"\n\n🔍 EXECUTED SQL QUERIES:"
            for i, sql in enumerate(sql_queries, 1):
                sql_section += f"\n{i}. {sql}"
            if count_tokens("\n".join(response_parts) + sql_section) < MAX_TOKENS * 0.9:
                response_parts.append(sql_section)
        
        # If there were tool calls but also a final AI message, append it if it's not just "OK" or similar
        if final_ai_message_content and len(final_ai_message_content) > 10: # Heuristic for meaningful summary
            summary_section = f"\n\n📝 AI Summary:\n{final_ai_message_content}"
            if count_tokens("\n".join(response_parts) + summary_section) < MAX_TOKENS * 0.95:
                 response_parts.append(summary_section)

        response = "\n".join(response_parts)
        response_tokens = count_tokens(response)
        if response_tokens > MAX_TOKENS:
            response = truncate_to_token_limit(response, MAX_TOKENS)
            response += "\n\n⚠️ Response was truncated due to token limits."
        logger.info(f"Response token count: {response_tokens}")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return f"❌ Error: {str(e)}\n\nTry rephrasing your question or check if the database connection is working."

# --- Async Initialization and Lifecycle Management ---

def run_async_loop(loop_to_run):
    """Target function for the asyncio thread."""
    global async_event_loop
    asyncio.set_event_loop(loop_to_run)
    async_event_loop = loop_to_run # Set the global variable for others to use
    loop_ready_event.set() # Signal that the loop is set and ready
    logger.info("Asyncio event loop started and ready in dedicated thread.")
    try:
        loop_to_run.run_forever()
    except Exception as e: # Log if run_forever unexpectedly exits
        logger.error(f"Asyncio event loop crashed: {e}", exc_info=True)
    finally:
        logger.info("Asyncio event loop shutting down...")
        # More robust shutdown for tasks
        tasks = [t for t in asyncio.all_tasks(loop=loop_to_run) if t is not asyncio.current_task(loop=loop_to_run)]
        if tasks:
            logger.info(f"Cancelling {len(tasks)} outstanding tasks...")
            for task in tasks:
                task.cancel()
            try:
                loop_to_run.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
                logger.info("Outstanding tasks cancelled/gathered.")
            except Exception as e:
                logger.error(f"Error during task cancellation/gathering: {e}", exc_info=True)
        
        if hasattr(loop_to_run, "shutdown_asyncgens"): # Python 3.6+ for async generators
             try:
                loop_to_run.run_until_complete(loop_to_run.shutdown_asyncgens())
                logger.info("Async generators shutdown.")
             except Exception as e:
                logger.error(f"Error shutting down async generators: {e}", exc_info=True)
        
        loop_to_run.close()
        logger.info("Asyncio event loop closed.")

async def _initialize_mcp_components():
    """The actual async initialization logic for MCP components."""
    global mcp_session, agent, schema_manager, initialization_successful
    global _mcp_stdio_cm, _mcp_session_cm # To store context managers

    try:
        logger.info("Starting MCP component initialization in async thread...")
        warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
        
        schema_manager = SchemaManager() # Initialize schema_manager instance

        _mcp_stdio_cm = stdio_client(server_params)
        read, write = await _mcp_stdio_cm.__aenter__()
        logger.info("Stdio client context entered.")

        _mcp_session_cm = ClientSession(read, write)
        mcp_session = await _mcp_session_cm.__aenter__()
        logger.info("MCP ClientSession context entered.")
        
        await mcp_session.initialize()
        logger.info("MCP session initialized with server.")

        tools = await load_mcp_tools(mcp_session)
        agent = create_react_agent(model, tools) # agent is now global
        logger.info(f"Agent created with {len(tools)} tools.")

        # Initialize schema (uses the now global `agent`)
        await schema_manager.initialize(agent) # Pass the created agent
        
        logger.info("MCP components fully initialized.")
        initialization_successful = True

    except Exception as e:
        logger.error(f"Failed to initialize MCP components: {e}", exc_info=True)
        initialization_successful = False
    finally:
        initialization_event.set() # Signal that initialization attempt is complete

async def _shutdown_mcp_components():
    """Shuts down MCP components gracefully."""
    global _mcp_session_cm, _mcp_stdio_cm, mcp_session
    logger.info("Shutting down MCP components...")
    try:
        if _mcp_session_cm and mcp_session:
            await _mcp_session_cm.__aexit__(None, None, None)
            logger.info("MCP ClientSession context exited.")
        if _mcp_stdio_cm:
            await _mcp_stdio_cm.__aexit__(None, None, None)
            logger.info("Stdio client context exited.")
    except Exception as e:
        logger.error(f"Error during MCP component shutdown: {e}", exc_info=True)
    finally:
        _mcp_session_cm = None
        _mcp_stdio_cm = None
        mcp_session = None
        logger.info("MCP component references cleared.")

def start_background_services():
    """Starts the asyncio thread and initializes MCP components. Called once."""
    global async_thread, initialization_event, initialization_successful

    if async_thread is not None: # Prevent multiple starts
        logger.warning("Background services already started or starting.")
        return initialization_event.wait(timeout=1) # Wait briefly if already starting

    new_loop = asyncio.new_event_loop()
    async_thread = threading.Thread(target=run_async_loop, args=(new_loop,), daemon=True)
    async_thread.start()

    loop_ready_event.wait(timeout=10) # Wait for the loop to be ready in the thread
    if not loop_ready_event.is_set():
        logger.error("Asyncio event loop did not start in time.")
        if async_thread.is_alive(): # Attempt to clean up thread if it started but didn't signal
            try:
                 if new_loop.is_running(): new_loop.call_soon_threadsafe(new_loop.stop)
            except Exception: pass # loop might not be fully there
            async_thread.join(timeout=5)
        return False # Return failure

    # Schedule the initialization coroutine on the new_loop (now global async_event_loop)
    asyncio.run_coroutine_threadsafe(_initialize_mcp_components(), async_event_loop)

    logger.info("Waiting for MCP component initialization to complete...")
    initialization_event.wait(timeout=180) # Wait up to 3 minutes for init
    
    if not initialization_event.is_set():
        logger.error("MCP Initialization timed out.")
        initialization_successful = False # Ensure this is set
        # Attempt to stop the loop and thread if timeout occurs
        if async_event_loop and async_event_loop.is_running():
            async_event_loop.call_soon_threadsafe(async_event_loop.stop)
        if async_thread.is_alive():
            async_thread.join(timeout=10)
        return False

    return initialization_successful

def stop_background_services():
    """Stops the asyncio loop and cleans up resources."""
    global async_event_loop, async_thread, executor
    
    if async_event_loop and async_event_loop.is_running():
        logger.info("Requesting MCP component shutdown...")
        future = asyncio.run_coroutine_threadsafe(_shutdown_mcp_components(), async_event_loop)
        try:
            future.result(timeout=30) # Wait for MCP shutdown
            logger.info("MCP component shutdown complete.")
        except asyncio.TimeoutError:
            logger.error("MCP component shutdown timed out.")
        except Exception as e: # Other errors from _shutdown_mcp_components
            logger.error(f"Error during MCP component shutdown: {e}", exc_info=True)
        
        logger.info("Stopping asyncio event loop...")
        async_event_loop.call_soon_threadsafe(async_event_loop.stop)
    else:
        logger.info("Asyncio event loop not running or not initialized.")

    if executor:
        logger.info("Shutting down ThreadPoolExecutor...")
        executor.shutdown(wait=True)
        logger.info("ThreadPoolExecutor shut down.")

    if async_thread and async_thread.is_alive():
        logger.info("Joining asyncio thread...")
        async_thread.join(timeout=10)
        if async_thread.is_alive():
            logger.warning("Asyncio thread did not exit cleanly.")
    logger.info("Background services stopped.")

# --- Flask Routes ---
def run_async_query_task(query_text: str):
    """Wrapper to run process_query, intended for ThreadPoolExecutor."""
    global agent, schema_manager, async_event_loop # Agent and schema_manager are now global
    
    if not async_event_loop or not async_event_loop.is_running():
        logger.error("Asyncio event loop is not running for query processing.")
        return {"success": False, "error": "Asyncio service unavailable."}

    if not agent or not schema_manager or not initialization_successful:
        logger.error("MCP components not initialized for query processing.")
        return {"success": False, "error": "MCP components not ready."}

    try:
        # agent and schema_manager are global, process_query will use them.
        coro = process_query(agent, query_text, [], query_counter=1, schema_manager=schema_manager)
        future = asyncio.run_coroutine_threadsafe(coro, async_event_loop)
        result = future.result(timeout=120) # 2 minute timeout for the query processing
        return {"success": True, "result": result}
    except asyncio.TimeoutError:
        logger.error(f"Query processing timed out for: {query_text[:100]}...")
        return {"success": False, "error": "Query processing timed out."}
    except Exception as e:
        logger.error(f"Error in async query execution: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.route('/query', methods=['POST'])
def query_database():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Missing query parameter'}), 400
        query_text = data['query'].strip()
        if not query_text:
            return jsonify({'success': False, 'error': 'Query cannot be empty'}), 400
        
        if not initialization_successful or not agent or not schema_manager:
            return jsonify({'success': False, 'error': 'MCP service not initialized or ready.'}), 503
        
        # Use the executor to run the task, offloading from Flask's request thread
        future = executor.submit(run_async_query_task, query_text)
        try:
            # This timeout is for the ThreadPoolExecutor task, which includes the async timeout
            result_payload = future.result(timeout=130) # Slightly more than run_async_query_task's internal timeout
        except TimeoutError: # Timeout from future.result() if task in executor hangs
             logger.error(f"Overall query submission timed out for: {query_text[:100]}...")
             return jsonify({'success': False, 'error': 'Query submission timed out.', 'query': query_text}), 500
        
        if result_payload['success']:
            return jsonify({'success': True, 'result': result_payload['result'], 'query': query_text})
        else:
            return jsonify({'success': False, 'error': result_payload['error'], 'query': query_text}), 500
            
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}", exc_info=True)
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy' if initialization_successful else 'initializing_or_failed',
        'mcp_initialized': initialization_successful,
        'async_loop_running': async_event_loop.is_running() if async_event_loop else False,
        'timestamp': time.time()
    })

@app.route('/schema', methods=['GET'])
def get_schema_route(): # Renamed to avoid conflict
    try:
        if not schema_manager or not schema_manager.initialized:
            return jsonify({'success': False, 'error': 'Schema not initialized'}), 503
        schema_summary = schema_manager.get_schema_summary()
        return jsonify({'success': True, 'schema': schema_summary, 'table_count': len(schema_manager.tables)})
    except Exception as e:
        logger.error(f"Error getting schema: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Unhandled internal server error: {error}", exc_info=True)
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

# ... (all your existing code: imports, classes, functions, Flask app = Flask(__name__), routes) ...

# --- Gunicorn/Production Initialization ---
_app_initialized_for_gunicorn = False

def ensure_services_started_for_gunicorn():
    global _app_initialized_for_gunicorn, initialization_successful
    # Add a very early log here
    logger.info("ensure_services_started_for_gunicorn() CALLED") # NEW LOG

    if not _app_initialized_for_gunicorn:
        logger.info("Gunicorn/WSGI: Initializing application background services...")
        if start_background_services(): # This sets initialization_successful
            logger.info("Gunicorn/WSGI: Background services and MCP components initialized successfully.")
            import atexit
            atexit.register(stop_background_services)
        else:
            logger.critical("Gunicorn/WSGI: Failed to initialize application background services. Application may not function correctly.")
        _app_initialized_for_gunicorn = True
    else: # NEW LOG
        logger.info("ensure_services_started_for_gunicorn() called but _app_initialized_for_gunicorn was already True.")
    return initialization_successful

# --- THIS IS THE CRITICAL PART FOR GUNICORN PRELOAD ---
# Place this block of code AT THE END of your app.py file,
# AFTER all class/function definitions, AFTER Flask 'app' instance creation,
# and AFTER all @app.route definitions, but BEFORE the if __name__ == '__main__': block.

logger.info(f"app.py: Module-level code executing. __name__ is '{__name__}'") # NEW LOG

if __name__ != '__main__':
    logger.info(f"app.py: __name__ is '{__name__}', indicating import (likely Gunicorn --preload). Calling ensure_services_started_for_gunicorn().") # NEW LOG
    ensure_services_started_for_gunicorn()
else:
    logger.info(f"app.py: __name__ is '{__name__}', indicating direct execution.") # NEW LOG
# --- END OF CRITICAL PART ---


if __name__ == '__main__':
    logger.info("Direct Run: Initializing application background services...")
    if not _app_initialized_for_gunicorn:
        if start_background_services(): # For direct run, call start_background_services directly
            logger.info("Direct Run: Background services and MCP components initialized successfully.")
            import atexit
            atexit.register(stop_background_services)
        else:
            logger.error("Direct Run: Failed to initialize application background services. Exiting.")
            stop_background_services()
            exit(1)
    elif not initialization_successful:
        logger.error("Direct Run: Background services failed to initialize (possibly via Gunicorn path check). Exiting.")
        exit(1)

    logger.info("Starting Flask development server on http://0.0.0.0:5000")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        logger.info("Flask server (direct run) shutting down (KeyboardInterrupt)...")
    finally:
        logger.info("Flask app (direct run) has exited. Cleanup (if registered by atexit) will occur.")
