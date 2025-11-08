from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from config import settings
from tools import get_api_tools, call_hcs_api, close_api_client
from llm.model import call_openrouter
from llm.planner import plan_api_calls
from rate_limit import is_rate_limited
from fastapi.middleware.cors import CORSMiddleware
from typing import TypedDict, List, Optional
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
import asyncio
import json, datetime
import logging
import re
import hashlib
import sys

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

app = FastAPI(title="HoloChatStats LLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "https://holochatstats.info",
        "https://llm.holochatstats.info",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentState(TypedDict):
    """
    Represents the state of our agent.
    """
    input: str
    language: str
    structured_query: dict
    chat_history: List[BaseMessage]
    plan: List[dict]
    tool_results: Optional[List[dict]]
    response: str

async def translate_and_structure_query(state: AgentState):
    """
    Translates non-English queries and structures them for the planner.
    """
    print("---TRANSLATING & STRUCTURING QUERY---")
    user_input = state['input']
    
    # Simple language detection
    lang = "ja" if any("\u3040" <= c <= "\u30ff" for c in user_input) else "en"
    
    if lang != "en":
        prompt = f"""Translate the following user query to English. Extract key entities like channel names and desired metrics.
        User Query: "{user_input}"
        VTuber Name Map: {settings.VTUBER_NAME_MAP}
        
        Respond with a JSON object with 'english_query' and 'channels' keys.
        """
        response = await call_openrouter([{"role": "user", "content": prompt}], temperature=0)
        try:
            structured_query = json.loads(response.get("text", "{}"))
        except json.JSONDecodeError:
            structured_query = {"english_query": user_input} # Fallback
    else:
        structured_query = {"english_query": user_input}

    return {"language": lang, "structured_query": structured_query}

async def planner(state: AgentState):
    """
    Creates a plan of tool calls to execute based on the structured query.
    """
    print("---PLANNING---")
    query = state['structured_query'].get('english_query', state['input'])
    plan = await plan_api_calls(
        query,
        state['chat_history'],
        SYSTEM_PROMPT,
        tools_description
    )
    # Append the user's latest message to history as a dictionary
    history = state.get('chat_history', []) + [{"role": "user", "content": state['input']}]
    return {"plan": plan, "chat_history": history}

async def execute_call(call_item: dict):
    """
    Executes a single tool call, either locally or via the remote API.
    This is the bridge between the planner and the actual tool execution.
    """
    endpoint = call_item.get("endpoint")
    params = call_item.get("params", {})

    if not isinstance(params, dict):
        params = {}

    try:
        # Check if the endpoint is a locally defined LangChain tool
        if endpoint in tools:
            tool = tools[endpoint]
            # Assumes the tool is async and can be called with params
            result = await tool.ainvoke(params)
        # Otherwise, assume it's a remote API call to the web server
        else:
            result = await call_hcs_api(endpoint, params)

        # Normalize the response if it's an error
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown API error"))

        return result

    except Exception as e:
        # Return a structured error that the graph can inspect
        return {
            "error": "ToolExecutionError",
            "message": f"Failed to execute tool '{endpoint}': {getattr(e, 'detail', str(e))}",
            "endpoint": endpoint,
            "params": params
        }


async def execute_tools(state: AgentState):
    """
    Executes the planned tool calls.
    """
    print("---EXECUTING TOOLS---")
    plan = state.get("plan", [])
    if not plan:
        return {"tool_results": []}
    
    # This logic is adapted from your original main.py
    results = await asyncio.gather(*(execute_call(call) for call in plan))
    return {"tool_results": results}

async def generate_response(state: AgentState):
    """
    Generates the final response by synthesizing tool results (if any)
    and responding in the user's original language.
    """
    print("---GENERATING FINAL RESPONSE---")
    persona = settings.SYSTEM_PERSONA
    tool_results = state.get("tool_results")
    
    # Start with the base persona and the full chat history
    messages = [
        {"role": "system", "content": persona},
        *state['chat_history'] 
    ]

    if tool_results:
        # If we have tool results, create a clear "tool" message block
        # This is a more standard and reliable way to feed tool output to a model
        tool_context_message = {
            "role": "assistant",
            "content": f"""Here is the data I found to answer your question:
{json.dumps(tool_results, indent=2, ensure_ascii=False)}

Based on this data, I will now formulate a response in {state['language']}.
"""
        }
        messages.append(tool_context_message)
        
        # Add a final prompt to ensure it responds correctly
        final_instruction = {"role": "user", "content": f"Great. Now, please provide your final, helpful answer in {state['language']}."}
        messages.append(final_instruction)
    
    # If there were no tool results, the history is already complete and the model will respond conversationally.

    final_response = await call_openrouter(messages, temperature=0.7)
    
    response_text = final_response.get("text", "")

    print(f"---FINAL RESPONSE GENERATED---\n{response_text}\n---------------------------")

    return {"response": response_text or "I'm sorry, I encountered an issue and can't provide a response right now."}

tools = {t.name: t for t in get_api_tools()}
tools_description = "\n".join([f"- {t.name}: {t.description}" for t in get_api_tools()])


SYSTEM_PROMPT = f"""
Today is {datetime.date.today()}.
"""

def sanitize_prompt(message: str) -> str:
    """
    Sanitizes a user's prompt by removing phrases from a denylist
    using case-insensitive regex patterns. Logs any attempts.

    Args:
        message: The user-provided input string.

    Returns:
        A sanitized version of the string.
    """
    if not settings.PROMPT_SANITIZATION_ENABLED:
        return message

    original_message = message
    found_patterns = []

    for pattern in settings.PROMPT_DENYLIST_PATTERNS:
        # Check if the pattern exists before performing a replacement
        if re.search(pattern, message):
            found_patterns.append(pattern)
            # re.sub can replace all occurrences of the pattern
            message = re.sub(pattern, "", message)

    if found_patterns:
        # Log a security warning if any forbidden patterns were found and removed
        logger.warning(
            "Potential prompt injection attempt detected. "
            f"Removed patterns: {found_patterns}. "
            f"Original message: '{original_message}'"
        )

    # Remove extra whitespace that may result from replacements
    return re.sub(r'\s+', ' ', message).strip()

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    client_ip = request.headers.get("CF-Connecting-IP") or request.client.host or "unknown"
    user_key = hashlib.sha256(client_ip.encode()).hexdigest()[:16]

    logger.info(f"User {user_key} prompt: {data.get('message', '')[:500]}")

    message = sanitize_prompt(data.get("message", ""))
    chat_history = data.get("chat_history", [])
    admin = data.get("admin_key") == settings.LLM_ADMIN_KEY

    if is_rate_limited(user_key, admin):
        raise HTTPException(status_code=429, detail={"key": "rate_limit_exceeded", "message": "Daily LLM limit reached"})

    inputs = {
        "input": message,
        "chat_history": chat_history,
    }

    async def event_stream():
        final_answer_sent = False
        try:
            async for event in app_runnable.astream_events(inputs, version="v1"):
                kind = event["event"]

                if kind == "on_chain_start":
                    node_name = event["name"]
                    status_key_map = {
                        "translator": "status_translating",
                        "planner": "status_planner",
                        "tools": "status_tools",
                        "responder": "status_responder"
                    }
                    if node_name in status_key_map:
                        yield f"data: {json.dumps({'type':'status','key':status_key_map[node_name]})}\n\n"

                elif kind == "on_chain_end":
                    node_name = event["name"]

                    if node_name == "responder":
                        output_data = event.get("data", {}).get("output", {})
                        final_answer = output_data.get(
                            "response", "I'm sorry, an error occurred."
                        )
                        yield f"data: {json.dumps({'type':'answer','message':final_answer})}\n\n"
                        final_answer_sent = True

            if not final_answer_sent:
                yield f"data: {json.dumps({'type':'error','message':'No response generated.'})}\n\n"

        except Exception as e:
            logger.error(f"Error during graph stream: {e}", exc_info=True)
            if not final_answer_sent:
                yield f"data: {json.dumps({'type':'error','message':'Internal error occurred.'})}\n\n"


    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.on_event("shutdown")
async def shutdown_event():
    await close_api_client()

def route_after_planning(state: AgentState):
    """
    Decides whether to call tools or respond directly.
    """
    if not state.get("plan"):
        print("---DECISION: NO PLAN, RESPONDING DIRECTLY---")
        return "responder"
    else:
        print("---DECISION: PLAN EXISTS, EXECUTING TOOLS---")
        return "tools"

def route_after_tools(state: AgentState):
    """
    Checks if tool execution was successful.
    In either case, it routes to the responder to generate a final answer.
    """
    print("---ASSESSING TOOL RESULTS---")
    tool_results = state.get("tool_results", [])

    if not tool_results or any(res is None or (isinstance(res, dict) and res.get("error")) for res in tool_results):
        print("---DECISION: TOOL FAILED OR RETURNED NO DATA. ROUTING TO RESPONDER.---")
    else:
        print("---DECISION: TOOL SUCCEEDED. ROUTING TO RESPONDER.---")

    return "responder"

# Build the state graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("translator", translate_and_structure_query)
workflow.add_node("planner", planner)
workflow.add_node("tools", execute_tools)
workflow.add_node("responder", generate_response)

# Define edges and entry point
workflow.set_entry_point("translator")
workflow.add_edge("translator", "planner")

# Conditional edge after planning
workflow.add_conditional_edges(
    "planner",
    route_after_planning,
    {
        "responder": "responder",
        "tools": "tools"
    }
)

# Conditional edge after tool execution
workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {
        "responder": "responder"
    }
)

workflow.add_edge("responder", END)

# Compile the graph
app_runnable = workflow.compile()