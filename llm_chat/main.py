"""LLM Server application with FastAPI and LangGraph workflow."""

import asyncio
import datetime
import hashlib
import json
import logging
import re
import sys
from typing import List, Optional, TypedDict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from config import settings
from llm.model import call_openrouter
from llm.planner import plan_api_calls
from rate_limit import is_rate_limited
from tools import call_hcs_api, close_api_client, get_api_tools

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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
    """Represents the state of the agent throughout the workflow.

    Attributes:
        input: The original user input string.
        language: The detected language code (e.g., 'en', 'ja').
        structured_query: A dictionary containing the processed query.
        chat_history: A list of previous messages in the conversation.
        plan: A list of planned tool calls to execute.
        tool_results: Optional list of results from tool executions.
        response: The final generated response string.
    """

    input: str
    language: str
    structured_query: dict
    chat_history: List[BaseMessage]
    plan: List[dict]
    tool_results: Optional[List[dict]]
    response: str


async def translate_and_structure_query(state: AgentState) -> dict:
    """Translate non-English queries and structure them for the planner.

    Detects the language of the user input, translates it to English if
    necessary, and extracts key entities for further processing.

    Args:
        state: The current agent state containing the user input.

    Returns:
        A dictionary with 'language' and 'structured_query' keys.
    """
    logger.info("Translating and structuring query")
    user_input = state["input"]

    # Simple language detection
    lang = (
        "ja" if any("\u3040" <= c <= "\u30ff" for c in user_input) else "en"
    )

    if lang != "en":
        prompt = f"""Translate the following user query to English. Extract key entities like channel names and desired metrics.
        User Query: "{user_input}"
        VTuber Name Map: {settings.VTUBER_NAME_MAP}
        
        Respond with a JSON object with 'english_query' and 'channels' keys.
        If the query has a page context, preserve it in the 'english_query'.
        """
        response = await call_openrouter(
            [{"role": "user", "content": prompt}], temperature=0
        )
        try:
            structured_query = json.loads(response.get("text", "{}"))
        except json.JSONDecodeError:
            structured_query = {"english_query": user_input}
    else:
        structured_query = {"english_query": user_input}

    return {"language": lang, "structured_query": structured_query}


async def planner(state: AgentState) -> dict:
    """Create a plan of tool calls based on the structured query.

    Analyzes the user's query and chat history to determine which API
    calls need to be made to fulfill the request.

    Args:
        state: The current agent state with structured query and history.

    Returns:
        A dictionary with 'plan' and updated 'chat_history' keys.
    """
    logger.info("Planning API calls")
    query = state["structured_query"].get("english_query", state["input"])
    plan = await plan_api_calls(
        query, state["chat_history"], SYSTEM_PROMPT, tools_description
    )
    # Append the user's latest message to history as a dictionary
    history = state.get("chat_history", []) + [
        {"role": "user", "content": state["input"]}
    ]
    return {"plan": plan, "chat_history": history}


async def execute_call(call_item: dict) -> dict:
    """Execute a single tool call locally or via remote API.

    Acts as a bridge between the planner and actual tool execution,
    handling both local LangChain tools and remote API calls.

    Args:
        call_item: A dictionary containing 'endpoint' and optional 'params'.

    Returns:
        The result of the tool execution, or an error dictionary if failed.
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
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Unknown API error"),
            )

        return result

    except Exception as e:
        # Return a structured error that the graph can inspect
        return {
            "error": "ToolExecutionError",
            "message": (
                f"Failed to execute tool '{endpoint}': "
                f"{getattr(e, 'detail', str(e))}"
            ),
            "endpoint": endpoint,
            "params": params,
        }


async def execute_tools(state: AgentState) -> dict:
    """Execute all planned tool calls concurrently.

    Args:
        state: The current agent state containing the plan.

    Returns:
        A dictionary with 'tool_results' containing execution results.
    """
    logger.info("Executing tools")
    plan = state.get("plan", [])
    if not plan:
        return {"tool_results": []}

    results = await asyncio.gather(*(execute_call(call) for call in plan))
    return {"tool_results": results}


async def generate_response(state: AgentState) -> dict:
    """Generate the final response by synthesizing tool results.

    Creates a response in the user's original language by combining
    the tool results with the conversation history.

    Args:
        state: The current agent state with tool results and history.

    Returns:
        A dictionary with the 'response' key containing the final answer.
    """
    logger.info("Generating final response")
    persona = settings.SYSTEM_PERSONA
    tool_results = state.get("tool_results")

    # Start with the base persona and the full chat history
    messages = [
        {"role": "system", "content": persona},
        *state["chat_history"],
    ]

    if tool_results:
        # Create a clear "tool" message block for tool output
        tool_context_message = {
            "role": "assistant",
            "content": (
                f"Here is the data I found to answer your question:\n"
                f"{json.dumps(tool_results, indent=2, ensure_ascii=False)}\n\n"
                f"Based on this data, I will now formulate a response in "
                f"{state['language']}."
            ),
        }
        messages.append(tool_context_message)

        # Add a final prompt to ensure it responds correctly
        final_instruction = {
            "role": "user",
            "content": (
                f"Great. Now, please provide your final, helpful answer in "
                f"{state['language']}."
            ),
        }
        messages.append(final_instruction)

    final_response = await call_openrouter(messages, temperature=0.7)

    response_text = final_response.get("text", "")

    logger.info(f"Final response generated: {response_text[:200]}...")

    return {
        "response": response_text
        or "I'm sorry, I encountered an issue and can't provide a response "
        "right now."
    }


tools = {t.name: t for t in get_api_tools()}
tools_description = "\n".join(
    [f"- {t.name}: {t.description}" for t in get_api_tools()]
)

SYSTEM_PROMPT = f"""
Today is {datetime.date.today()}.
"""


def sanitize_prompt(message: str) -> str:
    """Sanitize user prompts by removing forbidden patterns.

    Removes phrases from a denylist using case-insensitive regex patterns
    and logs any detected injection attempts.

    Args:
        message: The user-provided input string.

    Returns:
        A sanitized version of the string with forbidden patterns removed.
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
        # Log a security warning if forbidden patterns were found and removed
        logger.warning(
            "Potential prompt injection attempt detected. "
            f"Removed patterns: {found_patterns}. "
            f"Original message: '{original_message}'"
        )

    # Remove extra whitespace that may result from replacements
    return re.sub(r"\s+", " ", message).strip()


def parse_page_context(message: str) -> tuple[Optional[str], str]:
   """Parse page context from a user message if present.

   Extracts the page context JSON from the message and returns it
   separately from the remaining prompt text.

   Args:
       message: The user message potentially containing page context.

   Returns:
       A tuple of (page_context, cleaned_prompt) where page_context
       is the extracted context string or None, and cleaned_prompt
       is the message with context removed.
   """
   # Match [Page Context: {...}] pattern - using non-greedy match for content
   pattern = r"\[Page Context:\s*(\{.+?\})\]"
   match = re.search(pattern, message, re.DOTALL)

   if match:
       page_context = match.group(1)
       # Remove the page context from the message
       cleaned_prompt = re.sub(pattern, "", message, flags=re.DOTALL).strip()
       return page_context, cleaned_prompt

   return None, message


@app.post("/chat")
async def chat(request: Request):
   """Handle chat requests from users.

   Processes incoming chat messages, applies rate limiting, and streams
   the response back to the client using Server-Sent Events.

   Args:
       request: The FastAPI request object containing the chat data.

   Returns:
       A StreamingResponse with the chat response events.

   Raises:
       HTTPException: If the user has exceeded their rate limit (429).
   """
   data = await request.json()
   client_ip = (
       request.headers.get("CF-Connecting-IP")
       or request.client.host
       or "unknown"
   )
   user_key = hashlib.sha256(client_ip.encode()).hexdigest()[:16]

   raw_message = data.get("message", "")
   page_context, prompt_text = parse_page_context(raw_message)

   if page_context:
       logger.info(f"User {user_key} page context: {page_context}")
   
   # Log prompt on separate line, showing truncated version if too long
   prompt_preview = prompt_text[:500] if len(prompt_text) > 500 else prompt_text
   logger.info(f"User {user_key} prompt: {prompt_preview}")

   message = sanitize_prompt(raw_message)
   chat_history = data.get("chat_history", [])
   admin = data.get("admin_key") == settings.LLM_ADMIN_KEY

   if is_rate_limited(user_key, admin):
       raise HTTPException(
           status_code=429,
           detail={
               "key": "rate_limit_exceeded",
               "message": "Daily LLM limit reached",
           },
       )

   inputs = {
       "input": message,
       "chat_history": chat_history,
   }

   async def event_stream():
       """Generate Server-Sent Events for the chat response.

       Yields:
           JSON-formatted SSE data strings for status updates and responses.
       """
       final_answer_sent = False
       try:
           async for event in app_runnable.astream_events(
               inputs, version="v1"
           ):
               kind = event["event"]

               if kind == "on_chain_start":
                   node_name = event["name"]
                   status_key_map = {
                       "translator": "status_translating",
                       "planner": "status_planner",
                       "tools": "status_tools",
                       "responder": "status_responder",
                   }
                   if node_name in status_key_map:
                       status_data = {
                           "type": "status",
                           "key": status_key_map[node_name],
                       }
                       yield f"data: {json.dumps(status_data)}\n\n"

               elif kind == "on_chain_end":
                   node_name = event["name"]

                   if node_name == "responder":
                       output_data = event.get("data", {}).get("output", {})
                       final_answer = output_data.get(
                           "response", "I'm sorry, an error occurred."
                       )
                       answer_data = {
                           "type": "answer",
                           "message": final_answer,
                       }
                       yield f"data: {json.dumps(answer_data)}\n\n"
                       final_answer_sent = True

           if not final_answer_sent:
               error_data = {
                   "type": "error",
                   "message": "No response generated.",
               }
               yield f"data: {json.dumps(error_data)}\n\n"

       except Exception as e:
           logger.error(f"Error during graph stream: {e}", exc_info=True)
           if not final_answer_sent:
               error_data = {
                   "type": "error",
                   "message": "Internal error occurred.",
               }
               yield f"data: {json.dumps(error_data)}\n\n"

   return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on application shutdown.

    Closes the API client connection to prevent resource leaks.
    """
    await close_api_client()


def route_after_planning(state: AgentState) -> str:
    """Decide whether to execute tools or respond directly.

    Examines the plan generated by the planner to determine the next step
    in the workflow.

    Args:
        state: The current agent state containing the plan.

    Returns:
        The name of the next node: 'responder' if no plan exists,
        'tools' otherwise.
    """
    if not state.get("plan"):
        logger.debug("No plan generated, responding directly")
        return "responder"
    else:
        logger.debug("Plan exists, executing tools")
        return "tools"


def route_after_tools(state: AgentState) -> str:
    """Check tool execution results and route to responder.

    Evaluates whether tool execution was successful and always routes
    to the responder to generate a final answer.

    Args:
        state: The current agent state with tool results.

    Returns:
        Always returns 'responder' as the next node.
    """
    logger.debug("Assessing tool results")
    tool_results = state.get("tool_results", [])

    if not tool_results or any(
        res is None or (isinstance(res, dict) and res.get("error"))
        for res in tool_results
    ):
        logger.warning(
            "Tool execution failed or returned no data, routing to responder"
        )
    else:
        logger.debug("Tool execution succeeded, routing to responder")

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
    {"responder": "responder", "tools": "tools"},
)

# Conditional edge after tool execution
workflow.add_conditional_edges(
    "tools",
    route_after_tools,
    {"responder": "responder"},
)

workflow.add_edge("responder", END)

# Compile the graph
app_runnable = workflow.compile()