from fastapi import FastAPI, Request, HTTPException
from config import settings
from tools import get_api_tools, call_hcs_api, close_api_client
from llm.model import call_openrouter
from llm.planner import plan_api_calls
from rate_limit import is_rate_limited
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json, datetime
import logging
import re

logger = logging.getLogger(__name__)
app = FastAPI(title="HoloChatStats LLM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5000",
        "http://localhost:5000",
        "https://holochatstats.info",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    user_key = request.headers.get("CF-Connecting-IP", request.remote_addr)

    # Log user prompt
    logger.info(f"User {user_key} prompt: {data.get('message', '')[:500]}")

    message = sanitize_prompt(data.get("message", ""))
    chat_history = data.get("chat_history", [])
    admin = data.get("admin_key") == settings.LLM_ADMIN_KEY
    persona = settings.SYSTEM_PERSONA_ADMIN if admin else settings.SYSTEM_PERSONA

    if is_rate_limited(user_key, admin):
        raise HTTPException(status_code=429, detail="Daily LLM limit reached")


    # Step 1: Plan API calls
    plan = await plan_api_calls(message, chat_history, SYSTEM_PROMPT, tools_description)

    # Limit total calls for non-admins
    if not admin:
        plan = plan[:getattr(settings, "MAX_API_CALLS_PER_PROMPT", 3)]

    async def execute_call(call_item):
        endpoint = call_item.get("endpoint")
        params = call_item.get("params", {})

        if not isinstance(params, dict):
            params = {}

        try:
            # Local LangChain / Python tool
            if endpoint in tools:
                tool = tools[endpoint]

                if hasattr(tool, "arun"):
                    result = await tool.arun(tool_input=params)
                elif hasattr(tool, "ainvoke"):
                    result = await tool.ainvoke(params)
                elif callable(tool):
                    result = await tool(**params)
                else:
                    raise TypeError(f"Unsupported tool type for {endpoint}")

            # Remote HoloChatStats API call
            else:
                result = await call_hcs_api(endpoint, params)

            # Normalize response
            if isinstance(result, dict) and result.get("error"):
                raise HTTPException(status_code=500, detail=result["message"])

            return result

        except HTTPException as e:
            return {
                "error": "API_ERROR",
                "message": f"API call failed: {getattr(e, 'detail', str(e))}",
                "endpoint": endpoint,
                "params": params
            }

        except Exception as e:
            return {
                "error": "EXECUTION_ERROR",
                "message": f"Tool execution failed: {str(e)}",
                "endpoint": endpoint,
                "params": params
            }

    # Run all calls concurrently
    results = await asyncio.gather(*(execute_call(call) for call in plan))


    # Step 2: Feed results + final answer
    messages = [
        {"role": "system", "content": f"{persona}\n{SYSTEM_PROMPT}"},
        *chat_history,
    ]

    if results:
        messages.append({
            "role": "system",
            "content": (
                "You have just received structured API results from your tools. "
                "Use them to compose a concise, helpful answer to the user's question. "
                "Focus only on the most relevant insights."
            )
        })
        messages.append({
            "role": "assistant",
            "content": f"TOOL_RESULTS: {json.dumps(results, ensure_ascii=False)}"
        })
        messages.append({
            "role": "user",
            "content": "Now, based on those TOOL_RESULTS, give your final answer to my question."
        })
    else:
        messages.append({
            "role": "system",
            "content": (
                "No API data was required. Answer the user's question directly and naturally as Eri. "
                "Do not mention APIs, tools, or any technical details (such as TOOL_RESULTS)."
            )
        })


    final_answer = await call_openrouter(messages + [
        {"role": "system", "content": "Respond with a clear, concise text answer — not JSON."}
    ])

    return {
        "answer": final_answer.get("text", ""),
        "api_calls": plan,
        "results": results
    }

@app.on_event("shutdown")
async def shutdown_event():
    await close_api_client()
