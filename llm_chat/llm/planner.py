import json
import re
from llm.model import call_openrouter
from config import settings

async def plan_api_calls(message, chat_history, system_prompt, tools_description, max_retries=2):
    """
    Ask the model to plan API calls. Retries with stricter formatting if invalid/empty.
    """

    plan_instructions = f"""
You are an API planner that decides which data tools to call to answer the user's question.

You must respond **only** with a JSON array, matching *exactly* this schema:
[
  {{
    "endpoint": "<tool_name>",
    "params": {{
      "param1": "value1",
      "param2": "value2"
    }}
  }}
]

Strict rules:
- Output **only** valid JSON — no explanations or natural language.
- Never include comments or markdown.
- Do not include trailing commas.
- Ensure the JSON is syntactically complete.
- The assistant's name is Eri. Do not plan API calls if the user is addressing Eri casually (e.g., "Hey Eri, how's it going?").
- If the message is casual chitchat, small talk, or doesn't clearly request data (e.g., "What's your favorite anime?" or "Hi Eri!"), output an empty list [].
- Only plan API calls for questions that directly relate to the available tools (e.g., "What are Pekora's streaming hours last month?").
- If you are unsure, or the question can't be satisfied with the available tools, output an empty list [].
- If the user mentions a VTuber by nickname or in Japanese, use this mapping to find the correct channel parameter.
{settings.VTUBER_NAME_MAP}

Available tools:
{tools_description}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": plan_instructions},
        *chat_history,
        {"role": "user", "content": message},
    ]

    for attempt in range(max_retries):
        llm_resp = await call_openrouter(messages, temperature=0)
        text = llm_resp.get("text", "").strip()
        print(f"🔍 Raw LLM plan output (attempt {attempt+1}): {text[:300]}")

        # Extract JSON if the model added extra text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        json_str = match.group(0) if match else text

        try:
            plan = json.loads(json_str)
            if isinstance(plan, list):
                return plan
        except Exception:
            pass

        # Retry with stricter instruction
        messages.append({
            "role": "system",
            "content": "Your previous response was invalid. Output ONLY valid JSON, no commentary."
        })

    print("⚠️ Plan output invalid or empty after retries.")
    return []
