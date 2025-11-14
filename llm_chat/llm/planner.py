import json
import re
from typing import Optional
from llm.model import call_openrouter
from tool_store import tool_store

async def get_relevant_context(query: str) -> str:
    """
    Get relevant context from the knowledge base.
    
    Args:
        query: The user's query
        
    Returns:
        A formatted string with relevant knowledge
    """
    # Search for relevant knowledge
    knowledge_items = await tool_store.search_knowledge(
        query,
        top_k=5,
        similarity_threshold=0.35
    )
    
    if not knowledge_items:
        return ""
    
    # Format knowledge for the prompt
    context_parts = []
    for item in knowledge_items:
        context_parts.append(f"- {item.content}")
    
    return "\n".join(context_parts)

async def get_vtuber_canonical_name(query: str) -> Optional[str]:
    """
    Try to find the canonical VTuber name from aliases in the query.
    
    Args:
        query: The user's query
        
    Returns:
        Canonical name if found, None otherwise
    """
    # Search specifically in vtuber_alias category
    aliases = await tool_store.search_knowledge(
        query,
        category="vtuber_alias",
        top_k=1,
        similarity_threshold=0.5
    )
    
    if aliases and aliases[0].metadata:
        return aliases[0].metadata.get("canonical_name")
    
    return None

async def get_relevant_tools_description(query: str, max_tools: int = 10) -> str:
    """
    Get descriptions of only the most relevant tools for the query.
    
    Args:
        query: The user's query
        max_tools: Maximum number of tools to include
        
    Returns:
        A formatted string with relevant tool descriptions
    """
    # Search for relevant tools
    relevant_tools = await tool_store.search_relevant_tools(
        query, 
        top_k=max_tools,
        similarity_threshold=0.25
    )
    
    if not relevant_tools:
        # Fallback: get some default tools if no relevant ones found
        all_tool_names = await tool_store.get_all_tool_names()
        default_tools = [
            "get_channel_names",
            "get_monthly_streaming_hours",
            "search_highlights"
        ]
        relevant_tools = []
        for name in default_tools[:3]:
            if name in all_tool_names:
                tool = await tool_store.get_tool_by_name(name)
                if tool:
                    relevant_tools.append(tool)
    
    # Format tool descriptions for the prompt
    tool_descriptions = []
    for tool in relevant_tools:
        params_str = ", ".join([
            f"{name}: {info.get('type', 'str')}"
            for name, info in tool.parameters.items()
        ])
        tool_descriptions.append(
            f"- {tool.name}({params_str}): {tool.description}"
        )
    
    return "\n".join(tool_descriptions)

async def plan_api_calls(
    message: str, 
    chat_history: list, 
    system_prompt: str, 
    tools_description: str = None,
    max_retries: int = 2
):
    """
    Ask the model to plan API calls using only relevant tools and knowledge.
    """
    # Get relevant tools and knowledge
    relevant_tools = await get_relevant_tools_description(message)
    relevant_knowledge = await get_relevant_context(message)
    
    # Build the context section
    context_section = ""
    if relevant_knowledge:
        context_section = f"""
Relevant knowledge about what you can and cannot do:
{relevant_knowledge}
"""
    
    plan_instructions = f"""
You are an API planner that decides which data tools to call to answer the user's question.

{context_section}

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
- The assistant's name is Eri. Do not plan API calls if the user is addressing Eri casually.
- If the message is casual chitchat or doesn't clearly request data, output an empty list [].
- Only plan API calls for questions that directly relate to the available tools.
- If you are unsure, or the question can't be satisfied with the available tools, output an empty list [].
- If the user asks about capabilities (what you can/cannot do), output an empty list [] - you will answer from knowledge.
- If the message contains a page context, use it to inform your planning.

Available tools for this query:
{relevant_tools}
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

        # Extract JSON if the model added extra text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        json_str = match.group(0) if match else text

        try:
            plan = json.loads(json_str)
            if isinstance(plan, list):
                # Validate that planned tools actually exist
                valid_plan = []
                for call in plan:
                    if "endpoint" in call:
                        tool = await tool_store.get_tool_by_name(call["endpoint"])
                        if tool:
                            valid_plan.append(call)
                        else:
                            # Try to find a similar tool name
                            all_tools = await tool_store.get_all_tool_names()
                            similar = [t for t in all_tools if call["endpoint"].lower() in t.lower()]
                            if similar:
                                call["endpoint"] = similar[0]
                                valid_plan.append(call)
                
                return valid_plan
        except Exception:
            pass

        # Retry with stricter instruction
        messages.append({
            "role": "system",
            "content": "Your previous response was invalid. Output ONLY valid JSON, no commentary."
        })

    print("Plan output invalid or empty after retries.")
    return []