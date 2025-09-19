import ollama
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import asyncio
import re

# --- Ollama Client Initialization ---
# Using the official, native Ollama async client
client = ollama.AsyncClient(host='http://localhost:11434', timeout=60.0)

# --- Embedding Model Initialization ---
EMBEDDER = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda:0')
print("Embedding model loaded successfully.")

# --- API Definition ---
app = FastAPI()

def clean_model_output(raw_text: str) -> str:
    """
    Uses a regular expression to find and remove the <think>...</think> block,
    including multi-line thoughts.
    """
    # re.DOTALL makes '.' match newlines, so it can span multiple lines.
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    return clean_text.strip()

class Context(BaseModel):
    text: str
    channel_name: str

class ProcessRequest(BaseModel):
    contexts: list[Context]

async def process_single_context(context: Context):
    """Processes a single chat log using the channel name as context."""
    try:
        # --- NEW: The system prompt now includes the VTuber's name ---
        summary_messages = [
            {
                'role': 'system',
                'content': (
                    f"You are an expert AI assistant analyzing a YouTube live stream chat log from the female VTuber, {context.channel_name}. "
                    "Your task is to analyze the following live stream chat log and write a single, engaging, "
                    "narrative sentence in English that summarizes the most likely event that caused this reaction. "
                    "Your response must be a maximum of two sentences and must not contain any bullet points. "
                    "Respond with only the summary sentence."
                )
            },
            {'role': 'user', 'content': context.text}
        ]
        summary_response = await client.chat(model='deepseek-r1:8b-llama-distill-q4_K_M', messages=summary_messages)
        raw_summary = summary_response['message']['content']
        summary = clean_model_output(raw_summary)

        if not summary: return None
        
        # The topic prompt remains the same, using the generated summary
        topic_messages = [
            {'role': 'system', 'content': 'You are an AI tagger. Read a sentence and generate a concise, two or three-word topic label for it in English. Respond with only the topic label.'},
            {'role': 'user', 'content': summary}
        ]
        topic_response = await client.chat(model='deepseek-r1:8b-llama-distill-q4_K_M', messages=topic_messages)
        raw_topic = topic_response['message']['content']
        cleaned_topic = clean_model_output(raw_topic).replace('"', '')

        embedding = EMBEDDER.encode(summary).tolist()

        return {"summary": summary, "topic": cleaned_topic, "embedding": embedding}
    except Exception as e:
        print(f"Error processing context for {context.channel_name}: {e}")
        return None

@app.post("/process_batch")
async def process_batch(request: ProcessRequest):
    """Receives a batch of contexts and processes them concurrently."""
    tasks = [process_single_context(ctx) for ctx in request.contexts]
    results = await asyncio.gather(*tasks)
    successful_results = [res for res in results if res is not None]
    
    if successful_results:
        return {"status": "success", "data": successful_results}
    else:
        return {"status": "error", "message": "All contexts failed to process"}

print("Server is ready. Awaiting requests...")