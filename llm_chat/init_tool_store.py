"""
Script to initialize and populate the tool vector store.
Run this once when setting up or updating tools.
"""
import asyncio
import logging
from tools import get_api_tools
from tool_store import tool_store
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

async def initialize_tool_store():
    """Initialize the tool store and index all available tools."""
    try:
        # Log connection info (without password)
        logger.info(f"Connecting to PostgreSQL at {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
        
        # Initialize the database
        await tool_store.initialize()
        logger.info("Tool store initialized successfully")
        
        # Get all tools
        tools = get_api_tools()
        logger.info(f"Found {len(tools)} tools to index")
        
        # Index the tools
        await tool_store.index_tools(tools)
        logger.info("All tools indexed successfully")
        
    except Exception as e:
        logger.error(f"Error initializing tool store: {e}", exc_info=True)
        raise
    finally:
        await tool_store.close()

if __name__ == "__main__":
    asyncio.run(initialize_tool_store())