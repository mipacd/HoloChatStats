# manage_tools.py
"""
Management utilities for the tool vector store.
"""
import asyncio
import argparse
import logging
from tool_store import tool_store
from tools import get_api_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def list_tools():
    """List all tools in the database."""
    await tool_store.initialize()
    try:
        tools = await tool_store.get_all_tool_names()
        logger.info(f"Found {len(tools)} tools:")
        for i, tool in enumerate(tools, 1):
            print(f"{i:3d}. {tool}")
    finally:
        await tool_store.close()

async def search_tools(query: str):
    """Search for tools matching a query."""
    await tool_store.initialize()
    try:
        results = await tool_store.search_relevant_tools(query, top_k=5)
        if results:
            logger.info(f"Found {len(results)} relevant tools for '{query}':")
            for tool in results:
                print(f"\n• {tool.name} (relevance: {tool.relevance_score:.1%})")
                print(f"  {tool.description}")
        else:
            logger.info(f"No tools found for query: {query}")
    finally:
        await tool_store.close()

async def reindex_tools():
    """Reindex all tools (useful after adding new tools)."""
    await tool_store.initialize()
    try:
        tools = get_api_tools()
        await tool_store.index_tools(tools)
        logger.info(f"Successfully reindexed {len(tools)} tools")
    finally:
        await tool_store.close()

async def update_embeddings():
    """Update embeddings for all existing tools."""
    await tool_store.initialize()
    try:
        await tool_store.update_tool_embeddings()
        logger.info("All embeddings updated")
    finally:
        await tool_store.close()

def main():
    parser = argparse.ArgumentParser(description="Manage tool vector store")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    subparsers.add_parser("list", help="List all tools")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for tools")
    search_parser.add_argument("query", help="Search query")
    
    # Reindex command
    subparsers.add_parser("reindex", help="Reindex all tools")
    
    # Update embeddings command
    subparsers.add_parser("update-embeddings", help="Update all embeddings")
    
    args = parser.parse_args()
    
    if args.command == "list":
        asyncio.run(list_tools())
    elif args.command == "search":
        asyncio.run(search_tools(args.query))
    elif args.command == "reindex":
        asyncio.run(reindex_tools())
    elif args.command == "update-embeddings":
        asyncio.run(update_embeddings())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()