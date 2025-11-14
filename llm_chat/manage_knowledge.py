# manage_knowledge.py
"""
Management utilities for the knowledge base.
"""
import asyncio
import argparse
import json
import logging
from tool_store import tool_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def list_knowledge(category: str = None):
    """List all knowledge items, optionally filtered by category."""
    await tool_store.initialize()
    try:
        if category:
            items = await tool_store.get_all_knowledge_by_category(category)
            logger.info(f"Found {len(items)} items in category '{category}':")
        else:
            categories = await tool_store.get_knowledge_categories()
            logger.info(f"All knowledge categories: {', '.join(categories)}")
            items = []
            for cat in categories:
                cat_items = await tool_store.get_all_knowledge_by_category(cat)
                items.extend(cat_items)
            logger.info(f"Total: {len(items)} items")
        
        for item in items:
            print(f"\n[{item.category}] {item.key}")
            print(f"  {item.content[:100]}...")
            if item.metadata:
                print(f"  Metadata: {json.dumps(item.metadata, indent=2)}")
    finally:
        await tool_store.close()

async def search_knowledge(query: str, category: str = None):
    """Search for knowledge matching a query."""
    await tool_store.initialize()
    try:
        results = await tool_store.search_knowledge(query, category=category, top_k=5)
        if results:
            logger.info(f"Found {len(results)} relevant items for '{query}':")
            for item in results:
                print(f"\n• [{item.category}] {item.key} (relevance: {item.relevance_score:.1%})")
                print(f"  {item.content}")
                if item.metadata:
                    print(f"  Metadata: {json.dumps(item.metadata)}")
        else:
            logger.info(f"No knowledge found for query: {query}")
    finally:
        await tool_store.close()

async def add_knowledge_item(category: str, key: str, content: str, metadata: str = None):
    """Add a single knowledge item."""
    await tool_store.initialize()
    try:
        meta_dict = json.loads(metadata) if metadata else None
        item_id = await tool_store.add_knowledge(category, key, content, meta_dict)
        logger.info(f"Added knowledge item with ID {item_id}")
    finally:
        await tool_store.close()

async def delete_knowledge_item(category: str, key: str):
    """Delete a knowledge item."""
    await tool_store.initialize()
    try:
        success = await tool_store.delete_knowledge(category, key)
        if success:
            logger.info(f"Deleted knowledge item: [{category}] {key}")
        else:
            logger.warning(f"Knowledge item not found: [{category}] {key}")
    finally:
        await tool_store.close()

def main():
    parser = argparse.ArgumentParser(description="Manage knowledge base")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List knowledge items")
    list_parser.add_argument("--category", help="Filter by category")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search knowledge")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--category", help="Filter by category")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add knowledge item")
    add_parser.add_argument("category", help="Category")
    add_parser.add_argument("key", help="Unique key")
    add_parser.add_argument("content", help="Content")
    add_parser.add_argument("--metadata", help="JSON metadata")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete knowledge item")
    delete_parser.add_argument("category", help="Category")
    delete_parser.add_argument("key", help="Key")
    
    args = parser.parse_args()
    
    if args.command == "list":
        asyncio.run(list_knowledge(args.category))
    elif args.command == "search":
        asyncio.run(search_knowledge(args.query, args.category))
    elif args.command == "add":
        asyncio.run(add_knowledge_item(args.category, args.key, args.content, args.metadata))
    elif args.command == "delete":
        asyncio.run(delete_knowledge_item(args.category, args.key))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()