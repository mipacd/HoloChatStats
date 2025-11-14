# tool_store.py
import asyncio
import inspect
import json
import logging
from typing import List, Dict, Optional, Any, get_type_hints
from dataclasses import dataclass

import asyncpg
from sentence_transformers import SentenceTransformer
from langchain_core.tools import BaseTool

from config import settings

logger = logging.getLogger(__name__)

@dataclass
class ToolDefinition:
    """Represents a tool definition with its metadata and embedding."""
    name: str
    description: str
    parameters: Dict[str, Any]
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None

@dataclass
class KnowledgeItem:
    """Represents a knowledge base item with metadata and embedding."""
    id: Optional[int] = None
    category: str = ""  # e.g., "capability", "vtuber_alias", "general"
    key: str = ""  # Unique identifier within category
    content: str = ""  # The actual knowledge content
    metadata: Optional[Dict[str, Any]] = None  # Additional structured data
    relevance_score: Optional[float] = None

class ToolVectorStore:
    """Manages tool definitions and knowledge base in a PostgreSQL vector database."""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        # Use local MiniLM L6 v2 model - small, fast, and effective
        # This model produces 384-dimensional embeddings
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384  # MiniLM L6 v2 dimension
        
    async def initialize(self):
        """Initialize database connection pool and create necessary tables."""
        # Build connection string from Pydantic settings
        self.pool = await asyncpg.create_pool(
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
            database=settings.POSTGRES_DB,
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            min_size=1,
            max_size=10
        )
        
        # Create the tools table with vector extension
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Initialize tool definitions table
            await self._initialize_tool_table(conn)
            
            # Initialize knowledge base table
            await self._initialize_knowledge_table(conn)
            
            logger.info(f"Tool store initialized with {self.embedding_dimension}-dimensional embeddings")
    
    async def _initialize_tool_table(self, conn):
        """Initialize the tool definitions table."""
        # Check if table exists and get current vector dimension
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'tool_definitions'
            )
        """)
        
        recreate_table = False
        
        if table_exists:
            # Get the column type definition for embedding column
            try:
                vector_type_info = await conn.fetchval("""
                    SELECT format_type(a.atttypid, a.atttypmod) as type
                    FROM pg_attribute a
                    WHERE a.attrelid = 'tool_definitions'::regclass
                    AND a.attname = 'embedding'
                """)
                
                if vector_type_info:
                    import re
                    match = re.search(r'vector\((\d+)\)', vector_type_info)
                    if match:
                        current_dim = int(match.group(1))
                        if current_dim != self.embedding_dimension:
                            logger.info(
                                f"Tool embedding dimension changed from {current_dim} to {self.embedding_dimension}, "
                                f"recreating table..."
                            )
                            recreate_table = True
                    else:
                        logger.info("Could not determine current embedding dimension, recreating table...")
                        recreate_table = True
            except Exception as e:
                logger.warning(f"Error checking existing table structure: {e}")
                recreate_table = True
        
        if recreate_table:
            await conn.execute("DROP TABLE IF EXISTS tool_definitions CASCADE")
        
        # Create tools table with correct vector dimension
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS tool_definitions (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT NOT NULL,
                parameters JSONB NOT NULL,
                full_definition TEXT NOT NULL,
                embedding vector({self.embedding_dimension}),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Check if index exists before creating
        index_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM pg_indexes
                WHERE indexname = 'tool_embedding_idx'
            )
        """)
        
        if not index_exists:
            row_count = await conn.fetchval("SELECT COUNT(*) FROM tool_definitions")
            if row_count > 0:
                await conn.execute("""
                    CREATE INDEX tool_embedding_idx 
                    ON tool_definitions 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                logger.info("Created vector index for tool similarity search")
    
    async def _initialize_knowledge_table(self, conn):
        """Initialize the knowledge base table."""
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'knowledge_base'
            )
        """)
        
        recreate_table = False
        
        if table_exists:
            try:
                vector_type_info = await conn.fetchval("""
                    SELECT format_type(a.atttypid, a.atttypmod) as type
                    FROM pg_attribute a
                    WHERE a.attrelid = 'knowledge_base'::regclass
                    AND a.attname = 'embedding'
                """)
                
                if vector_type_info:
                    import re
                    match = re.search(r'vector\((\d+)\)', vector_type_info)
                    if match:
                        current_dim = int(match.group(1))
                        if current_dim != self.embedding_dimension:
                            logger.info(
                                f"Knowledge embedding dimension changed from {current_dim} to {self.embedding_dimension}, "
                                f"recreating table..."
                            )
                            recreate_table = True
            except Exception as e:
                logger.warning(f"Error checking existing knowledge table structure: {e}")
                recreate_table = True
        
        if recreate_table:
            await conn.execute("DROP TABLE IF EXISTS knowledge_base CASCADE")
        
        # Create knowledge base table
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id SERIAL PRIMARY KEY,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSONB,
                embedding vector({self.embedding_dimension}),
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(category, key)
            )
        """)
        
        # Create indexes
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS knowledge_category_idx 
            ON knowledge_base(category)
        """)
        
        index_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM pg_indexes
                WHERE indexname = 'knowledge_embedding_idx'
            )
        """)
        
        if not index_exists:
            row_count = await conn.fetchval("SELECT COUNT(*) FROM knowledge_base")
            if row_count > 0:
                await conn.execute("""
                    CREATE INDEX knowledge_embedding_idx 
                    ON knowledge_base 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                logger.info("Created vector index for knowledge similarity search")
    
    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
    
    def _format_vector_for_postgres(self, embedding: List[float]) -> str:
        """Format a vector embedding as a string for PostgreSQL."""
        return '[' + ','.join(str(x) for x in embedding) + ']'
    
    def _extract_tool_info(self, tool: BaseTool) -> Dict[str, Any]:
        """Extract tool information from a LangChain tool."""
        params = {}
        
        # Try to get the underlying function from various possible attributes
        func = None
        if hasattr(tool, 'func') and callable(tool.func):
            func = tool.func
        elif hasattr(tool, 'coroutine') and callable(tool.coroutine):
            func = tool.coroutine
        elif hasattr(tool, '_run'):
            func = tool._run
        elif hasattr(tool, '_arun'):
            func = tool._arun
        
        # Extract parameters from the function signature if available
        if func and callable(func):
            try:
                sig = inspect.signature(func)
                
                try:
                    type_hints = get_type_hints(func)
                except:
                    type_hints = {}
                
                for param_name, param in sig.parameters.items():
                    if param_name not in ['self', 'cls', 'run_manager', 'callbacks']:
                        param_type = "str"
                        if param_name in type_hints:
                            param_type = str(type_hints[param_name])
                        elif param.annotation != inspect.Parameter.empty:
                            param_type = str(param.annotation)
                        
                        param_type = param_type.replace("<class '", "").replace("'>", "").replace("typing.", "")
                        
                        param_info = {
                            "type": param_type,
                            "required": param.default == inspect.Parameter.empty
                        }
                        if param.default != inspect.Parameter.empty:
                            param_info["default"] = str(param.default)
                        params[param_name] = param_info
            except Exception as e:
                logger.debug(f"Could not extract signature from tool {tool.name}: {e}")
        
        if not params and hasattr(tool, 'args') and tool.args:
            for arg_name, arg_schema in tool.args.items():
                param_info = {
                    "type": arg_schema.get('type', 'str'),
                    "required": arg_name in tool.args.get('required', [])
                }
                if 'default' in arg_schema:
                    param_info['default'] = str(arg_schema['default'])
                params[arg_name] = param_info
        
        full_definition = f"""
Tool: {tool.name}
Function: {tool.name.replace('_', ' ')}
Description: {tool.description}
Parameters: {json.dumps(params, indent=2)}

Use this tool when: {tool.description.lower()}
Keywords: {' '.join(tool.name.split('_'))} {tool.description.lower()}
"""
        
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": params,
            "full_definition": full_definition
        }
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using the local SentenceTransformer model."""
        embeddings = self.embeddings_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding using the local SentenceTransformer model."""
        embedding = self.embeddings_model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embedding.tolist()
    
    # ==================== TOOL METHODS ====================
    
    async def index_tools(self, tools: List[BaseTool]):
        """Index all tools in the vector database."""
        logger.info(f"Indexing {len(tools)} tools...")
        
        tool_infos = [self._extract_tool_info(tool) for tool in tools]
        
        texts = [info["full_definition"] for info in tool_infos]
        logger.info("Generating embeddings with local MiniLM model...")
        embeddings = await asyncio.to_thread(self._generate_embeddings, texts)
        
        async with self.pool.acquire() as conn:
            for tool_info, embedding in zip(tool_infos, embeddings):
                embedding_str = self._format_vector_for_postgres(embedding)
                
                await conn.execute("""
                    INSERT INTO tool_definitions 
                    (name, description, parameters, full_definition, embedding)
                    VALUES ($1, $2, $3, $4, $5::vector)
                    ON CONFLICT (name) 
                    DO UPDATE SET
                        description = EXCLUDED.description,
                        parameters = EXCLUDED.parameters,
                        full_definition = EXCLUDED.full_definition,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                """, 
                    tool_info["name"],
                    tool_info["description"],
                    json.dumps(tool_info["parameters"]),
                    tool_info["full_definition"],
                    embedding_str
                )
            
            index_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE indexname = 'tool_embedding_idx'
                )
            """)
            
            if not index_exists:
                logger.info("Creating vector index for tool similarity search...")
                await conn.execute("""
                    CREATE INDEX tool_embedding_idx 
                    ON tool_definitions 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
                logger.info("Vector index created successfully")
        
        logger.info(f"Successfully indexed {len(tools)} tools with {self.embedding_dimension}-dimensional embeddings")
    
    async def search_relevant_tools(
        self, 
        query: str, 
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[ToolDefinition]:
        """Search for the most relevant tools based on the query."""
        query_embedding = await asyncio.to_thread(self._generate_embedding, query)
        query_embedding_str = self._format_vector_for_postgres(query_embedding)
        
        async with self.pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT 
                    name,
                    description,
                    parameters,
                    1 - (embedding <=> $1::vector) as similarity
                FROM tool_definitions
                WHERE 1 - (embedding <=> $1::vector) > $2
                ORDER BY similarity DESC
                LIMIT $3
            """, query_embedding_str, similarity_threshold, top_k)
        
        tools = []
        for row in results:
            tools.append(ToolDefinition(
                name=row["name"],
                description=row["description"],
                parameters=json.loads(row["parameters"]),
                relevance_score=float(row["similarity"])
            ))
        
        logger.debug(f"Found {len(tools)} relevant tools for query: {query[:50]}...")
        return tools
    
    async def get_tool_by_name(self, name: str) -> Optional[ToolDefinition]:
        """Get a specific tool by its name."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT name, description, parameters
                FROM tool_definitions
                WHERE name = $1
            """, name)
            
            if row:
                return ToolDefinition(
                    name=row["name"],
                    description=row["description"],
                    parameters=json.loads(row["parameters"])
                )
        return None
    
    async def get_all_tool_names(self) -> List[str]:
        """Get a list of all tool names in the database."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("SELECT name FROM tool_definitions ORDER BY name")
            return [row["name"] for row in rows]
    
    # ==================== KNOWLEDGE BASE METHODS ====================
    
    async def add_knowledge(
        self,
        category: str,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add or update a knowledge item.
        
        Args:
            category: Category of knowledge (e.g., "capability", "vtuber_alias", "general")
            key: Unique key within the category
            content: The actual knowledge content (will be embedded)
            metadata: Optional additional structured data
            
        Returns:
            The ID of the created/updated knowledge item
        """
        # Generate embedding for the content
        embedding = await asyncio.to_thread(self._generate_embedding, content)
        embedding_str = self._format_vector_for_postgres(embedding)
        
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO knowledge_base 
                (category, key, content, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5::vector)
                ON CONFLICT (category, key) 
                DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding,
                    updated_at = NOW()
                RETURNING id
            """, 
                category,
                key,
                content,
                json.dumps(metadata) if metadata else None,
                embedding_str
            )
            
            # Create index if it doesn't exist and we have data
            index_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE indexname = 'knowledge_embedding_idx'
                )
            """)
            
            if not index_exists:
                row_count = await conn.fetchval("SELECT COUNT(*) FROM knowledge_base")
                if row_count > 0:
                    await conn.execute("""
                        CREATE INDEX knowledge_embedding_idx 
                        ON knowledge_base 
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 100)
                    """)
            
            return result["id"]
    
    async def add_knowledge_batch(self, items: List[Dict[str, Any]]) -> int:
        """
        Add multiple knowledge items at once.
        
        Args:
            items: List of dicts with keys: category, key, content, metadata (optional)
            
        Returns:
            Number of items added/updated
        """
        if not items:
            return 0
        
        # Generate all embeddings at once
        contents = [item["content"] for item in items]
        embeddings = await asyncio.to_thread(self._generate_embeddings, contents)
        
        async with self.pool.acquire() as conn:
            for item, embedding in zip(items, embeddings):
                embedding_str = self._format_vector_for_postgres(embedding)
                await conn.execute("""
                    INSERT INTO knowledge_base 
                    (category, key, content, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5::vector)
                    ON CONFLICT (category, key) 
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                """,
                    item["category"],
                    item["key"],
                    item["content"],
                    json.dumps(item.get("metadata")) if item.get("metadata") else None,
                    embedding_str
                )
            
            # Create index if needed
            index_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE indexname = 'knowledge_embedding_idx'
                )
            """)
            
            if not index_exists:
                logger.info("Creating vector index for knowledge similarity search...")
                await conn.execute("""
                    CREATE INDEX knowledge_embedding_idx 
                    ON knowledge_base 
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100)
                """)
        
        return len(items)
    
    async def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.4
    ) -> List[KnowledgeItem]:
        """
        Search for relevant knowledge items.
        
        Args:
            query: The search query
            category: Optional category filter
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of relevant knowledge items with similarity scores
        """
        query_embedding = await asyncio.to_thread(self._generate_embedding, query)
        query_embedding_str = self._format_vector_for_postgres(query_embedding)
        
        async with self.pool.acquire() as conn:
            if category:
                results = await conn.fetch("""
                    SELECT 
                        id,
                        category,
                        key,
                        content,
                        metadata,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM knowledge_base
                    WHERE category = $2
                    AND 1 - (embedding <=> $1::vector) > $3
                    ORDER BY similarity DESC
                    LIMIT $4
                """, query_embedding_str, category, similarity_threshold, top_k)
            else:
                # Different query without category parameter
                results = await conn.fetch("""
                    SELECT 
                        id,
                        category,
                        key,
                        content,
                        metadata,
                        1 - (embedding <=> $1::vector) as similarity
                    FROM knowledge_base
                    WHERE 1 - (embedding <=> $1::vector) > $2
                    ORDER BY similarity DESC
                    LIMIT $3
                """, query_embedding_str, similarity_threshold, top_k)
        
        items = []
        for row in results:
            items.append(KnowledgeItem(
                id=row["id"],
                category=row["category"],
                key=row["key"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row["metadata"] else None,
                relevance_score=float(row["similarity"])
            ))
        
        logger.debug(f"Found {len(items)} relevant knowledge items for query: {query[:50]}...")
        return items
    
    async def get_knowledge_by_key(self, category: str, key: str) -> Optional[KnowledgeItem]:
        """Get a specific knowledge item by category and key."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, category, key, content, metadata
                FROM knowledge_base
                WHERE category = $1 AND key = $2
            """, category, key)
            
            if row:
                return KnowledgeItem(
                    id=row["id"],
                    category=row["category"],
                    key=row["key"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None
                )
        return None
    
    async def get_all_knowledge_by_category(self, category: str) -> List[KnowledgeItem]:
        """Get all knowledge items in a category."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, category, key, content, metadata
                FROM knowledge_base
                WHERE category = $1
                ORDER BY key
            """, category)
            
            return [
                KnowledgeItem(
                    id=row["id"],
                    category=row["category"],
                    key=row["key"],
                    content=row["content"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else None
                )
                for row in rows
            ]
    
    async def delete_knowledge(self, category: str, key: str) -> bool:
        """Delete a knowledge item."""
        async with self.pool.acquire() as conn:
            result = await conn.execute("""
                DELETE FROM knowledge_base
                WHERE category = $1 AND key = $2
            """, category, key)
            return result != "DELETE 0"
    
    async def get_knowledge_categories(self) -> List[str]:
        """Get all unique knowledge categories."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT DISTINCT category 
                FROM knowledge_base 
                ORDER BY category
            """)
            return [row["category"] for row in rows]

# Singleton instance
tool_store = ToolVectorStore()