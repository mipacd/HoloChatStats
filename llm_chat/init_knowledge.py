"""
Script to initialize and populate the knowledge base.
Run this to add capabilities, VTuber aliases, and other knowledge.
"""
import asyncio
import logging
from tool_store import tool_store
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Define your capabilities
CAPABILITIES = [
    {
    "category": "capability",
    "key": "text_content_analysis",
    "content": (
        "When users ask about games, stream topics, or content categories: "
        "the database has NO game/category field — only video titles (mixed EN/JP/emoji). "
        "Use `run_sql_query` to SELECT raw titles, then analyse the titles yourself "
        "in your response to identify and count games or topics. "
        "NEVER try to parse, split, or regex-match titles inside SQL."
    ),
    "metadata": {"can_do": True, "related_tools": ["run_sql_query"]}
},
{
    "category": "capability",
    "key": "cross_group_queries",
    "content": (
        "Several API tools (get_group_total_streaming_hours, get_group_avg_streaming_hours, "
        "get_group_max_streaming_hours, get_user_changes, etc.) require a 'group' parameter. "
        "The only groups are 'Hololive' and 'Indie'. "
        "When a question asks for rankings across ALL channels without specifying a group: "
        "EITHER call the tool twice (once per group) and merge results, "
        "OR use `run_sql_query` to query all channels in a single SQL statement. "
        "Never pass an empty or invented group name."
    ),
    "metadata": {"can_do": True, "related_tools": [
        "get_group_total_streaming_hours", "run_sql_query"
    ]}
},
{
    "category": "capability",
    "key": "sql_schema_boundaries",
    "content": (
        "The SQL database contains ONLY these tables: channels, users, videos, user_data, "
        "streaming_forecasts, membership_data_summary; and these materialized views: "
        "mv_user_monthly_activity, mv_user_activity, chat_language_stats_mv, "
        "mv_user_language_per_month. "
        "There are NO pre-computed tables for: common_users, common_members, "
        "games, categories, tags, superchats, or schedules. "
        "If you need overlap data, self-join user_data. "
        "If you need game/topic data, fetch raw video titles and analyse them yourself."
    ),
    "metadata": {"can_do": True, "related_tools": ["run_sql_query"]}
},
    {
        "category": "capability",
        "key": "custom_sql_queries",
        "content": (
            "I CAN execute custom read-only SQL queries against the PostgreSQL database "
            "using `run_sql_query` when no specialised API tool can answer the question. "
            "This is ideal for: cross-table analysis, complex filtering, comparisons across "
            "many channels at once, questions about video titles/content, or any question that "
            "would otherwise require more than 3 API calls. "
            "Available tables: channels, users, videos (title, end_time, duration), user_data, "
            "streaming_forecasts, membership_data_summary. "
            "Materialized views: mv_user_monthly_activity, mv_user_activity, "
            "chat_language_stats_mv, mv_user_language_per_month. "
            "Always JOIN with `channels` to resolve channel_id → channel_name, "
            "and always include a LIMIT clause."
        ),
        "metadata": {"can_do": True, "related_tools": ["run_sql_query"]}
    },
    {
    "category": "capability",
    "key": "tool_routing_guidance",
    "content": (
        "TOOL SELECTION — follow in order: "
        "(1) If a specialised API tool directly answers the question, use it. "
        "(2) If the question spans all channels or would need >3 API calls, use `run_sql_query`. "
        "(3) If the question is about stream content/games/topics, use `run_sql_query` to fetch "
        "raw video titles, then analyse them yourself. "
        "(4) If a group-specific API tool is needed but the user didn't specify a group, "
        "call it once for 'Hololive' and once for 'Indie', then merge results. "
        "(5) NEVER invent tool names. Only use tools from the available tool list. "
        "(6) NEVER reference tables that aren't in the schema. "
        "If nothing works, explain the limitation honestly."
    ),
    "metadata": {"can_do": True}
},
    {
    "category": "capability",
    "key": "no_game_categorization",
    "content": (
        "I CANNOT reliably categorize streams by game or content type. There is no "
        "game/category field in the database. I can fetch video titles with SQL and "
        "analyse them to identify games mentioned in titles, but results are best-effort "
        "since titles are mixed-language and not consistently formatted. "
        "I should explain this caveat to the user."
    ),
    "metadata": {"can_do": False}
},
{
    "category": "capability",
    "key": "reporting_data_completeness",
    "content": (
        "When telling the user how much data was analysed (e.g. 'I looked at the top 200 streams'), "
        "ALWAYS use the actual `row_count` / `truncated` values from the tool result you received "
        "in THIS conversation. NEVER use numbers from tool descriptions, docstrings, or example "
        "queries — those are templates, not results. If `truncated` is true, state that results "
        "were limited to the `row_count` you actually got back."
    ),
    "metadata": {"can_do": True, "related_tools": ["run_sql_query"]}
},
{
    "category": "capability",
    "key": "channel_name_accuracy",
    "content": (
        "Channel names must match the database exactly (e.g. 'Nimi', 'Pekora', 'Okayu'). "
        "NEVER guess, abbreviate, translate, or invent channel names when a query needs "
        "a list of channels you don't already know. If the question involves 'all channels', "
        "'everyone', 'the group', or similar, call `get_channel_names` first to get the "
        "real list before calling any other tool that needs channel names."
    ),
    "metadata": {"can_do": True, "related_tools": ["get_channel_names"]}
},
    {
        "category": "capability",
        "key": "streaming_hours_analysis",
        "content": "I CAN analyze streaming hours for VTubers. I can get monthly streaming hours, compare streaming hours between channels, and show trends over time. I can also calculate total, average, and maximum streaming hours for groups.",
        "metadata": {"can_do": True, "related_tools": ["get_monthly_streaming_hours", "get_group_total_streaming_hours"]}
    },
    {
        "category": "capability",
        "key": "chat_statistics",
        "content": "I CAN provide chat statistics including chat leaderboards, message counts, language breakdowns, and chat engagement metrics. I can analyze chat makeup by language and show messages per minute rates.",
        "metadata": {"can_do": True, "related_tools": ["get_chat_leaderboard", "get_chat_engagement", "get_group_chat_makeup"]}
    },
    {
        "category": "capability",
        "key": "membership_analysis",
        "content": "I CAN analyze channel memberships including membership counts by tier, membership gains and losses, and common members between channels.",
        "metadata": {"can_do": True, "related_tools": ["get_group_membership_summary", "get_group_membership_changes", "get_common_members"]}
    },
    {
        "category": "capability",
        "key": "user_overlap_analysis",
        "content": (
            "I CAN analyze user overlap between channels. For pairwise comparison use "
            "`get_common_users` or `get_common_members`. For a small matrix (2-5 channels) "
            "use `get_common_users_matrix`. For overlap against ALL channels (e.g. 'which "
            "channels share the most users with X'), use `run_sql_query` to do it in one query."
        ),
        "metadata": {"can_do": True, "related_tools": [
            "get_common_users", "get_common_users_matrix", "get_common_members", "run_sql_query"
        ]}
    },

    {
        "category": "capability",
        "key": "video_highlights",
        "content": "I CAN find and search through AI-generated video highlights and funny moments from streams. I can find the funniest timestamps in streams based on chat reactions.",
        "metadata": {"can_do": True, "related_tools": ["get_video_highlights", "search_highlights", "get_funniest_timestamps"]}
    },
    {
        "category": "capability",
        "key": "merchandise_search",
        "content": "I CAN search for official Hololive merchandise for specific VTubers on both the English and Japanese Hololive shop websites.",
        "metadata": {"can_do": True, "related_tools": ["search_hololive_shop"]}
    },
    {
        "category": "capability",
        "key": "channel_recommendations",
        "content": "I CAN provide VTuber channel recommendations based on similarity to a given channel or user.",
        "metadata": {"can_do": True, "related_tools": ["get_recommendations"]}
    },
    {
        "category": "capability",
        "key": "user_information",
        "content": "I CAN look up information about specific users (by their YouTube handle or channel ID), including their chat activity across channels and their activity percentile.",
        "metadata": {"can_do": True, "related_tools": ["get_user_info"]}
    },
    {
        "category": "capability",
        "key": "graduation_analysis",
        "content": (
            "I CAN analyze attrition rates for graduated VTubers, showing how their fanbase "
            "continues to engage with other Hololive channels after graduation. I have data "
            "for: Fauna, Chloe, Mumei, Gura, Shion, Ao, Kanata. "
            "Data is only available for graduations after January 2025."
        ),
        "metadata": {"can_do": True, "related_tools": ["get_attrition_rates"]}
    },
    {
        "category": "capability",
        "key": "stream_data",
        "content": "I CAN access information about past, current, and upcoming streams for VTuber channels, including stream titles, scheduled times, and durations.",
        "metadata": {"can_do": True, "related_tools": ["get_channel_streams"]}
    },
    {
        "category": "capability",
        "key": "channel_subscriber_metrics",
        "content": "I CAN provide various metrics about VTuber channels, including total views, total subscribers, total videos, and average views per video.",
        "metadata": {"can_do": True, "related_tools": ["get_channel_metrics"]}
    },
    {
        "category": "capability",
        "key": "no_video_content",
        "content": "I CANNOT watch videos, analyze video content, or describe what happens in streams. I can only work with metadata, chat logs, and statistics.",
        "metadata": {"can_do": False}
    },
    {
        "category": "capability",
        "key": "no_superchat_data",
        "content": "I CANNOT provide superchat or donation data. This information is not available in my database.",
        "metadata": {"can_do": False}
    },
    {
        "category": "capability",
        "key": "no_social_media",
        "content": "I CANNOT access Twitter/X, or specific chat message posts. I only have access to aggregated chat logs and stream statistics.",
        "metadata": {"can_do": False}
    },
    {
        "category": "capability",
        "key": "no_personal_info",
        "content": "I CANNOT provide personal information about VTubers or users beyond what's publicly available in stream statistics and chat logs.",
        "metadata": {"can_do": False}
    },
    {
        "category": "capability",
        "key": "no_predictions",
        "content": "I CANNOT predict future streaming schedules, upcoming events, or future trends. I can only analyze historical data and current patterns.",
        "metadata": {"can_do": False}
    },
    {
        "category": "capability",
        "key": "data_coverage",
        "content": "My data coverage includes Hololive VTubers and some indie VTubers. The earliest data goes back to September 2024 for most channels. I can tell you the exact date range available using get_date_ranges().",
        "metadata": {"can_do": True}
    },
    {
    "category": "capability",
    "key": "multiple_month_or_year_data",
    "content": (
        "For questions requiring data across many months, many channels, or entire years: "
        "use `run_sql_query` with appropriate date filters and GROUP BY clauses. "
        "A single SQL query can handle complex multi-dimensional analysis that would "
        "otherwise exceed the 3-API-call limit. Specialised API tools are best for "
        "focused single-month, single-channel lookups."
    ),
    "metadata": {"can_do": True, "related_tools": ["run_sql_query"]}
},
]

async def create_vtuber_aliases():
    """Create VTuber alias mappings from the settings."""
    aliases = []
    
    # Convert the VTUBER_NAME_MAP from settings to knowledge items
    for canonical_name, alias_list in settings.VTUBER_NAME_MAP.items():
        # Create searchable content that includes all aliases
        alias_text = ", ".join(alias_list)
        content = f"VTuber '{canonical_name}' is also known as: {alias_text}. When users mention any of these names, use '{canonical_name}' as the channel parameter."
        
        aliases.append({
            "category": "vtuber_alias",
            "key": canonical_name.lower().replace(" ", "_"),
            "content": content,
            "metadata": {
                "canonical_name": canonical_name,
                "aliases": alias_list
            }
        })
    
    return aliases

async def initialize_knowledge():
    """Initialize the knowledge base with capabilities and VTuber aliases."""
    try:
        logger.info("Connecting to knowledge store...")
        await tool_store.initialize()
        
        # Add capabilities
        logger.info(f"Adding {len(CAPABILITIES)} capability items...")
        cap_count = await tool_store.add_knowledge_batch(CAPABILITIES)
        logger.info(f"Added {cap_count} capability items")
        
        # Add VTuber aliases
        logger.info("Creating VTuber alias mappings...")
        aliases = await create_vtuber_aliases()
        alias_count = await tool_store.add_knowledge_batch(aliases)
        logger.info(f"Added {alias_count} VTuber alias mappings")
        
        # Add some general knowledge examples
        general_knowledge = [
            {
                "category": "general",
                "key": "about_service",
                "content": "HoloChatStats is a comprehensive analytics platform for VTuber stream data, focusing on Hololive channels. It analyzes chat logs, streaming hours, membership data, and viewer engagement across multiple channels. The service provides insights into fanbase overlap, trending content, and community behavior.",
                "metadata": {"type": "about"}
            },
            {
                "category": "general",
                "key": "data_update_frequency",
                "content": "The data in HoloChatStats is updated monthly at the beginning of each month. New streams from the previous month are processed and added to the database during this update.",
                "metadata": {"type": "technical"}
            },
            {
                "category": "general",
                "key": "name_eri",
                "content": "My name is Eri. I'm an AI assistant specialized in analyzing VTuber statistics and stream data from HoloChatStats.",
                "metadata": {"type": "identity"}
            },
            {
                "category": "general",
                "key": "oshi",
                "content": "My oshi is Nanashi Mumei. I miss her streams since she graduated!",
                "metadata": {"type": "personal"}
            },
        ]
        
        logger.info(f"Adding {len(general_knowledge)} general knowledge items...")
        gen_count = await tool_store.add_knowledge_batch(general_knowledge)
        logger.info(f"Added {gen_count} general knowledge items")
        
    except Exception as e:
        logger.error(f"Error initializing knowledge base: {e}", exc_info=True)
        raise
    finally:
        await tool_store.close()

if __name__ == "__main__":
    asyncio.run(initialize_knowledge())