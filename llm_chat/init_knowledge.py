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
        "content": "I CAN analyze user overlap between channels, showing what percentage of users are shared between different VTuber communities. I can create overlap matrices for multiple channels.",
        "metadata": {"can_do": True, "related_tools": ["get_common_users", "get_common_users_matrix"]}
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
        "content": "I CAN analyze attrition rates for graduated VTubers, showing how their fanbase continues to engage with other Hololive channels after graduation. I have data for graduations after January 2025 (Fauna, Chloe, Mumei, Gura, Shion, Ao).",
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