import httpx
from langchain_core.tools import tool
from config import settings
from typing import Optional, Literal
import logging

# Create a single, shared async HTTP client for calling your 'web' API
# The base_url is configured from the WEB_API_URL environment variable
api_client = httpx.AsyncClient(
    base_url=settings.WEB_API_URL,
    headers={"User-Agent": settings.WEB_API_USER_AGENT},
    timeout=30.0
)

async def close_api_client():
    await api_client.aclose()



logger = logging.getLogger(__name__)

async def call_hcs_api(endpoint: str, params: dict) -> dict:
    try:
        resp = await api_client.get(endpoint, params=params, timeout=30.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        error_type = "API_ERROR"
        details = f"HTTP error {e.response.status_code} - {e.response.text}"
    except httpx.TimeoutException:
        error_type = "TIMEOUT"
        details = "Request timed out after 30 seconds"
    except httpx.NetworkError:
        error_type = "NETWORK_ERROR"
        details = "Network connection failed"
    except Exception as e:
        error_type = "UNKNOWN_ERROR"
        details = str(e)
    
    logger.error(f"API call failed: {error_type} - {endpoint} - {details}")
    return {
        "error": True,
        "type": error_type,
        "message": f"API request failed ({error_type})",
        "details": details
    }


@tool
async def get_recommendations(identifier: str) -> dict:
    """
    Returns VTuber channel recommendations for the given YouTube user, including a similarity score from 0–100.

    Args:
        identifier (str): The YouTube handle (starting with "@") or channel ID (starting with "UC") 
            to get recommendations for.

    Returns:
        dict: {
            "recommended_channels": [
                {
                    "channel_name": str,   # The recommended channel's name
                    "score": float         # Similarity score between 0–100
                }
            ]
        }
    """
    return await call_hcs_api("/api/recommend", {"identifier": identifier})

@tool
async def get_monthly_streaming_hours(channel: str) -> dict:
    """
    Returns the total streaming hours per month for a given VTuber channel.

    Args:
        channel (str): The name of the VTuber whose monthly
            streaming hours should be retrieved.

    Returns:
        dict: {
            "success": True,
            "data": [
                {
                    "month": str,                  # Month in 'YYYY-MM' format
                    "total_streaming_hours": float # Total streaming hours for that month
                }
            ]
        }
    """
    return await call_hcs_api("/api/get_monthly_streaming_hours", {"channel": channel})


@tool
async def get_group_total_streaming_hours(group: str, month: str) -> dict:
    """
    Returns the total streaming hours for all channels in a given VTuber group during a specific month.

    Args:
        group (str): The VTuber group name (e.g., "Hololive", "Indie").
        month (str): The target month in 'YYYY-MM' format (e.g., "2025-09").

    Returns:
        dict: {
            "success": True,
            "data": [
                {
                    "channel": str,     # Channel name
                    "month": str,       # Month in 'YYYY-MM' format
                    "hours": float      # Total streaming hours for that channel during the month
                    "change": float     # Change in hours from the previous month
                }
            ]
        }
    """
    return await call_hcs_api("/api/get_group_total_streaming_hours", {"group": group, "month": month})


@tool
async def get_group_avg_streaming_hours(group: str, month: str) -> dict:
    """
    Returns the average streaming hours across all channels in the given VTuber group for a specific month.

    Args:
        group (str): The VTuber group name (e.g., "Hololive", "Indie").
        month (str): The target month in 'YYYY-MM' format.

    Returns:
        dict: {
            "success": True,
            "data": [
                {
                    "channel": str,     # Channel name
                    "month": str,       # Month in 'YYYY-MM' format
                    "hours": float      # Average streaming hours for that channel
                }
            ]
        }
    """
    return await call_hcs_api("/api/get_group_avg_streaming_hours", {"group": group, "month": month})


@tool
async def get_group_max_streaming_hours(group: str, month: str) -> dict:
    """
    Returns the longest single-stream duration (maximum streaming hours) for each channel in the given group
    during a specific month.

    Args:
        group (str): The VTuber group name (e.g., "Hololive", "Indie").
        month (str): The target month in 'YYYY-MM' format.

    Returns:
        dict: {
            "success": True,
            "data": [
                {
                    "channel": str,     # Channel name
                    "month": str,       # Month in 'YYYY-MM' format
                    "hours": float      # Duration of the longest stream (hours)
                }
            ]
        }
    """
    return await call_hcs_api("/api/get_group_max_streaming_hours", {"group": group, "month": month})


@tool
async def get_group_chat_makeup(group: str, month: str) -> dict:
    """
    Returns chat makeup statistics for all channels in a given VTuber group for a specific month,
    showing average messages per minute by language and emoji usage.

    Args:
        group (str): The VTuber group name (e.g., "Hololive", "Indie").
        month (str): The target month in 'YYYY-MM' format.

    Returns:
        dict: {
            "success": True,
            "data": [
                {
                    "channel_name": str,               # Channel name
                    "observed_month": str,             # Month in 'YYYY-MM' format
                    "es_en_id_rate_per_minute": float, # Combined ES/EN/ID message rate per minute
                    "jp_rate_per_minute": float,       # Japanese message rate per minute
                    "kr_rate_per_minute": float,       # Korean message rate per minute
                    "ru_rate_per_minute": float,       # Russian message rate per minute
                    "emoji_rate_per_minute": float     # Emoji message rate per minute
                }
            ]
        }
    Notes:
        - Metrics are normalized by total streaming time in minutes.
        - Results are ordered by total streaming time descending.
    """
    return await call_hcs_api("/api/get_group_chat_makeup", {"group": group, "month": month})


@tool
async def get_common_users(channel_a: str, month_a: str, channel_b: str, month_b: str) -> dict:
    """
    Returns overlap statistics for chat users between two VTuber channels across specified months.

    Args:
        channel_a (str): Name of the first VTuber channel.
        month_a (str): Month for the first channel, in 'YYYY-MM' format.
        channel_b (str): Name of the second VTuber channel.
        month_b (str): Month for the second channel, in 'YYYY-MM' format.

    Returns:
        dict: {
            "channel_a": str,                 # First channel name
            "channel_b": str,                 # Second channel name
            "month_a": str,                   # Month for channel A
            "month_b": str,                   # Month for channel B
            "num_common_users": int,          # Number of users active in both
            "percent_A_to_B_users": float,    # % of A's users also active in B
            "percent_B_to_A_users": float     # % of B's users also active in A
        }
    Notes:
        - Only users with recorded activity for each month are counted.
    """
    return await call_hcs_api("/api/get_common_users", {
        "channel_a": channel_a,
        "month_a": month_a,
        "channel_b": channel_b,
        "month_b": month_b
    })


@tool
async def get_common_users_matrix(channels: list[str], month: str, members_only: bool = False) -> dict:
    """
    Returns a matrix of overlap percentages between multiple VTuber channels for a specific month.
    Each cell [i][j] represents the percentage of channel i's users who were also active in channel j.

    Args:
        channels (list[str]): A list of two or more VTuber channel names.
        month (str): Month in 'YYYY-MM' format (e.g., '2025-09').
        members_only (bool, optional): If True, limits the matrix to members only.
            Defaults to False.

    Returns:
        dict: {
            "labels": [str],   # Ordered list of channel names
            "matrix": [        # NxN matrix of percentages
                [float, ...],
                ...
            ]
        }
    Notes:
        - Each diagonal cell (i,i) will always be 100% if that channel has users.
        - Percentages are based on the user set size of the *row* channel.
    """
    return await call_hcs_api("/api/get_common_users_matrix", {
        "channels": ",".join(channels),
        "month": month,
        "members_only": str(members_only).lower()
    })


@tool
async def get_common_members(channel_a: str, month_a: str, channel_b: str, month_b: str) -> dict:
    """
    Returns overlap statistics for members between two VTuber channels across specific months.

    Args:
        channel_a (str): Name of the first VTuber channel.
        month_a (str): Month for the first channel, in 'YYYY-MM' format.
        channel_b (str): Name of the second VTuber channel.
        month_b (str): Month for the second channel, in 'YYYY-MM' format.

    Returns:
        dict: {
            "channel_a": str,                    # First channel name
            "channel_b": str,                    # Second channel name
            "month_a": str,                      # Month for channel A
            "month_b": str,                      # Month for channel B
            "num_common_members": int,           # Number of members shared between both
            "percent_A_to_B_members": float,     # % of A's members who are also in B
            "percent_B_to_A_members": float      # % of B's members who are also in A
        }
    Notes:
        - Membership is evaluated based on last message activity per month.
    """
    return await call_hcs_api("/api/get_common_members", {
        "channel_a": channel_a,
        "month_a": month_a,
        "channel_b": channel_b,
        "month_b": month_b
    })


@tool
async def get_group_membership_summary(
    group: str, 
    month: str, 
    membership_rank: str | None = None
) -> list | dict:
    """
    Returns membership summary data for each channel in a given VTuber group for a specific month.

    Args:
        group (str): The VTuber group name (e.g., "Hololive", "Indie").
        month (str): The target month in 'YYYY-MM' format.
        membership_rank (str): Specific membership tier to fetch.
            -1 for non-members, 0 for new members, 1 for 1-month members, 2 for 2-month members, 3 for 6 months, 4 for 1 year, etc. or "total" for the sum of all members.

    Returns:
        list[list] | dict:
            - If membership_rank has a numerical value: List of entries per channel:
                [
                    channel_name (str),
                    membership_rank (int),
                    membership_count (float),
                    percentage_total (float)
                ]
            - If membership_rank = "total":
                {
                    channel_name: str,
                    total_members: int
                },

    Raises:
        ValueError: If neither membership_rank is not provided
    """
    payload = {
        "channel_group": group,
        "month": month,
        "membership_rank": membership_rank
    }

    return await call_hcs_api("/api/get_group_membership_summary", payload)


@tool
async def get_group_membership_changes(group: str, month: str) -> dict:
    """
    Returns membership gains, losses, and net differential for all channels in a VTuber group for a specific month.

    Args:
        group (str): The VTuber group name (e.g., "Hololive", "Indie").
        month (str): Month in 'YYYY-MM' format (e.g., '2025-09').

    Returns:
        dict: {
            "success": True,
            "data": [
                {
                    "channel_name": str,   # Channel name
                    "observed_month": str, # Month of observation (YYYY-MM)
                    "gains_count": int,    # Number of new members gained
                    "losses_count": int,   # Number of members lost
                    "differential": int    # Gains minus losses
                }
            ]
        }
    Notes:
        - Gains are defined as users whose membership_rank increased from -1.
        - Losses are users whose membership_rank dropped to -1.
        - Results are ordered by differential descending.
    """
    return await call_hcs_api("/api/get_group_membership_changes", {
        "channel_group": group,
        "month": month
    })

@tool
async def get_group_streaming_hours_diff(group: Optional[str], month: str) -> dict:
    """
    Returns the total streaming hours and change since the previous month for all channels in a given VTuber group.

    Args:
        group (str, optional): The VTuber group name (e.g., "Hololive", "Indie").
            If omitted, includes all channels across all groups.
        month (str): The target month in 'YYYY-MM' format (e.g., "2025-09").

    Returns:
        dict: {
            "success": bool,                # Whether the request succeeded
            "data": [
                {
                    "channel": str,         # Channel name
                    "month": str,           # Observed month (YYYY-MM)
                    "hours": float,         # Total streaming hours in the given month
                    "change": float         # Change in hours from the previous month
                }
            ]
        }

    Notes:
        - The `change` field may be negative if the channel streamed fewer hours.
        - If a channel did not stream in the previous month, `change` equals `hours`.
        - Results are ordered by `change` descending.
    """
    return await call_hcs_api("/api/get_group_streaming_hours_diff", {
        "group": group,
        "month": month
    })


@tool
async def get_chat_leaderboard(channel_name: str, month: str) -> list[dict]:
    """
    Returns the top 10 most active chatters in a channel for a given month.

    Args:
        channel_name (str): The VTuber channel name (e.g., "Miko").
        month (str): The month to query in 'YYYY-MM' format (e.g., "2025-09").

    Returns:
        list[dict]: [
            {
                "user_name": str,       # Display name of the chatter, either a YouTube handle (starting with @) or a channel ID (starting with "UC")
                "message_count": int    # Number of messages sent that month
            }
        ]

    Notes:
        - Only messages during live streams are counted.
        - The leaderboard includes only the top 10 users by message volume.
        - If no chat data exists for that month, the API returns an empty list or 404.
    """
    return await call_hcs_api("/api/get_chat_leaderboard", {
        "channel_name": channel_name,
        "month": month
    })


@tool
async def get_user_changes(group: str, month: str) -> list[dict]:
    """
    Returns user activity changes (gains, losses, and net change) for all channels within a given VTuber group and month.

    Args:
        group (str): The VTuber group name (e.g., "Hololive", "Indie").
        month (str): The month to analyze in 'YYYY-MM' format (e.g., "2025-09").

    Returns:
        list[dict]: [
            {
                "channel": str,         # Channel name
                "users_gained": int,    # Number of users newly active this month
                "users_lost": int,      # Number of users active last month but not this one
                "net_change": int       # users_gained - users_lost
            }
        ]

    Notes:
        - A user is considered "active" if they sent at least 5 messages that month.
        - Channels with no data in either month are excluded.
        - Net change can be negative if losses exceed gains.
    """
    return await call_hcs_api("/api/get_user_changes", {
        "group": group,
        "month": month
    })


@tool
async def get_exclusive_chat_users(channel: str) -> list[dict]:
    """
    Returns the percentage of chat users who exclusively chat in a given channel (i.e., not active in any other channels of the same group).

    Args:
        channel (str): The VTuber channel name (e.g., "Pekora").

    Returns:
        list[dict]: [
            {
                "month": str,        # Month in 'YYYY-MM' format
                "percent": float     # Percentage of exclusive users for that month
            }
        ]

    Notes:
        - Only considers users within the same channel group (e.g., Hololive, Indie).
        - The percentage is (exclusive_users / total_users) × 100.
        - Useful for measuring how loyal a fanbase is to a specific channel.
    """
    return await call_hcs_api("/api/get_exclusive_chat_users", {
        "channel": channel
    })


@tool
async def get_message_type_percents(channel: str, language: str) -> list[dict]:
    """
    Returns monthly percentages of messages in a specified language for a given channel, along with the message rate (messages per minute).

    Args:
        channel (str): The VTuber channel name (e.g., "Miko").
        language (str): The target language code (one of: "EN", "JP", "KR", "RU").

    Returns:
        list[dict]: [
            {
                "month": str,             # Month in 'YYYY-MM' format
                "percent": float,         # Percentage of messages in the given language
                "message_rate": float     # Messages per minute for that language
            }
        ]

    Notes:
        - `message_rate` is computed as total language-specific messages / total stream minutes.
        - Only includes streams that have available chat logs.
        - Percentages exclude emoji-only messages.
        - Languages are detected by character set. Any message not categorized as "JP", "KR", or "RU" is considered "EN".
    """
    return await call_hcs_api("/api/get_message_type_percents", {
        "channel": channel,
        "language": language
    })


@tool
async def get_attrition_rates(
    channel: str,
    month: str = None,
    announce_date: str = None,
    graduation_date: str = None
) -> list[dict]:
    """
    Calculates attrition rates for a VTuber’s top 1000 chatters — showing what percentage
    continue chatting in Hololive channels after a graduation.

    You can call this tool in two ways:

    1. **Manual mode**
       - Use when analyzing a normal period.
       - Args:
         - channel: e.g., "Pekora"
         - month: e.g., "2025-09"

    2. **Graduation mode**
       - Use when a VTuber has announced and graduated.
       - Args:
         - channel: e.g., "Nanashi Mumei"
         - announce_date: e.g., "2025-03-27"
         - graduation_date: e.g., "2025-04-27"

    Returns:
        list[dict]: [
            {"month": "YYYY-MM", "percent": 85.23},
            ...
        ]

    Logic:
        - Baseline = 3 months ending 1 month before announce_date
        - Measurement starts 1 month after graduation_date
        - Measures continued activity in Hololive channels

    Announce/Graduation Dates/Fanbase Names:
    Mumei: 2025-03-27 / 2025-04-27 / Hoomans
    Fauna: 2024-11-30 / 2025-01-03 / Saplings
    Gura: 2025-04-15 / 2025-04-30 / Chumbuds
    Shion: 2025-03-06 / 2025-04-26 / Shiokko
    Chloe: 2024-11-29 / 2025-01-26 / Handlers (shiikuin)
    Ao: 2025-09-08 / 2025-09-08 / Dokusha

    You have no data on graduations other than the ones listed above. Data is only available for graduations after January 2025.
    """
    params = {"channel": channel}
    if announce_date and graduation_date:
        params.update({
            "announce_date": announce_date,
            "graduation_date": graduation_date
        })
    elif month:
        params["month"] = month
    else:
        raise ValueError("Must provide either `month` or both `announce_date` and `graduation_date`.")

    return await call_hcs_api("/api/get_attrition_rates", params)



@tool
async def get_jp_user_percent(channel: str) -> list[dict]:
    """
    Returns the monthly percentage of users whose messages are primarily in Japanese (>50%), excluding emoji-only messages.

    Args:
        channel (str): The VTuber channel name (e.g., "Marine").

    Returns:
        list[dict]: [
            {
                "month": str,             # Month in 'YYYY-MM' format
                "jp_user_percent": float  # Percentage of Japanese-dominant users for that month
            }
        ]

    Notes:
        - A user is classified as “Japanese-dominant” if over 50% of their non-emoji messages are in Japanese.
        - Useful for measuring audience localization or language shift trends over time.
    """
    return await call_hcs_api("/api/get_jp_user_percent", {
        "channel": channel
    })


@tool
async def get_channel_names() -> list[str]:
    """
    Returns a list of all known VTuber channel names in alphabetical order.

    Returns:
        list[str]: A list of channel names, e.g.:
            ["Aki", "Fubuki", "Gura", "Marine", "Miko", ...]

    Notes:
        - This endpoint does not require parameters.
    """
    return await call_hcs_api("/api/get_channel_names", {})


@tool
async def get_date_ranges() -> list[str]:
    """
    Returns the date range (earliest and latest stream) available in the database, for all channels with chat logs.

    Returns:
        list[str]: [
            "YYYY-MM-DD",   # Earliest end_time with chat log
            "YYYY-MM-DD"    # Latest end_time with chat log
        ]

    Notes:
        - Dates correspond to the `end_time` of streams that have valid chat logs.
        - Can be used to constrain queries or chart axes to valid data ranges.
    """
    return await call_hcs_api("/api/get_date_ranges", {})


@tool
async def get_number_of_chat_logs() -> int:
    """
    Returns the total number of video entries that have chat logs available.

    Returns:
        int: Total number of videos with chat logs.

    Notes:
        - Useful for summarizing dataset completeness or data coverage metrics.
    """
    return await call_hcs_api("/api/get_number_of_chat_logs", {})


@tool
async def get_num_messages() -> int:
    """
    Returns the total number of chat messages recorded in the dataset.

    Returns:
        int: The sum of all messages from all users across all channels.

    Notes:
        - Includes all messages (not just live ones) from every user and channel.
        - Can be used to report dataset scale or user engagement statistics.
    """
    return await call_hcs_api("/api/get_num_messages", {})


@tool
async def get_funniest_timestamps(channel: str, month: str) -> list[dict]:
    """
    Returns timestamps for the “funniest moments” in a channel’s streams for a given month,
    based on the chat reactions.

    Args:
        channel (str): The VTuber channel name (e.g., "Okayu").
        month (str): The month to query in 'YYYY-MM' format (e.g., "2025-09").

    Returns:
        list[dict]: [
            {
                "title": str,       # Video title
                "video_id": str,    # YouTube video ID
                "timestamp": int    # Offset (in seconds) relative to stream duration
            }
        ]

    Notes:
        - Only includes videos with a `funniest_timestamp` and valid chat logs.
        - Results are sorted by the video’s end time.
        - Useful for automatically generating highlight clip points.
    """
    return await call_hcs_api("/api/get_funniest_timestamps", {
        "channel": channel,
        "month": month
    })


@tool
async def get_user_info(identifier: str, month: str) -> dict:
    """
    Returns a summary of a user's chat activity across all channels for a given month,
    including message counts and percentile rankings per channel.

    Args:
        identifier (str): Either a YouTube user handle (starting with '@') or a YouTube channel ID (starting with "UC").
        month (str): The target month in 'YYYY-MM' format (e.g., '2025-09').

    Returns:
        dict: {
            "success": bool,
            "data": [
                {
                    "channel_name": str,     # Channel name where user chatted
                    "message_count": int,    # Number of messages by user
                    "percentile": float      # User's message frequency percentile for that channel
                }
            ]
        }

    Notes:
        - Percentile is relative to all users on that channel for the given month.
        - Useful for identifying how “active” or “core” a fan is within multiple communities.
    """
    return await call_hcs_api("/api/get_user_info", {
        "identifier": identifier,
        "month": month
    })


@tool
async def get_chat_engagement(month: Optional[str] = None, group: Optional[str] = None) -> dict:
    """
    Returns chat engagement metrics per channel, including total users, total messages,
    and average messages per user.

    Args:
        month (str, optional): The month to analyze, in 'YYYY-MM' format. Defaults to the current month.
        group (str, optional): Channel group (e.g., 'Hololive', 'Indie'). If omitted, includes all groups.

    Returns:
        dict: {
            "success": bool,
            "data": [
                {
                    "channel": str,
                    "total_users": int,
                    "total_messages": int,
                    "avg_messages_per_user": float
                }
            ]
        }

    Notes:
        - Channels are sorted by `avg_messages_per_user` in descending order.
        - This metric measures how “engaged” each fanbase is during a given month.
        - Particularly useful for identifying active or “chatty” communities.
    """
    return await call_hcs_api("/api/get_chat_engagement", {
        "month": month,
        "group": group
    })


@tool
async def get_video_highlights(channel_name: str, month: str) -> dict:
    """
    Fetches AI-generated highlight clips for a specific channel and month.
    Each highlight includes topic tags, summaries, and YouTube URLs with timestamps.

    Args:
        channel_name (str): The VTuber channel name (e.g., 'Korone').
        month (str): The month to query, in 'YYYY-MM' format (e.g., '2025-09').

    Returns:
        dict: {
            "success": bool,
            "data": [
                {
                    "video_id": str,
                    "video_title": str,
                    "timestamps": [
                        {
                            "topic": str,
                            "summary": str,
                            "youtube_url": str  # Direct link to timestamped clip
                        }
                    ]
                }
            ]
        }

    Notes:
        - Highlights are grouped by video.
    """
    return await call_hcs_api("/api/get_video_highlights", {
        "channel_name": channel_name,
        "month": month
    })


@tool
async def search_highlights(query: str) -> dict:
    """
    Searches AI-generated video highlights using semantic similarity and optional structured filters.

    Args:
        query (str): The natural-language query string.
                     You can include operators:
                       - channel:<name>   → Filter by channel
                       - from:<YYYY-MM-DD> → Start date
                       - to:<YYYY-MM-DD>   → End date
                     Example: "funny moments channel:Marine from:2025-01-01"

    Returns:
        dict: {
            "success": bool,
            "data": [
                {
                    "summary": str,             # Highlight summary text
                    "topic": str,               # Topic label
                    "video_id": str,            # YouTube video ID
                    "video_title": str,         # Stream title
                    "date": str,                # Stream date
                    "youtube_url": str,         # Timestamped YouTube link
                    "similarity_score": float   # 0–1 score; higher = more relevant
                }
            ]
        }

    Notes:
        - Supports both natural-language search and structured filters.
        - Uses vector similarity under the hood for semantic matching.
        - Ideal for content discovery, clip recommendation, or AI-assisted search interfaces.
    """
    return await call_hcs_api("/api/search_highlights", {
        "query": query
    })

@tool
async def search_hololive_shop(vtuber_name: str, language: Literal["en", "jp"] = "en") -> dict:
    """
    Searches the official Hololive shop for merchandise related to a specific VTuber.
    It can search either the English (en) or Japanese (jp) version of the site.
    Use Markdown to format the results for better readability, including product images.

    Args:
        vtuber_name (str): The name of the VTuber to search for (e.g., "Gawr Gura", "Tokino Sora").
                         English names work on both the 'en' and 'jp' sites.
        language (str): The shop language to use. Must be either "en" for English or "jp" for Japanese.
                      If the user's query is in Japanese, set this to "jp". Otherwise, default to "en".

    Returns:
        A dictionary containing a list of up to 10 products, including their name, price, link, and image URL.
    """
    return await call_hcs_api("/api/search_merchandise", {
        "vtuber_name": vtuber_name,
        "language": language
    })

@tool
async def get_channel_streams(
    channel: str,
    stream_type: str = "all",
    limit: int = 5
) -> dict:
    """
    Returns past, current, or upcoming streams/videos for a given VTuber channel.
    Use this to find out when a VTuber last streamed, is currently streaming, 
    or will stream next. Includes live streams, premieres, and regular video uploads.

    Args:
        channel (str): The name of the VTuber whose streams should be retrieved.
            Can be a single channel name or comma-separated list of channel names.
        stream_type (str): Type of content to retrieve. Options are:
            - "past" for completed streams, premieres, and video uploads
            - "live" for currently active streams or premieres
            - "upcoming" for scheduled future streams and premieres
            - "all" for all content types (default)
        limit (int): Maximum number of items to return per channel (default 5).
            For "last stream" queries, use limit=1.
            For "next stream" queries, use limit=1 with stream_type="upcoming".

    Returns:
        dict: {
            "success": True,
            "data": [
                {
                    "channel": str,           # Channel display name
                    "organization": str,      # Organization (e.g., "Hololive", "Indie")
                    "channel_id": str,        # YouTube channel ID
                    "streams": [
                        {
                            "title": str,             # Video/stream title
                            "video_id": str,          # YouTube video ID
                            "url": str,               # Full YouTube URL
                            "thumbnail": str,         # Thumbnail URL
                            "status": str,            # "live", "upcoming", or "completed"
                            "content_type": str,      # "stream", "premiere", or "video"
                            "published_at": str,      # ISO timestamp
                            "scheduled_start": str,   # ISO timestamp (for upcoming)
                            "started_at": str,        # ISO timestamp (for live/completed)
                            "ended_at": str,          # ISO timestamp (for completed)
                            "view_count": int,        # Total views (for completed)
                            "duration": str           # ISO 8601 duration (for completed)
                        }
                    ]
                }
            ]
        }
    
    Examples:
        - "What was Miko's last stream?" -> channel="Miko", stream_type="past", limit=1
        - "When is Azki streaming next?" -> channel="Azki", stream_type="upcoming", limit=1
        - "Is Roboco live right now?" -> channel="Roboco", stream_type="live"
        - "Show me Dooby's recent streams" -> channel="Dooby", stream_type="past", limit=5
        - "What videos has Nimi uploaded recently?" -> channel="Nimi", stream_type="past", limit=5
    """
    params = {
        "channel": channel,
        "stream_type": stream_type,
        "limit": limit
    }
    return await call_hcs_api("/api/get_channel_streams", params)

@tool
async def get_channel_metrics(channel: str) -> dict:
    """
    Returns channel metrics such as subscriber count, total views, and video count
    for a given VTuber channel. Use this to answer questions about channel statistics.

    Args:
        channel (str): The name of the VTuber whose channel metrics should be retrieved.
            Can be a single channel name or comma-separated list of channel names.

    Returns:
        dict: {
            "success": True,
            "data": [
                {
                    "channel": str,                    # Channel display name
                    "organization": str,               # Organization (e.g., "Hololive", "Indie")
                    "channel_id": str,                 # YouTube channel ID
                    "custom_url": str,                 # YouTube custom URL (e.g., "@ChannelName")
                    "description": str,                # Channel description (truncated)
                    "thumbnail": str,                  # Channel profile picture URL
                    "created_at": str,                 # Channel creation date (ISO timestamp)
                    "subscriber_count": int,           # Number of subscribers
                    "subscriber_count_hidden": bool,   # Whether sub count is hidden
                    "total_view_count": int,           # Total views across all videos
                    "video_count": int                 # Total number of uploaded videos
                }
            ]
        }
    
    Examples:
        - "How many subscribers does Miko have?" -> channel="Miko"
        - "What are Azki's channel stats?" -> channel="Azki"
        - "Compare Dooby and Nimi's subscriber counts" -> channel="Dooby,Nimi"
        - "How many videos has Roboco uploaded?" -> channel="Roboco"
        - "How many total views does Miko have?" -> channel="Miko"
    """
    params = {
        "channel": channel
    }
    return await call_hcs_api("/api/get_channel_metrics", params)

def get_api_tools():
    """
    Returns a list of all defined API tools for the agent to use.
    This function is called by the agent to get its capabilities.
    """
    return [
        get_recommendations,
        get_monthly_streaming_hours,
        get_group_total_streaming_hours,
        get_group_avg_streaming_hours,
        get_group_max_streaming_hours,
        get_group_chat_makeup,
        get_common_users,
        get_common_users_matrix,
        get_common_members,
        get_group_membership_summary,
        get_group_streaming_hours_diff,
        get_chat_leaderboard,
        get_user_changes,
        get_exclusive_chat_users,
        get_message_type_percents,
        get_attrition_rates,
        get_jp_user_percent,
        get_channel_names,
        get_date_ranges,
        get_number_of_chat_logs,
        get_num_messages,
        get_funniest_timestamps,
        get_user_info,
        get_chat_engagement,
        get_video_highlights,
        search_highlights,
        search_hololive_shop,
        get_channel_streams,
        get_channel_metrics
    ]
