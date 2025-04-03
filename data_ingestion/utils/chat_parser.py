import re
import emoji
import regex

# Determine membership rank from badge text
def parse_membership_rank(badge_text):
    """
    Parse the membership rank from a badge text.

    This function interprets the badge text to determine the membership rank based on time.
    It returns the rank in months. If the user is a new member, it returns 0. If the badge 
    text specifies a time period (in months or years), it calculates the equivalent in months.
    If no valid membership information is found or the user is not a member, it returns -1.

    Args:
        badge_text (str): The text on the badge indicating membership status.

    Returns:
        int: The membership rank in months, or -1 if no valid membership is determined.
    """
    # User is not a member
    if not badge_text:
        return -1
    
    rank_text = badge_text.lower().strip()

    if "new member" in rank_text:
        return 0
    
    match = re.search(r"(\d+)\s*(month|year)", rank_text)
    if match:
        number = int(match.group(1))
        unit = match.group(2)
        return number * 12 if unit.startswith("year") else number
    
    return -1

# Categorize message into language or emoji
def categorize_message(message):
    """
    Categorize a message as emoji, language-based, or other.

    This function analyzes a given message to determine its category. It first checks if the 
    message consists of YouTube style emotes or Unicode emojis, categorizing it as "emoji" 
    if so. If not, it then checks for the presence of Japanese, Korean, or Russian language 
    characters using regular expressions and categorizes the message accordingly as "jp", "kr", 
    or "ru". If the message is purely numeric, it returns "number". For all other messages, 
    it defaults to "es_en_id".

    Args:
        message (str): The message to be categorized.

    Returns:
        str: The category of the message, which can be "emoji", "jp", "kr", "ru", "number", 
             or "es_en_id".
    """

    if not message or not isinstance(message, str):
        return None

    msg_lower = message.strip().lower()

    if not msg_lower:
        return None

    # Check for YouTube style emotes and unicode emojis. Message must be start and end with either of these.
    if (msg_lower.startswith(":") and msg_lower.endswith(":")) or (emoji.is_emoji(msg_lower[0]) and emoji.is_emoji(msg_lower[-1])):
        return "emoji"
    
    # Check for language-based messages
    jp_regex = regex.compile(r"[\p{Hiragana}\p{Katakana}\p{Han}]+")
    jp_punctuation = regex.compile(r"[！？]")
    jp_laugh = regex.compile(r"^[wｗ]+$")
    kr_regex = regex.compile(r"[\p{Hangul}]+")
    ru_regex = regex.compile(r"[\p{Cyrillic}]+")

    if jp_regex.search(msg_lower) or jp_punctuation.search(msg_lower) or jp_laugh.search(msg_lower):
        return "jp"
    elif kr_regex.search(msg_lower):
        return "kr"
    elif ru_regex.search(msg_lower):
        return "ru"
    elif msg_lower.isnumeric():
        return "number"
    
    return "es_en_id"