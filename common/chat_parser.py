import re
import emoji
import regex

# Matches a single YouTube-style shortcode, e.g. :_konkonmori: or :face_with_tears_of_joy:
_SHORTCODE_RE = re.compile(r":[^:\s]+:")
# Characters that may appear between/inside emoji sequences but carry no text meaning
_EMOJI_FILLER_RE = re.compile(r"[\s\u200d\ufe0e\ufe0f]")

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

def _is_pure_emoji(msg):
    """
    Returns True if msg consists solely of YouTube shortcodes and/or unicode
    emoji (optionally separated by whitespace), and contains at least one of them.
    """
    # Remove all shortcodes
    without_shortcodes = _SHORTCODE_RE.sub("", msg)
    # Remove all unicode emoji
    without_emoji = emoji.replace_emoji(without_shortcodes, replace="")
    # Remove whitespace / ZWJ / variation selectors that may remain
    remainder = _EMOJI_FILLER_RE.sub("", without_emoji)
    if remainder:
        return False
    # Ensure we actually removed something (i.e. the message wasn't just whitespace)
    return without_shortcodes != msg or without_emoji != without_shortcodes


def categorize_message(message):
    """
    Categorize a message as emoji, language-based, or other.
    ...
    """
    if not message or not isinstance(message, str):
        return None

    msg_stripped = message.strip()
    if not msg_stripped:
        return None

    # Message is purely a chain of YouTube shortcodes and/or unicode emoji
    if _is_pure_emoji(msg_stripped):
        return "emoji"

    msg_lower = msg_stripped.lower()

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