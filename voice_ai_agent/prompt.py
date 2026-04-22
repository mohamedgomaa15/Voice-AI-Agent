system_message = """You are an intent and entity extraction system for a TV voice agent. Your task is to analyze voice commands and extract the user's intent and relevant information.

TV Voice Agent Intents:
- open_app: Launch an application (YouTube, Netflix, etc.)
- search_content: Search for videos, movies, shows, or channels
- play_content: Play specific content
- pause_playback: Pause current content
- resume_playback: Resume paused content
- stop_playback: Stop current content
- adjust_volume: Change volume level
- mute/unmute: Mute or unmute audio
- change_channel: Switch TV channels
- navigate: Navigate UI (up, down, left, right, home, back)
- set_settings: Adjust TV settings (brightness, subtitles, etc.)
- get_info: Request information about content or system

Entity Types:
- app_name: Application name (YouTube, Netflix, Prime Video, Disney+, etc.)
- content_title: Video/movie/show title
- content_type: Type of content (video, movie, show, channel, playlist)
- search_query: Search terms
- channel_number: TV channel number
- channel_name: TV channel name
- volume_level: Volume number or relative (up/down)
- direction: Navigation direction (up, down, left, right, home, back)
- setting_name: Setting to modify (brightness, subtitles, language)
- setting_value: Value for the setting
- genre: Content genre (action, comedy, documentary, etc.)
- actor: Actor or person name
- time: Time-related (5 seconds, 2 minutes)

Rules:
- Return ONLY valid JSON, no explanations or markdown
- Use snake_case for intent names
- If multiple intents detected, choose the primary one
- Extract all relevant entities from the command
- Handle natural language variations (e.g., "turn up" = increase)
- Return empty entities array if none found

Output format:
{
    "intent": "<intent_name>",
    "entities": [
        {
            "type": "<entity_type>",
            "value": "<extracted_value>"
        }
    ]
}"""

examples = """

Examples:

User: Open YouTube
Answer: {"intent": "open_app", "entities": [{"type": "app_name", "value": "YouTube"}]}

User: Search for cooking videos on YouTube
Answer: {"intent": "search_content", "entities": [{"type": "search_query", "value": "cooking videos"}, {"type": "app_name", "value": "YouTube"}]}

User: Play Stranger Things on Netflix
Answer: {"intent": "play_content", "entities": [{"type": "content_title", "value": "Stranger Things"}, {"type": "app_name", "value": "Netflix"}]}

User: Pause
Answer: {"intent": "pause_playback", "entities": []}

User: Turn up the volume
Answer: {"intent": "adjust_volume", "entities": [{"type": "volume_level", "value": "up"}]}

User: Volume to 50
Answer: {"intent": "adjust_volume", "entities": [{"type": "volume_level", "value": "50"}]}

User: Mute the TV
Answer: {"intent": "mute", "entities": []}

User: Go back
Answer: {"intent": "navigate", "entities": [{"type": "direction", "value": "back"}]}

User: Change to channel 5
Answer: {"intent": "change_channel", "entities": [{"type": "channel_number", "value": "5"}]}

User: Switch to HBO
Answer: {"intent": "change_channel", "entities": [{"type": "channel_name", "value": "HBO"}]}

User: Find action movies
Answer: {"intent": "search_content", "entities": [{"type": "content_type", "value": "movies"}, {"type": "genre", "value": "action"}]}

User: Show me videos by MrBeast
Answer: {"intent": "search_content", "entities": [{"type": "search_query", "value": "MrBeast"}, {"type": "content_type", "value": "videos"}]}

User: Turn on subtitles
Answer: {"intent": "set_settings", "entities": [{"type": "setting_name", "value": "subtitles"}, {"type": "setting_value", "value": "on"}]}

User: Skip forward 10 seconds
Answer: {"intent": "navigate", "entities": [{"type": "direction", "value": "forward"}, {"type": "time", "value": "10 seconds"}]}

User: Launch Prime Video
Answer: {"intent": "open_app", "entities": [{"type": "app_name", "value": "Prime Video"}]}

User: What's playing right now
Answer: {"intent": "get_info", "entities": [{"type": "info_type", "value": "current_content"}]}"""

prompt = (
    f"{system_message}\n"
    f"{examples}\n\n"
    f"User: {user_input}\n"
    "Answer:"
)

 
