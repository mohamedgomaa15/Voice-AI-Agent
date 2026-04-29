import numpy as np
import evaluate
import json
import os


# def compute_metrics(eval_pred):
#      metric = evaluate.load('accuracy')
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

examples = ["دور على أخبار الرياضة", 
           "زود الإضاءة", 
           "دور على أفلام جديدة", 
           "افتح تطبيق نتفليكس",
           "اختبرني في الرياضيات",
           "طفي الوضع الصامت",
           "روح على YouTube واعرض كتب مفيد",
           "شغل برنامج Chrome",
           ]

intents = ["search", 
           "settings",
           "search",
           "open_app",
           "out_of_scope",
           "settings",
           "open_app_and_search",
           "open_app",
           ]

entities = [
    [{"type": "app_name", "value": "YouTube"}],
    [{"type": "search_query", "value": "cooking videos"}, {"type": "app_name", "value": "YouTube"}],
    [{"type": "settings_action", "value": "volume_up"}],
    [{"type": "content_type", "value": "movies"}, {"type": "genre", "value": "action"}],
    [],
    [{"type": "settings_action", "value": "mute"}],
    [{"type": "content_title", "value": "Ronaldo Goals"}],
    [{"type": "app_name", "value": "Netflix"}],
]

# Known apps list
KNOWN_APPS = {
    "Netflix": ["netflix", "net flix", "نتفليكس"],
    "YouTube": ["youtube", "you tube", "يوتيوب"],
    "Google": ["google", "google chrome", "chrome", "chrome browser", "جوجل", "كروم"],
    "Prime Video": ["prime video", "prime_video", "video prime", "prime", "برايم فيديو"],
    "Disney": ["disney", "disney plus", "disney+", "ديزني", "ديزني بلس"],
    "Hulu": ["hulu", "هولو"],
    "Spotify": ["spotify", "سبوتيفاي"],
    "HBO": ["hbo max", "hbo", "اتش بي او"],
    "Apple TV": ["apple tv", "apple_tv", "ابل تي في"],
    "Streaming Service": ["streaming service", "streaming", "video streaming", "streaming_service", "خدمة بث"],
    "Video Content": ["video content app", "video content", "content app", "video_app",  "video app", "تطبيق محتوى الفيديو"],
    "Music Player": ["music player", "music", "music_player", "موسيقى", "مشغل الموسيقى"],
    "Paramount": ["paramount", "paramount plus", "paramount+", "باراماونت", "باراماونت بلس"],
}

# Settings mapping with flexible pattern matching
SETTINGS_PATTERNS = {
    "volume_max": {
        "targets": ["volume", "audio", "sound", "tv", "speaker"],
        "actions": ["max", "maximum", "full", "maximize", "maximizing", "maxed", "highest", "blast", "all the way"]
    },
    "volume_up": {
        "targets": ["volume", "audio", "sound", "tv", "speaker", "noise"],
        "actions": ["up", "turn up", "louder", "loud", "raise", "raising", "increase", "increasing", "high", "higher", "boost", "boosting", "more", "amplify"]
    },
    "volume_down": {
        "targets": ["volume", "audio", "sound", "tv", "speaker", "noise"],
        "actions": ["down", "turn down", "lower", "low", "decrease", "decreasing", "quieter", "quiet", "reduce", "reducing", "less", "minimize", "minimizing", "softer", "soft"]
    },
    "unmute": {
        "targets": [""],
        "actions": ["unmute", "unsilence", "unsilent", "unmuting", "unmuted", "turn on audio", "sound back", "restore sound", "audio back"]
    },
    "mute": {
        "targets": [""],  # Can work without target word
        "actions": ["mute", "silence", "muting", "silent", "mutes", "muted", "shut up", "shut", "off", "turn off audio", "kill sound"]
    },
    "brightness_max": {
        "targets": ["brightness", "screen", "display", "picture"],
        "actions": ["maximum", "full", "maximize", "brightest", "all the way"]
    },
    "brightness_min": {
        "targets": ["brightness", "screen", "display", "picture"],
        "actions": ["minimum", "minimize", "darkest", "lowest"]
    },
    "brightness_up": {
        "targets": ["brightness", "screen", "display", "picture"],
        "actions": ["max", "up", "brighter", "increase", "increasing", "brighten", "brightening", "raise", "raising", "more", "high", "higher"]
    },
    "brightness_down": {
        "targets": ["brightness", "screen", "display", "picture"],
        "actions": ["min", "down", "darker", "decrease", "decreasing", "reduce", "reducing", "dim", "dimming", "lower", "lowering", "less", "dark"]
    },
}

def extract_settings_action(command):
    """
    Extract settings action using flexible pattern matching.
    Handles cases like:
    - "turn up the volume"
    - "the sound is too loud, decrease it"
    - "reduce the tv volume"
    - "make screen brighter"
    """
    command_lower = command.lower()
    
    # Try to match each settings action
    for action_name, patterns in SETTINGS_PATTERNS.items():
        targets = patterns["targets"]
        actions = patterns["actions"]
        
        # Check if any target word is present
        
        target_found = any(target_word in command_lower for target_word in targets)

        if not target_found:
            continue
        
        action_found = any(action_word in command_lower for action_word in actions)
            
        if target_found and action_found:
                return {"settings_action": action_name}
    
    # Fallback: unknown action
    return {"settings_action": "unknown"}

def _load_apps(path):
    try:
        # Get project root (parent of voice_ai_agent)
        base_dir = os.path.dirname(os.path.dirname(__file__))

        full_path = os.path.join(base_dir, path)

        with open(full_path, "r", encoding="utf-8") as f:
            apps = json.load(f)

        return apps

    except Exception as e:
        print("Error loading apps:", e)
        return []


def extract_app_name(command):
    """Extract app name using simple matching - NO LLM!"""
    command_lower = command.lower()

    # remove punctuation 
    for punc in [',', '.', '!', '?', '،', '؟']:
        command_lower = command_lower.replace(punc, '')
    
    command_lower = command_lower.strip()

    apps = _load_apps('data/clean_apps.json')
    
    # Track best match for fuzzy matching
    best_match = None
    best_match_score = 0
    
    for app in apps:
        app_name = app['name']
        aliases = app.get('aliases', [])
        
        # Check all possible names (name + aliases)
        all_names = [app_name.lower()] + [alias.lower() for alias in aliases]
        
        for name_variant in all_names:
            # Exact word match (better than substring)
            words = command_lower.split()
            for word in words:
                if name_variant == word:
                    return {"app_name": app_name}
            
            # Partial match with scoring
            if name_variant in command_lower:
                score = len(name_variant)
                if score > best_match_score:
                    best_match_score = score
                    best_match = app_name
    
    if best_match:
        return {"app_name": best_match}
    
    # Fallback
    return {"app_name": "unknown"}



def evaluate_perf_latency(results_file="results.json", expected_file="test_data.json"):
    """Simple evaluation: accuracy and timing per category"""
    
    # Load files
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(expected_file, 'r') as f:
        expected = json.load(f)
    
    # Create lookup
    expected_dict = {item['idx']: item['llm_out'] for item in expected}
    
    # Track metrics per category
    categories = {
        'search': {'correct': 0, 'total': 0, 'times': []},
        'open_app': {'correct': 0, 'total': 0, 'times': []},
        'open_app_and_search': {'correct': 0, 'total': 0, 'times': []},
        'settings': {'correct': 0, 'total': 0, 'times': []},
        'out_of_scope': {'correct': 0, 'total': 0, 'times': []}
    }
    
    total_correct = 0
    total_examples = len(results)
    
    # Evaluate each example
    for result in results:
        idx = result['idx']
        category = result['classifier_out']
        actual = result['llm_out']
        expected_output = expected_dict[idx]
        total_time = result['total_time']
        
        # Check if correct (1 or 0)
        if category == "search": 
            score = 1 if actual != "null" and expected_output != "null" else 0
        else:
            score = 1 if actual == expected_output else 0
        
        # Update category stats
        categories[category]['total'] += 1
        categories[category]['correct'] += score
        categories[category]['times'].append(total_time)
        
        # Update total
        total_correct += score
    
    # Per category
    for cat_name, stats in categories.items():
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            avg_time = np.mean(stats['times'])
            print(f"\n{cat_name.upper()}")
            print(f"  Accuracy: {stats['correct']}/{stats['total']} = {accuracy:.2f}%")
            print(f"  Avg Time: {avg_time:.2f}s")
    
    # Total
    total_accuracy = (total_correct / total_examples) * 100
    all_times = []
    for stats in categories.values():
        all_times.extend(stats['times'])
    avg_total_time = np.mean(all_times)
    
    print("\n" + "=" * 60)
    print("TOTAL")
    print(f"  Accuracy: {total_correct}/{total_examples} = {total_accuracy:.2f}%")
    print(f"  Avg Time: {avg_total_time:.2f}s")
    print("=" * 60)