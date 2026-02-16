import numpy as np
import evaluate
import json


# def compute_metrics(eval_pred):
#      metric = evaluate.load('accuracy')
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)

examples = ["Open YouTube", 
           "Search for cooking videos on YouTube", 
           "Turn up the volume", 
           "Find action movies",
           "Good to see you again!",
           "Mute the TV",
           "I want to watch Ronaldo Goals",
           "Launch Netflix",
           ]

intents = ["open_app", 
           "open_app_and_search",
           "settings",
           "search",
           "out_of_scope",
           "settings",
           "search",
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
    "netflix": "Netflix",
    "youtube": "YouTube",
    "prime video": "Prime Video",
    "prime": "Prime Video",
    "disney": "Disney Plus",
    "disney plus": "Disney Plus",
    "hulu": "Hulu",
    "spotify": "Spotify",
    "hbo": "HBO Max",
    "apple tv": "Apple TV",
    "streaming service": "streaming service",
    "video content app": "video content app",
    "music player": "music player",
}

# Settings mapping
SETTINGS_PATTERNS = {
    "volume_up": ["volume up", "turn up", "louder", "raise volume", "increase volume"],
    "volume_down": ["volume down", "turn down", "lower the audio", "decrease volume", "quieter"],
    "volume_max": ["max volume", "maximum volume", "full volume", "maximize volume"],
    "mute": ["mute", "silence"],
    "unmute": ["unmute", "unsilence"],
    "brightness_up": ["brighter", "brightness up", "increase brightness", "brighten"],
    "brightness_down": ["darker", "brightness down", "reduce brightness", "dim"],
}

def extract_settings_action(command):
    """Extract settings action using pattern matching"""
    command_lower = command.lower()
    
    for action, patterns in SETTINGS_PATTERNS.items():
        for pattern in patterns:
            if pattern in command_lower:
                return json.dumps({"settings_action": action}, separators=(",", ":"))
    
    # Fallback
    return json.dumps({"settings_action": "unknown"}, separators=(",", ":"))


def extract_app_name(command):
    """Extract app name using simple matching - NO LLM!"""
    command_lower = command.lower()
    
    for keyword, app_name in KNOWN_APPS.items():
        if keyword in command_lower:
            return json.dumps({"app_name": app_name}, separators=(",", ":"))
    
    # Fallback
    return json.dumps({"app_name": "unknown"}, separators=(",", ":"))



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