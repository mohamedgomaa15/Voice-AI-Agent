import numpy as np
import evaluate

metric = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

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
           "search",
           "settings",
           "search",
           "None",
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