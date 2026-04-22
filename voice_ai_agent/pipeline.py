# from speech import speech_model
from voice_ai_agent.classiefier import setting_classifier_model, intent_classifier_model
from voice_ai_agent.language_model import LanguageModel
from voice_ai_agent.utils import evaluate_perf_latency, extract_app_name, extract_settings_action
import torch
import re
import json

# Force CPU execution in environments without CUDA support
llm = LanguageModel(device='cpu')


def _parse_open_app_and_search(command):
    text_lower = command.lower().strip()
    patterns = [
        r"(?:open|launch|start|go to|switch to)\s+(?P<app>.+?)\s+(?:and|,)\s*(?:search for|find|search|look for|look up)\s+(?P<query>.+)$",
        r"(?:search for|find|look for|look up)\s+(?P<query>.+)\s+(?:on|in)\s+(?P<app>.+)$"
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            app_raw = match.group('app').strip()
            query_raw = match.group('query').strip()
            if app_raw and query_raw:
                return {
                    'app_name': app_raw.title() if app_raw.islower() else app_raw,
                    'search_query': query_raw,
                }
    return {}


def agent_system_with_class_llm(command):
    # speech_out = speech_model(command)
    classifier_out = intent_classifier_model(command)[0]
    llm_out = llm.generate(command, classifier=classifier_out)

    return {
        "intent": classifier_out,
        "entity": llm_out,
    }


def agent_system_setclass_appmatch(command):
    # speech_out = speech_model(command)
    normalized = command.strip().lower()
    greeting_phrases = {
        "hello",
        "hi",
        "hey",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
    }

    if normalized in greeting_phrases or normalized.startswith("hello") or normalized.startswith("hi"):
        entities = {}
        return {
            "intent": "out_of_scope",
            "entity": entities,
            "entities": entities,
        }

    classifier_out = intent_classifier_model(command)[0]

    if classifier_out == "open_app":
        entities = extract_app_name(command)

    elif classifier_out == "settings":
        entities = setting_classifier_model(command)

    elif classifier_out == "out_of_scope":
        entities = {
            "message": "I can help with searching for content, opening applications, and control settings."
        }

    elif classifier_out == "open_app_and_search":
        entities = _parse_open_app_and_search(command)
        if not entities:
            llm_out = llm.generate(command, classifier=classifier_out)
            try:
                entities = json.loads(llm_out)
            except (json.JSONDecodeError, TypeError):
                entities = {}

    elif classifier_out == "search":
        llm_out = llm.generate(command, classifier="search")
        try:
            entities = json.loads(llm_out)
        except (json.JSONDecodeError, TypeError):
            entities = {}

    else:
        entities = {}

    return {
        "intent": classifier_out,
        "entity": entities,
        "entities": entities,
    }


    


