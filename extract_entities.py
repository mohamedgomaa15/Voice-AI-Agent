import time
import ast
from typing import Dict, List, Tuple, Optional
from transformers import pipeline
import argparse
# Assuming you have llm_generate function from your notebook
from language_model import llm_generate
from utils import examples, intents as true_intents, entities as true_entities

class HybridIntentSystem:
    def __init__(self, classifier_model_path: str = "mohamedgomaaa/intent-classifier-multilingual"):
        """
        Initialize hybrid system with your fine-tuned classifier
        """
        # Load the fine-tuned classifier
        self.classifier = pipeline(
            "text-classification",
            model=classifier_model_path,
            top_k=None
        )
        
        # Intent to prompt mapping
        self.intent_prompts = {
            "search": """
            Extract the search query from the user's input. 
            Return ONLY a JSON object in this format:
            {{"search_query": "extracted query here"}}
            
            User input: "{user_input}"
            """,
            
            "open_app": """
            Extract the app name from the user's input.
            Return ONLY a JSON object in this format:
            {{"app_name": "extracted app name here"}}
            
            User input: "{user_input}"
            """,
            
            "open_app_and_search": """
            Extract BOTH the app name and search query from the user's input.
            Return ONLY a JSON object in this format:
            {{"app_name": "app name here", "search_query": "search query here"}}
            
            User input: "{user_input}"
            """,
            
            "settings": """
            Extract the settings action and any parameters from the user's input.
            Common settings actions: volume_up, volume_down, mute, brightness_up, 
            brightness_down, change_channel, power_off, etc.
            Return ONLY a JSON object in this format:
            {{"settings_action": "action name here", "parameter": "value if any"}}
            
            User input: "{user_input}"
            """,
            
            "play_media": """
            Extract media details from the user's input.
            Return ONLY a JSON object in this format:
            {{"content_type": "movie/show/video", "title": "content title", "platform": "platform if specified"}}
            
            User input: "{user_input}"
            """,
            
            "out_of_scope": """
            This is not a valid command. Return empty entities.
            Return ONLY: {{}}
            
            User input: "{user_input}"
            """
        }
        
        # Metrics tracking
        self.metrics = {
            "total_queries": 0,
            "classifier_correct": 0,
            "llm_correct": 0,
            "total_time": 0,
            "classifier_times": [],
            "llm_times": []
        }
    
    def classify_intent(self, text: str) -> Tuple[str, float, float]:
        """
        Fast intent classification using the trained classifier
        Returns: (intent_label, confidence, processing_time)
        """
        start_time = time.time()
        classifier_output = self.classifier(text)
        result = None
        if isinstance(classifier_output, list) and len(classifier_output) > 0:
            first = classifier_output[0]
            if isinstance(first, list) and len(first) > 0:
                result = first[0]
            elif isinstance(first, dict):
                result = first
            else:
                result = first
        elif isinstance(classifier_output, dict):
            result = classifier_output
        else:
            # Fallback: coerce to string label with low confidence
            result = {"label": str(classifier_output), "score": 0.0}
        end_time = time.time()
        
        processing_time = end_time - start_time
        self.metrics["classifier_times"].append(processing_time)
        
        label = result.get("label") if isinstance(result, dict) else str(result)
        score = result.get("score", 0.0) if isinstance(result, dict) else 0.0

        return label, score, processing_time
    
    def extract_entities_with_llm(self, text: str, intent: str) -> Optional[Dict]:
        """
        Use LLM with intent-specific prompt to extract entities
        """
        if intent not in self.intent_prompts or intent == "out_of_scope":
            return {}  # No entities for out_of_scope or unknown intents
        
        # Format the prompt for this intent
        prompt = self.intent_prompts[intent].format(user_input=text)
        
        start_time = time.time()
        llm_response = llm_generate(prompt)

        # Robust parsing: try json.loads, then ast.literal_eval, then extract JSON substring
        import json, re
        entities = {}
        resp = llm_response.strip()

        # Always log raw LLM response for debugging when it doesn't parse
        def _log_raw(resp_text, intent_label, query_text):
            try:
                with open('llm_raw.log', 'a', encoding='utf-8') as f:
                    f.write(f"INTENT={intent_label} QUERY={query_text} RESPONSE={resp_text}\n---\n")
            except Exception:
                pass

        entities_candidate = None
        try:
            # If the model returned extra text, try to extract the first JSON object
            if not resp.startswith("{"):
                m = re.search(r"\{.*\}", resp, re.DOTALL)
                resp_json = m.group(0) if m else resp
            else:
                resp_json = resp

            # Try json first (safer)
            entities_candidate = json.loads(resp_json)
        except Exception:
            try:
                entities_candidate = ast.literal_eval(resp_json)
            except Exception as e:
                _log_raw(resp, intent, text)
                # Parsing failed; we'll fallback to heuristics below
                entities_candidate = None

        # Normalize different response shapes into a simple entity dict
        if isinstance(entities_candidate, dict):
            if "entities" in entities_candidate and isinstance(entities_candidate["entities"], list):
                normalized = {}
                for ent in entities_candidate["entities"]:
                    if isinstance(ent, dict) and "type" in ent and "value" in ent:
                        key = ent["type"]
                        val = ent["value"]
                        if key in normalized:
                            if isinstance(normalized[key], list):
                                normalized[key].append(val)
                            else:
                                normalized[key] = [normalized[key], val]
                        else:
                            normalized[key] = val
                entities = normalized
            else:
                # Assume the dict already maps entity_name->value
                entities = entities_candidate

        end_time = time.time()

        # If parsing failed or entities empty for a known intent, apply deterministic heuristics
        if (not entities or entities == {}) and intent in ["settings", "search", "open_app", "open_app_and_search"]:
            heur = self._heuristic_extract(text, intent)
            # Merge heuristic entities only if we didn't get anything from LLM
            if heur:
                entities = heur
        
        processing_time = end_time - start_time
        self.metrics["llm_times"].append(processing_time)
        
        return entities
    
    def process_query(self, text: str) -> Dict:
        """
        Main processing pipeline
        """
        self.metrics["total_queries"] += 1
        start_time = time.time()
        
        # Step 1: Fast intent classification
        intent, confidence, classifier_time = self.classify_intent(text)
        
        # Step 2: Entity extraction with LLM (only for certain intents)
        if intent in ["search", "open_app", "open_app_and_search", "settings", "play_media"]:
            entities = self.extract_entities_with_llm(text, intent)
        else:
            entities = {}
        
        end_time = time.time()
        total_time = end_time - start_time
        self.metrics["total_time"] += total_time
        
        return {
            "text": text,
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "timing": {
                "classifier_ms": classifier_time * 1000,
                "total_ms": total_time * 1000
            }
        }
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate and return performance metrics
        """
        if not self.metrics["classifier_times"]:
            return {}
        
        avg_classifier_time = sum(self.metrics["classifier_times"]) / len(self.metrics["classifier_times"])
        avg_llm_time = sum(self.metrics["llm_times"]) / len(self.metrics["llm_times"]) if self.metrics["llm_times"] else 0
        
        return {
            "total_queries": self.metrics["total_queries"],
            "avg_classifier_time_ms": avg_classifier_time * 1000,
            "avg_llm_time_ms": avg_llm_time * 1000,
            "avg_total_time_ms": self.metrics["total_time"] / self.metrics["total_queries"] * 1000 if self.metrics["total_queries"] > 0 else 0,
            "classifier_accuracy": self.metrics["classifier_correct"] / self.metrics["total_queries"] if self.metrics["total_queries"] > 0 else 0,
            "llm_accuracy": self.metrics["llm_correct"] / len(self.metrics["llm_times"]) if self.metrics["llm_times"] else 0
        }

    def _heuristic_extract(self, text: str, intent: str) -> Dict:
        """
        Fallback rule-based extraction for common intents when LLM fails or returns malformed JSON.
        """
        import re
        t = text.lower()
        out = {}

        if intent == "settings":
            # Volume controls
            if re.search(r"mute|unmute", t):
                out["settings_action"] = "mute"
                # Try to detect target
                m = re.search(r"mute the (\w+)", t)
                if m:
                    out["parameter"] = m.group(1)
                return out

            if re.search(r"turn up|increase|raise", t):
                out["settings_action"] = "turn_up"
                if "volume" in t:
                    out["parameter"] = "volume"
                return out

            if re.search(r"turn down|decrease|lower", t):
                out["settings_action"] = "turn_down"
                if "volume" in t:
                    out["parameter"] = "volume"
                return out

            if re.search(r"brightness", t):
                if re.search(r"up|increase|raise", t):
                    out["settings_action"] = "brightness_up"
                elif re.search(r"down|decrease|lower", t):
                    out["settings_action"] = "brightness_down"
                else:
                    out["settings_action"] = "change_brightness"
                return out

        if intent in ["search", "open_app_and_search"]:
            # Patterns like "Search for X on YouTube" or "Find X on Netflix"
            m = re.search(r"search for (.+) on (\w+)", t)
            if not m:
                m = re.search(r"find (.+) on (\w+)", t)
            if m:
                out["search_query"] = m.group(1).strip()
                out["app_name"] = m.group(2).strip().title()
                return out

            # Pattern like "Search YouTube for cooking videos"
            m = re.search(r"(search|find) (.+) on (\w+)", t)
            if m:
                out["search_query"] = m.group(2).strip()
                out["app_name"] = m.group(3).strip().title()
                return out

            # If just "Search for X" -> search_query only
            m = re.search(r"search for (.+)", t)
            if m:
                out["search_query"] = m.group(1).strip()
                return out

            # Look for "on <App>" suffix
            m = re.search(r"(.+) on (\w+)$", t)
            if m:
                out["search_query"] = m.group(1).strip()
                out["app_name"] = m.group(2).strip().title()
                return out

        if intent == "open_app":
            # Try to extract the app name
            m = re.search(r"open (.+)$", t)
            if m:
                out["app_name"] = m.group(1).strip().title()
                return out

        return out

# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid intent system interactive runner")
    parser.add_argument("--eval", action="store_true", help="Run evaluation using utils.examples (existing behavior)")
    parser.add_argument("--query", type=str, help="Process a single query and exit")
    args = parser.parse_args()

    # Create system instance
    system = HybridIntentSystem()

    def run_evaluation():
        for query, true_intent, true_ent in zip(examples, true_intents, true_entities):
            result = system.process_query(query)

            # Intent accuracy
            if result['intent'] == true_intent:
                system.metrics['classifier_correct'] += 1

            # Normalize predicted entities -> list of dicts
            pred_entities = []
            for k, v in result['entities'].items():
                if isinstance(v, list):
                    for item in v:
                        pred_entities.append({'type': k, 'value': item})
                else:
                    pred_entities.append({'type': k, 'value': v})

            # Check that every true entity is found in predicted entities
            llm_ok = True
            for te in true_ent:
                found = any(pe.get('type') == te.get('type') and str(pe.get('value')).lower() == str(te.get('value')).lower() for pe in pred_entities)
                if not found:
                    llm_ok = False
                    break
            if llm_ok:
                system.metrics['llm_correct'] += 1

            print(f"\nQuery: {query}")
            print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2%})")
            print(f"Entities: {result['entities']}")
            print(f"Time: {result['timing']['total_ms']:.2f}ms")

        # Get performance metrics
        metrics = system.get_performance_metrics()
        print(f"\n{'='*50}")
        print("PERFORMANCE METRICS:")
        print(f"{'='*50}")
        for key, value in metrics.items():
            if "accuracy" in key:
                print(f"{key}: {value:.2%}")
            elif "time" in key:
                print(f"{key}: {value:.2f}ms")
            else:
                print(f"{key}: {value}")

    if args.eval:
        run_evaluation()
    elif args.query:
        res = system.process_query(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Intent: {res['intent']} (confidence: {res['confidence']:.2%})")
        print(f"Entities: {res['entities']}")
        print(f"Time: {res['timing']['total_ms']:.2f}ms")
    else:
        # Interactive prompt
        print("Interactive mode. Enter queries (type 'exit' or empty to quit).")
        while True:
            try:
                q = input("> ").strip()
            except EOFError:
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            res = system.process_query(q)
            print(f"Intent: {res['intent']} (confidence: {res['confidence']:.2%})")
            print(f"Entities: {res['entities']}")
            print(f"Time: {res['timing']['total_ms']:.2f}ms\n")