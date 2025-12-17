# multilingual_extract_entities.py
import time
import hashlib
import json
import re
from typing import Dict, List, Optional, Tuple
from transformers import pipeline
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
import arabic_reshaper
from bidi.algorithm import get_display
import unicodedata

# Optional torch import for LLM generation; if missing, LLM calls will fallback
try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

class MultilingualHybridSystem:
    def __init__(self, 
                 classifier_path: str = "mohamedgomaaa/intent-classifier-multilingual",
                 use_fast_llm: bool = True,
                 confidence_threshold: float = 0.8):
        
        # Load multilingual classifier (already trained on English+Arabic)
        self.classifier = pipeline(
            "text-classification",
            model=classifier_path,
            top_k=None,
            device=-1  # CPU for faster startup
        )
        
        # Cache for common queries (both languages)
        self.query_cache = {}
        self.cache_lock = threading.Lock()
        
        # Language detection model (fast)
        try:
            from langdetect import detect
            self.langdetect = detect
        except:
            self.langdetect = self._simple_lang_detect
        
        # Settings
        self.use_fast_llm = use_fast_llm
        self.confidence_threshold = confidence_threshold
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Multilingual prompts (English and Arabic)
        self.intent_prompts = {
            # English prompts
            "search": {
                "en": 'Query: "{user_input}"\nExtract search term.\n{{"search_query": ""}}',
                "ar": 'الاستعلام: "{user_input}"\nاستخرج مصطلح البحث.\n{{"search_query": ""}}'
            },
            "open_app": {
                "en": 'Query: "{user_input}"\nExtract app name.\n{{"app_name": ""}}',
                "ar": 'الاستعلام: "{user_input}"\nاستخرج اسم التطبيق.\n{{"app_name": ""}}'
            },
            "open_app_and_search": {
                "en": 'Query: "{user_input}"\nExtract app and search.\n{{"app_name": "", "search_query": ""}}',
                "ar": 'الاستعلام: "{user_input}"\nاستخرج التطبيق والبحث.\n{{"app_name": "", "search_query": ""}}'
            },
            "settings": {
                "en": 'Query: "{user_input}"\nExtract action.\n{{"settings_action": ""}}',
                "ar": 'الاستعلام: "{user_input}"\nاستخرج الإجراء.\n{{"settings_action": ""}}'
            },
            "play_media": {
                "en": 'Query: "{user_input}"\nExtract media.\n{{"title": "", "platform": ""}}',
                "ar": 'الاستعلام: "{user_input}"\nاستخرج الوسائط.\n{{"title": "", "platform": ""}}'
            },
            "out_of_scope": {
                "en": '{{}}',
                "ar": '{{}}'
            }
        }
        
        # Common app names in both languages
        self.app_translations = {
            "youtube": {"en": "YouTube", "ar": "يوتيوب"},
            "netflix": {"en": "Netflix", "ar": "نتفليكس"},
            "spotify": {"en": "Spotify", "ar": "سبوتيفاي"},
            "instagram": {"en": "Instagram", "ar": "إنستغرام"},
            "facebook": {"en": "Facebook", "ar": "فيسبوك"},
            "twitter": {"en": "Twitter", "ar": "تويتر"},
            "whatsapp": {"en": "WhatsApp", "ar": "واتساب"},
            "tiktok": {"en": "TikTok", "ar": "تيك توك"},
            "disney": {"en": "Disney+", "ar": "ديزني+"},
            "prime": {"en": "Amazon Prime", "ar": "أمازون برايم"},
            "hbo": {"en": "HBO Max", "ar": "إتش بي أو ماكس"},
            "apple": {"en": "Apple TV", "ar": "آبل تي في"}
        }
        
        # Precomputed common queries (both languages)
        self._init_common_queries()
        
        # Arabic normalization helper
        self._init_arabic_normalizer()
        
        # Metrics
        self.metrics = {
            "total_queries": 0,
            "english_queries": 0,
            "arabic_queries": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "rule_based_calls": 0,
            "rule_based_successes": 0,
            "total_time": 0,
            "intent_correct": 0,
            "entity_correct": 0
        }
    
    def _init_arabic_normalizer(self):
        """Initialize Arabic text normalizer"""
        # Common Arabic variations
        self.ar_variations = {
            # Aleph variations
            "إ": "ا",
            "أ": "ا",
            "آ": "ا",
            # Yeh variations
            "ى": "ي",
            "ئ": "ي",
            # Teh variations
            "ة": "ه",
            # Space variations
            "\u200f": " ",  # Right-to-left mark
            "\u200e": " ",  # Left-to-right mark
        }
    
    def _normalize_arabic(self, text: str) -> str:
        """Normalize Arabic text for consistent matching"""
        if not self._is_arabic(text):
            return text
        
        # Apply variations
        for old, new in self.ar_variations.items():
            text = text.replace(old, new)
        
        # Remove diacritics (tashkeel)
        text = ''.join(c for c in text if not unicodedata.category(c).startswith('M'))
        
        # Normalize spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _is_arabic(self, text: str) -> bool:
        """Detect if text contains Arabic characters"""
        arabic_chars = set('؀-ۿ')  # Arabic Unicode range
        return any(char in arabic_chars for char in text)
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text"""
        # Quick check for Arabic
        if self._is_arabic(text):
            return "ar"
        
        # Fallback to langdetect or simple detection
        try:
            return self.langdetect(text)
        except:
            # Simple detection: check for common English words
            common_en = ["open", "search", "find", "play", "turn", "volume", "youtube", "netflix"]
            text_lower = text.lower()
            if any(word in text_lower for word in common_en):
                return "en"
            return "en"  # Default to English
    
    def _init_common_queries(self):
        """Initialize cache with most common queries in both languages"""
        common_en = [
            ("open youtube", {"app_name": "YouTube"}),
            ("open netflix", {"app_name": "Netflix"}),
            ("search youtube", {"app_name": "YouTube", "search_query": ""}),
            ("turn up volume", {"settings_action": "volume_up"}),
            ("mute tv", {"settings_action": "mute", "parameter": "tv"}),
            ("launch spotify", {"app_name": "Spotify"}),
            ("search for movies", {"search_query": "movies"}),
            ("open instagram", {"app_name": "Instagram"}),
            ("find action movies", {"search_query": "action movies"}),
        ]
        
        common_ar = [
            ("افتح يوتيوب", {"app_name": "يوتيوب"}),
            ("افتح نتفليكس", {"app_name": "نتفليكس"}),
            ("ابحث في يوتيوب", {"app_name": "يوتيوب", "search_query": ""}),
            ("زد الصوت", {"settings_action": "volume_up"}),
            ("اكتم التلفزيون", {"settings_action": "mute", "parameter": "تلفزيون"}),
            ("افتح سبوتيفاي", {"app_name": "سبوتيفاي"}),
            ("ابحث عن أفلام", {"search_query": "أفلام"}),
            ("افتح انستغرام", {"app_name": "إنستغرام"}),
            ("ابحث عن أفلام أكشن", {"search_query": "أفلام أكشن"}),
            ("شغل فيديو", {"settings_action": "play", "parameter": "فيديو"}),
        ]
        
        # Add English queries
        for query, entities in common_en:
            key = f"en_{query.lower().strip()}"
            self.query_cache[key] = entities
        
        # Add Arabic queries (normalized)
        for query, entities in common_ar:
            normalized = self._normalize_arabic(query)
            key = f"ar_{normalized}"
            self.query_cache[key] = entities
    
    def _get_cache_key(self, text: str, intent: str, lang: str) -> str:
        """Fast hash for caching with language prefix"""
        normalized = self._normalize_arabic(text.lower()) if lang == "ar" else text.lower()
        return f"{lang}_{normalized}_{intent}"
    
    @lru_cache(maxsize=2000)
    def _cached_classify(self, text: str) -> Tuple[str, float]:
        """Cached intent classification (works for both languages)"""
        result = self.classifier(text)[0]
        if isinstance(result, list):
            result = result[0]
        return result.get("label", "out_of_scope"), result.get("score", 0.0)
    
    def classify_intent_fast(self, text: str) -> Tuple[str, float, float, str]:
        """Optimized intent classification with language detection"""
        start = time.time()
        
        # Detect language
        lang = self.detect_language(text)
        if lang == "ar":
            self.metrics["arabic_queries"] += 1
        else:
            self.metrics["english_queries"] += 1
        
        # Get intent from classifier
        intent, confidence = self._cached_classify(text)
        elapsed = time.time() - start
        
        return intent, confidence, elapsed, lang
    
    def extract_entities_multilingual(self, text: str, intent: str, lang: str) -> Dict:
        """
        Multilingual entity extraction with caching and fallbacks
        """
        # 1. Check cache first
        cache_key = self._get_cache_key(text, intent, lang)
        with self.cache_lock:
            if cache_key in self.query_cache:
                self.metrics["cache_hits"] += 1
                return self.query_cache[cache_key]
        
        # 2. Use rule-based extraction (language-specific)
        entities = self._rule_based_extraction_multilingual(text, intent, lang)
        # mark that rule-based was attempted
        self.metrics["rule_based_calls"] += 1
        if entities:
            # successful rule-based extraction
            self.metrics["rule_based_successes"] += 1

        # 3. Use LLM when rule-based failed AND intent likely needs it
        if not entities and intent in ["open_app_and_search", "play_media", "search", "open_app"]:
            # Complex or unmatched cases might need LLM
            entities = self._extract_with_fast_llm(text, intent, lang)
            self.metrics["llm_calls"] += 1
        
        # Cache the result
        if entities:
            with self.cache_lock:
                self.query_cache[cache_key] = entities
        
        return entities
    
    def _rule_based_extraction_multilingual(self, text: str, intent: str, lang: str) -> Dict:
        """Fast regex-based extraction for both English and Arabic"""
        
        if lang == "ar":
            return self._rule_based_arabic(text, intent)
        else:
            return self._rule_based_english(text, intent)
    
    def _rule_based_english(self, text: str, intent: str) -> Dict:
        """English rule-based extraction"""
        text_lower = text.lower()
        
        # Combined patterns: open <app> and search/find <query>
        combined_patterns = [
            r"(?:open|launch|start|go to|switch to) (?:the )?(?P<app>[\w\s\+\-\&]+?) (?:and|,)? (?:search for|find|look for|search) (?P<query>.+)$",
            r"(?:(?:search for|find|look for) (?P<query>.+) (?:on|in) (?P<app>\w+))$"
        ]
        for pattern in combined_patterns:
            match = re.search(pattern, text_lower)
            if match:
                gd = match.groupdict()
                app_raw = gd.get('app')
                query_raw = gd.get('query')
                app = None
                if app_raw:
                    app = self.app_translations.get(app_raw.strip().lower(), {}).get('en', app_raw.strip().title())
                q = ''
                if query_raw:
                    q = re.sub(r'^(find|search for|look for)\s+', '', query_raw.strip())
                result = {}
                if app:
                    result['app_name'] = app
                if q:
                    result['search_query'] = q
                if result:
                    return result
        
        if intent == "open_app":
            # Patterns: "open X", "launch X", "start X"
            patterns = [
                r"open (?:the )?(?:app )?(.+)$",
                r"launch (?:the )?(.+)$",
                r"start (?:the )?(.+)$",
                r"(?:go to|navigate to) (.+)$"
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    app_raw = match.group(1).strip()
                    # Translate common app names
                    app = self.app_translations.get(app_raw.lower(), {}).get("en", app_raw.title())
                    return {"app_name": app}
        
        elif intent == "search":
            # English search patterns
            patterns = [
                r"search (?:for )?(.+) on (\w+)",
                r"search (?:for )?(.+)",
                r"find (.+) on (\w+)",
                r"find (.+)",
                r"look for (.+)",
                r"look up (.+)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    result = {"search_query": match.group(1).strip()}
                    if len(match.groups()) > 1:
                        app_raw = match.group(2).strip().lower()
                        app = self.app_translations.get(app_raw, {}).get("en", app_raw.title())
                        result["app_name"] = app
                    return result
        
        elif intent == "settings":
            # English settings patterns
            if "mute" in text_lower:
                # Detect target (TV, device) if present
                m = re.search(r"mute (?:the )?(?P<target>\w+)", text_lower)
                param = m.group('target') if m else None
                return {"settings_action": "mute", **({"parameter": param} if param else {})}
            elif any(x in text_lower for x in ["turn up", "increase", "raise"]):
                if "volume" in text_lower:
                    return {"settings_action": "volume_up", "parameter": "volume"}
                elif "brightness" in text_lower:
                    return {"settings_action": "brightness_up", "parameter": "brightness"}
            elif any(x in text_lower for x in ["turn down", "decrease", "lower"]):
                if "volume" in text_lower:
                    return {"settings_action": "volume_down", "parameter": "volume"}
                elif "brightness" in text_lower:
                    return {"settings_action": "brightness_down", "parameter": "brightness"}
            elif "channel" in text_lower:
                # try to extract channel number/name
                m = re.search(r"(?:channel )(?P<ch>\w+)", text_lower)
                return {"settings_action": "change_channel", **({"parameter": m.group('ch')} if m else {})}
        
        elif intent == "play_media":
            # English media patterns
            patterns = [
                r"play (.+) on (\w+)",
                r"watch (.+) on (\w+)",
                r"play (.+)",
                r"watch (.+)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    result = {"title": match.group(1).strip()}
                    if len(match.groups()) > 1:
                        app_raw = match.group(2).strip().lower()
                        app = self.app_translations.get(app_raw, {}).get("en", app_raw.title())
                        result["platform"] = app
                    return result
        
        return {}
    
    def _rule_based_arabic(self, text: str, intent: str) -> Dict:
        """Arabic rule-based extraction"""
        # Normalize Arabic text
        text_norm = self._normalize_arabic(text)
        
        # Combined: "افتح <app> وابحث عن <query>"
        comb = re.search(r"(?:افتح|شغل|ابدأ)\s+(.+?)\s+(?:و(?:ابحث عن|ابحث|دور على|ابحث))\s+(.+)$", text_norm)
        if comb:
            app_raw = comb.group(1).strip()
            query_raw = comb.group(2).strip()
            app = self._translate_app_to_arabic(app_raw)
            return {"app_name": app, "search_query": query_raw}

        if intent == "open_app":
            # Arabic open patterns: "افتح X", "شغل X", "ابدأ X"
            patterns = [
                r"افتح (.+)$",
                r"شغل (.+)$",
                r"ابدأ (.+)$",
                r"اذهب الى (.+)$",
                r"اذهب إلى (.+)$"
            ]
            for pattern in patterns:
                match = re.search(pattern, text_norm)
                if match:
                    app_raw = match.group(1).strip()
                    # Translate common app names to Arabic
                    app = self._translate_app_to_arabic(app_raw)
                    return {"app_name": app}
        
        elif intent == "search":
            # Arabic search patterns
            patterns = [
                r"ابحث عن (.+) في (\w+)",
                r"ابحث عن (.+)",
                r"ابحث في (\w+) عن (.+)",
                r"ابحث في (\w+)",
                r"جد (.+)",
                r"دلني على (.+)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text_norm)
                if match:
                    if len(match.groups()) == 2:
                        # Pattern: "ابحث عن X في Y"
                        if "في" in pattern:
                            result = {
                                "search_query": match.group(1).strip(),
                                "app_name": self._translate_app_to_arabic(match.group(2).strip())
                            }
                        # Pattern: "ابحث في Y عن X"
                        else:
                            result = {
                                "app_name": self._translate_app_to_arabic(match.group(1).strip()),
                                "search_query": match.group(2).strip()
                            }
                        return result
                    else:
                        # Single group pattern
                        return {"search_query": match.group(1).strip()}
        
        elif intent == "settings":
            # Arabic settings patterns
            if "اكتم" in text_norm or "صامت" in text_norm:
                return {"settings_action": "mute"}
            elif any(x in text_norm for x in ["زد", "زود", "ارفع", "اعلى"]):
                if any(x in text_norm for x in ["صوت", "صوتي"]):
                    return {"settings_action": "volume_up"}
                elif "سطوع" in text_norm:
                    return {"settings_action": "brightness_up"}
            elif any(x in text_norm for x in ["خفف", "اقل", "انقص", "اخفض"]):
                if any(x in text_norm for x in ["صوت", "صوتي"]):
                    return {"settings_action": "volume_down"}
                elif "سطوع" in text_norm:
                    return {"settings_action": "brightness_down"}
            elif "قناة" in text_norm:
                return {"settings_action": "change_channel"}
        
        elif intent == "play_media":
            # Arabic media patterns
            patterns = [
                r"شغل (.+) في (\w+)",
                r"شغل (.+) على (\w+)",
                r"شاهد (.+) في (\w+)",
                r"شاهد (.+)",
                r"عرض (.+)"
            ]
            for pattern in patterns:
                match = re.search(pattern, text_norm)
                if match:
                    if len(match.groups()) == 2:
                        return {
                            "title": match.group(1).strip(),
                            "platform": self._translate_app_to_arabic(match.group(2).strip())
                        }
                    else:
                        return {"title": match.group(1).strip()}
        
        return {}
    
    def _translate_app_to_arabic(self, app_name: str) -> str:
        """Translate common app names to Arabic"""
        app_lower = app_name.lower()
        
        # Check direct translations
        for eng_app, translations in self.app_translations.items():
            if eng_app in app_lower:
                return translations.get("ar", app_name)
        
        # If not found, return original with Arabic spelling guess
        # Simple transliteration for common apps
        translit_map = {
            "youtube": "يوتيوب",
            "netflix": "نتفليكس",
            "spotify": "سبوتيفاي",
            "instagram": "إنستغرام",
            "facebook": "فيسبوك",
            "twitter": "تويتر",
            "whatsapp": "واتساب",
            "tiktok": "تيك توك",
            "telegram": "تيليجرام",
            "snapchat": "سناب شات"
        }
        
        return translit_map.get(app_lower, app_name)
    
    def _extract_with_fast_llm(self, text: str, intent: str, lang: str) -> Dict:
        """
        Use a fast LLM for complex multilingual extraction
        """
        # Get appropriate prompt for language
        if intent in self.intent_prompts:
            prompt_template = self.intent_prompts[intent].get(lang, self.intent_prompts[intent]["en"])
        else:
            prompt_template = '{{}}'
        
        prompt = prompt_template.format(user_input=text)
        
        try:
            # Use a fast multilingual model
            result = self._call_fast_multilingual_llm(prompt, lang)
            return self._parse_json_response(result)
        except:
            return {}
    
    def _call_fast_multilingual_llm(self, prompt: str, lang: str) -> str:
        """
        Call a fast multilingual LLM (local or API)
        Options:
        1. Google's T5 multilingual
        2. mBART
        3. XLM-RoBERTa for classification
        """
        try:
            # Option 1: Use Google's multilingual T5 (very fast)
            from transformers import T5Tokenizer, T5ForConditionalGeneration
            
            model_name = "google/mt5-small"  # 300M params, supports 101 languages
            
            # Load once and cache
            if not hasattr(self, 'mt5_tokenizer'):
                self.mt5_tokenizer = T5Tokenizer.from_pretrained(model_name)
                # If torch is not available, avoid attempting to map to GPU
                if _HAS_TORCH:
                    device_map = "auto" if torch.cuda.is_available() else "cpu"
                else:
                    device_map = "cpu"
                self.mt5_model = T5ForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=device_map
                )
            
            # Tokenize and generate
            inputs = self.mt5_tokenizer(prompt, return_tensors="pt", max_length=100, truncation=True)
            
            with torch.no_grad():
                outputs = self.mt5_model.generate(
                    inputs.input_ids,
                    max_length=50,
                    num_beams=1,  # Fast decoding
                    do_sample=False
                )
            
            result = self.mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
            
        except Exception as e:
            # Fallback to tiny local model or return empty
            print(f"LLM fallback: {e}")
            return "{}"
    
    def process_query_optimized(self, text: str) -> Dict:
        """Main optimized processing pipeline for any language"""
        start_time = time.time()
        self.metrics["total_queries"] += 1

        # Quick heuristic: common short greetings are out_of_scope
        t_lower = text.strip().lower()
        greetings = {"hi", "hello", "hey", "مرحبا", "أهلا", "اهلا", "سلام"}
        if t_lower in greetings or re.fullmatch(r"(hi|hello|hey|hey there|h[ey]{2})[!.]?", t_lower):
            return {
                "text": text,
                "language": "en",
                "intent": "out_of_scope",
                "confidence": 1.0,
                "entities": {},
                "timing": {"classifier_ms": 0.0, "total_ms": (time.time() - start_time) * 1000}
            }
        
        # 1. Fast intent classification with language detection
        intent, confidence, classifier_time, lang = self.classify_intent_fast(text)
        
        # 2. Decide extraction strategy
        entities = {}
        need_entities = intent in ["search", "open_app", "open_app_and_search", "settings", "play_media"]
        
        if need_entities:
            # Try rule-based first for speed
            entities = self._rule_based_extraction_multilingual(text, intent, lang)
            # If rule-based failed, try the multi-strategy extractor (which may call LLM)
            if not entities:
                entities = self.extract_entities_multilingual(text, intent, lang)
        
        total_time = time.time() - start_time
        self.metrics["total_time"] += total_time
        
        return {
            "text": text,
            "language": lang,
            "intent": intent,
            "confidence": confidence,
            "entities": entities,
            "timing": {
                "classifier_ms": classifier_time * 1000,
                "total_ms": total_time * 1000
            }
        }
    
    def process_batch(self, texts: List[str]) -> List[Dict]:
        """Process multiple queries in parallel"""
        with self.executor as executor:
            results = list(executor.map(self.process_query_optimized, texts))
        return results
    
    def get_metrics(self) -> Dict:
        """Get comprehensive performance metrics"""
        if self.metrics["total_queries"] == 0:
            return {}
        
        avg_time = self.metrics["total_time"] / self.metrics["total_queries"]
        
        # Compute clearer rates
        rule_based_success_rate = self.metrics["rule_based_successes"] / self.metrics["total_queries"] if self.metrics["total_queries"] > 0 else 0

        return {
            "total_queries": self.metrics["total_queries"],
            "english_queries": self.metrics["english_queries"],
            "arabic_queries": self.metrics["arabic_queries"],
            "language_distribution": {
                "english": self.metrics["english_queries"] / self.metrics["total_queries"],
                "arabic": self.metrics["arabic_queries"] / self.metrics["total_queries"]
            },
            "cache_hit_rate": self.metrics["cache_hits"] / self.metrics["total_queries"],
            "avg_processing_time_ms": avg_time * 1000,
            "llm_call_rate": self.metrics["llm_calls"] / self.metrics["total_queries"],
            "rule_based_success_rate": rule_based_success_rate,
            "intent_accuracy": self.metrics["intent_correct"] / self.metrics["total_queries"] if self.metrics["total_queries"] > 0 else 0,
            "entity_accuracy": self.metrics["entity_correct"] / self.metrics["total_queries"] if self.metrics["total_queries"] > 0 else 0
        }

# =================== EVALUATION SCRIPT ===================

def evaluate_multilingual_system():
    """Evaluate the multilingual system on test data"""
    import pandas as pd
    from utils import examples, intents as true_intents, entities as true_entities
    
    system = MultilingualHybridSystem()
    
    print("Evaluating Multilingual Hybrid System...")
    print("=" * 70)
    
    all_results = []
    total_start = time.time()
    
    # Process all examples
    for query, true_intent, true_ent in zip(examples, true_intents, true_entities):
        result = system.process_query_optimized(query)
        
        # Check intent accuracy
        if result['intent'] == true_intent:
            system.metrics["intent_correct"] += 1
        
        # Check entity accuracy
        predicted_entities = result['entities']
        entity_match = False
        
        if true_ent and predicted_entities:
            # Simple matching logic - adapt as needed
            true_values = [e.get('value', '') for e in true_ent]
            pred_values = list(predicted_entities.values())
            
            # Check if any true entity value is in predicted values
            for tv in true_values:
                if any(str(tv).lower() in str(pv).lower() for pv in pred_values):
                    entity_match = True
                    break
        
        if entity_match or (not true_ent and not predicted_entities):
            system.metrics["entity_correct"] += 1
        
        all_results.append({
            "query": query,
            "true_intent": true_intent,
            "predicted_intent": result['intent'],
            "true_entities": true_ent,
            "predicted_entities": predicted_entities,
            "language": result['language'],
            "confidence": result['confidence'],
            "time_ms": result['timing']['total_ms']
        })
        
        print(f"\nQuery: {query}")
        print(f"Language: {result['language']}")
        print(f"Intent: {result['intent']} (true: {true_intent})")
        print(f"Entities: {predicted_entities}")
        print(f"Time: {result['timing']['total_ms']:.1f}ms")
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    metrics = system.get_metrics()
    
    print(f"\nPerformance:")
    print(f"  Total queries: {metrics['total_queries']}")
    print(f"  English queries: {metrics['english_queries']}")
    print(f"  Arabic queries: {metrics['arabic_queries']}")
    print(f"  Intent accuracy: {metrics.get('intent_accuracy', 0):.1%}")
    print(f"  Entity accuracy: {metrics.get('entity_accuracy', 0):.1%}")
    print(f"  Average time per query: {metrics['avg_processing_time_ms']:.1f}ms")
    print(f"  Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    print(f"  LLM call rate: {metrics['llm_call_rate']:.1%}")
    print(f"  Total evaluation time: {total_time:.2f}s")
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("multilingual_results.csv", index=False, encoding='utf-8')
    
    return system, results_df

# =================== INTERACTIVE TEST ===================

def interactive_test():
    """Interactive test of the multilingual system"""
    system = MultilingualHybridSystem()
    
    print("Multilingual Hybrid System - Interactive Mode")
    print("Supports English and Arabic queries")
    print("Type 'exit' to quit\n")
    
    test_queries = [
        "افتح يوتيوب",  # Arabic: Open YouTube
        "Open Netflix",  # English
        "ابحث عن أفلام أكشن",  # Arabic: Search for action movies
        "Search for cooking videos on YouTube",  # English
        "زد الصوت",  # Arabic: Turn up volume
        "Mute the TV",  # English
        "افتح انستغرام وابحث عن فيديوهات",  # Arabic: Open Instagram and search for videos
        "Launch Spotify and find jazz music",  # English
    ]
    
    print("Example queries:")
    for q in test_queries:
        print(f"  - {q}")
    print()
    
    while True:
        try:
            query = input("Enter query (or 'test' for examples): ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                break
            
            if query.lower() == 'test':
                for q in test_queries:
                    result = system.process_query_optimized(q)
                    print(f"\nQuery: {q}")
                    print(f"Language: {result['language']}")
                    print(f"Intent: {result['intent']} ({result['confidence']:.1%})")
                    print(f"Entities: {result['entities']}")
                    print(f"Time: {result['timing']['total_ms']:.1f}ms")
                continue
            
            # Process user query
            result = system.process_query_optimized(query)
            
            print(f"\nResults:")
            print(f"  Language: {result['language']}")
            print(f"  Intent: {result['intent']} ({result['confidence']:.1%})")
            print(f"  Entities: {result['entities']}")
            print(f"  Processing time: {result['timing']['total_ms']:.1f}ms\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Show metrics
    metrics = system.get_metrics()
    print("\n" + "=" * 50)
    print("Session Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            if key in ["intent_accuracy", "entity_accuracy", "cache_hit_rate", "llm_call_rate"]:
                print(f"  {key}: {value:.1%}")
            elif "time" in key:
                print(f"  {key}: {value:.1f}ms")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

# =================== MAIN ENTRY POINT ===================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multilingual Hybrid Intent System")
    parser.add_argument("--eval", action="store_true", help="Run evaluation on test data")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    parser.add_argument("--query", type=str, help="Process a single query")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    
    args = parser.parse_args()
    
    if args.eval:
        evaluate_multilingual_system()
    elif args.query:
        system = MultilingualHybridSystem()
        result = system.process_query_optimized(args.query)
        print(f"\nQuery: {args.query}")
        print(f"Language: {result['language']}")
        print(f"Intent: {result['intent']} ({result['confidence']:.1%})")
        print(f"Entities: {result['entities']}")
        print(f"Time: {result['timing']['total_ms']:.1f}ms")
    elif args.benchmark:
        # Run benchmark with mixed queries
        test_queries = [
            "افتح يوتيوب",
            "Open Netflix",
            "ابحث عن أفلام",
            "Search for music",
            "زد الصوت",
            "Mute TV",
            "افتح تطبيق",
            "Launch app"
        ]
        system = MultilingualHybridSystem()
        
        print("Running benchmark...")
        start = time.time()
        results = system.process_batch(test_queries)
        total_time = time.time() - start
        
        print(f"\nProcessed {len(test_queries)} queries in {total_time:.2f}s")
        print(f"Average per query: {(total_time/len(test_queries)*1000):.1f}ms")
        
        metrics = system.get_metrics()
        print(f"\nCache hit rate: {metrics['cache_hit_rate']:.1%}")
        print(f"LLM call rate: {metrics['llm_call_rate']:.1%}")
    else:
        interactive_test()