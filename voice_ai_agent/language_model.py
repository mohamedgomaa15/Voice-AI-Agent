from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
# from voice_ai_agent.utils import examples, intents, entities
# from llama_cpp import Llama
# from system import get_cpu_cores
import time
import re
import sys
import os
import json
import site

lm_llama = "meta-llama/Llama-3.2-1B-Instruct"
lm_qwen = "Qwen/Qwen2.5-0.5B-Instruct"
lm_qwen_large = "Qwen/Qwen2.5-1.5B-Instruct"
# llama_cpp_model = "./qwen2.5-1.5b-instruct-q8_0.gguf"

MODELS = {
    'qwen0.5': lm_qwen,
    'qwen1.5': lm_qwen_large,
    # 'llama_cpp_qwen': llama_cpp_model,
}

# 4-bit quantization config
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

_cuda_setup_done = False

def setup_llama_cuda():
    """Setup CUDA paths for llama-cpp-python (called once)"""
    global _cuda_setup_done
    
    if _cuda_setup_done or sys.platform != 'win32':
        return
    
    user_site = site.getusersitepackages()
    nvidia_path = os.path.join(user_site, 'nvidia')
    
    if os.path.exists(nvidia_path):
        for lib in ['cublas', 'cuda_runtime']:
            bin_path = os.path.join(nvidia_path, lib, 'bin')
            if os.path.exists(bin_path):
                try:
                    os.add_dll_directory(bin_path)
                except:
                    pass
    
    _cuda_setup_done = True


class LanguageModel:

    def __init__(self, model_name='qwen0.5', device='cpu', apps_json_path = 'data/clean_apps.json'):
           if model_name not in MODELS:
               model_name = 'qwen0.5'
           self.model_name = model_name
           self.model_path = MODELS[model_name]
           if device == 'cuda' and not torch.cuda.is_available():
               print('CUDA unavailable; falling back to CPU')
               device = 'cpu'
           self.device = device
           self.tokenizer = None
           self.model = None
           self.apps_list = self._load_apps(apps_json_path)

           # llama_cpp support disabled; always use HuggingFace models
           self.hf_model()
           
    def _load_apps(self, path):
        try:
            # Get project root (parent of voice_ai_agent)
            base_dir = os.path.dirname(os.path.dirname(__file__))

            full_path = os.path.join(base_dir, path)

            with open(full_path, "r", encoding="utf-8") as f:
                apps = json.load(f)

            return [app["name"] for app in apps]

        except Exception as e:
            print("Error loading apps:", e)
            return []
    # def llama_cpp_model(self):
    #         self.model = Llama(
    #                 model_path=self.model_path,
    #                 n_ctx=1024,  
    #                 n_threads=get_cpu_cores(), 
    #                 n_gpu_layers=0 if self.device=='cpu' else -1,
    #                 verbose=False,
    #             )
    #         self.tokenizer = AutoTokenizer.from_pretrained(MODELS['qwen1.5'])

    def hf_model(self): 
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            dtype = torch.float32 if self.device == 'cpu' else torch.bfloat16
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=dtype,
                device_map=self.device,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, user_input, classifier=None):
        """
        Generate entities extraction using the LLM with proper chat format.
        
        Args:
            user_input: The user's command or query
            
        Returns:
            JSON string with intent and entities
        """

        if classifier is not None:
            prompt_template = build_template_prompt(classifier, user_input, self.apps_list)
        
        
        else:
            prompt_template = build_template_full_prompt(user_input)

        # Format messages into text prompt
        formatted_prompt = self.tokenizer.apply_chat_template(
            prompt_template, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # llama_cpp branch disabled; always generate with HuggingFace models
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        result_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Post-process: try to return only the JSON object if extra text was generated
        if not result_text.startswith("{"):
            m = re.search(r"\{.*\}", result_text, re.DOTALL)
            if m:
                result_text = m.group(0)

        return result_text

    def __del__(self):
        """Proper cleanup to avoid the NoneType error"""
        # llama_cpp cleanup disabled; HuggingFace models are managed by PyTorch
        pass


def build_template_full_prompt(user_input):
    """
    Builds messages in the proper chat format.
    
    Args:
        user_input: The user's command or query
        
    Returns:
        List of message dictionaries
    """
    
    system_message = """You are a specialized intent and entity extraction system. Analyze user commands and extract the intent and entities.

Common intents: open_app, search, settings, play_media
Common entity types: app_name, search_query, content_type, genre, settings_action

Rules:
- Return ONLY valid JSON
- No explanations, no markdown, no extra text
- Use lowercase for intent names with underscores

Output format:
{"intent": "<intent_name>", "entities": [{"type": "<entity_type>", "value": "<extracted_value>"}]}"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": "Open YouTube"},
        {"role": "assistant", "content": '{"intent": "open_app", "entities": [{"type": "app_name", "value": "YouTube"}]}'},
        {"role": "user", "content": "Search for cooking videos on YouTube"},
        {"role": "assistant", "content": '{"intent": "search", "entities": [{"type": "search_query", "value": "cooking videos"}, {"type": "app_name", "value": "YouTube"}]}'},
        {"role": "user", "content": "Turn up the volume"},
        {"role": "assistant", "content": '{"intent": "settings", "entities": [{"type": "settings_action", "value": "volume_up"}]}'},
        {"role": "user", "content": user_input}
    ]
    
    return messages

def build_template_prompt(classifier_output, inp, apps_list=None):

    system_base = (
        "You are an information extraction system. "
        "You must return only a valid JSON object or the literal null. "
        "Do not return any extra text."
    )

    if classifier_output == "open_app":
        apps_text = ", ".join(apps_list) if apps_list else "unknown"

        messages = [
            {
                "role": "system",
                "content": (
                "You are an intelligent assistant that extracts application names.\n"
                "The user may speak naturally, with long sentences.\n"
                "Your job is to find the intended application from the request.\n\n"
                f"Available applications:\n{apps_text}\n\n"
                "Rules:\n"
                "- Match the closest application name\n"
                "- Understand synonyms (e.g., 'code editor' → 'Visual Studio Code')\n"
                "- Ignore extra words\n"
                "- Return ONLY JSON\n\n"
                'Format: {"app_name": "<app_name>"}'
                )
            },
            {
            "role": "user",
            "content": "hello assistant can you open vscode for me"
            },
            {
                "role": "assistant",
                "content": '{"app_name":"Visual Studio Code"}'
            },
            {
                "role": "user",
                "content": "please launch my code editor"
            },
            {
                "role": "assistant",
                "content": '{"app_name":"Visual Studio Code"}'
            },
            {
                "role": "user",
                "content": "open telegram"
            },
            {
                "role": "assistant",
                "content": '{"app_name":"Telegram"}'
            },

            {
                "role": "user",
                "content": inp
            }
            
        ]

    elif classifier_output == "search":
        messages = [
            {
                "role": "system",
                "content": system_base + " Extract the search query from the user request."
            },
            {
                "role": "user",
                "content": "find comedy movies"
            },
            {
                "role": "assistant",
                "content": '{"search_query":"comedy movies"}'
            },
            {
                "role": "user",
                "content": "I am looking for Messi goals"
            },
            {
                "role": "assistant",
                "content": '{"search_query":"Messi goals"}'
            },
            {
                "role": "user",
                "content": inp
            }
        ]

    elif classifier_output == "open_app_and_search":
        messages = [
            {
                "role": "system",
                "content": system_base + " Extract the application name and the search query from the user request."
            },
            {
                "role": "user",
                "content": "launch YouTube and find cat videos"
            },
            {
                "role": "assistant",
                "content": '{"app_name":"YouTube","search_query":"cat videos"}'
            },
            {
                "role": "user",
                "content": inp
            }
        ]

    elif classifier_output == "settings":
        messages = [
              {
                "role": "system",
                "content": (system_base + "\n\n"
                    "Extract the settings action from the user request."
                    "Valid values are: volume_up, volume_down, volume_max, unmute, mute, "
                    "brightness_up, brightness_down.\n\n"
                    "STRICT CATEGORY RULES:\n"
                    "- SOUND/AUDIO/VOLUME words (sound, audio, volume, noisy, loud, quiet, hear, mute, unmute) → ONLY volume_up or volume_down or volume_max or mute or unmute\n"
                    "- SCREEN/DISPLAY words (brightness, screen, display, picture, light, dim) → ONLY brightness_up or brightness_down\n"
                    "DIRECTION RULES:\n"
                    "- increase / up / raise / higher / more / louder / noisy / too high / max → _max or _up\n"
                    "- decrease / down / lower / reduce / less / quieter / too low → _down\n"
                    "- silence / turn off / off / shut up / without → mute\n"
                    "- unsilent / turn on / active tv / on / return / restore / back → unmute"
                        )
            },
            {
                "role": "user",
                "content": "make the sound silent"
            },
            {
                "role": "assistant",
                "content": '{"settings_action":"mute"}'
            },
            {
                "role": "user",
                "content": "I want to increase the volume"
            },
            {
                "role": "assistant",
                "content": '{"settings_action":"volume_up"}'
            },
            {
                "role": "user",
                "content": "increase the screen brightness"
            },
            {
                "role": "assistant",
                "content": '{"settings_action":"brightness_up"}'
            },
            {
                "role": "user",
                "content": inp
            }
        ]

    elif classifier_output == "out_of_scope":
        messages = [
            {
                "role": "system",
                "content": system_base + (
                    " If the request is outside the supported capabilities, return that you can help with following. "
                    "Supported capabilities: search for content, open applications, change settings."
                )
            },
            {
                "role": "user",
                "content": inp
            }
        ]

    else:
        messages = [
            {
                "role": "system",
                "content": system_base + " Always return null."
            },
            {
                "role": "user",
                "content": inp
            }
        ]

    return messages


if __name__ == "__main__":
    
    start = time.time()
    llm = LanguageModel(model_name='llama_cpp_qwen', device='cuda')
    print("model Loaded.....")
    print("example: ", "I want to watch Intersteller movie")
    print("output: ", llm.generate("I want to watch Intersteller movie", classifier='search'))
    print("Model generated......")
    print("total time: ", time.time()-start)
    print("model_path: ", llm.model_path)
    #print("model: ", True)
    # print("tokenizer: ", llm.tokenizer)
