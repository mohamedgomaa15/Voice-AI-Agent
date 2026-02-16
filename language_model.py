from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import examples, intents, entities

lm_llama = "meta-llama/Llama-3.2-1B-Instruct"
lm_qwen = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

lm_tokenizer = AutoTokenizer.from_pretrained(lm_qwen)
lm_model = AutoModelForCausalLM.from_pretrained(
    lm_qwen, 
    dtype=torch.bfloat16, 
).to(device)

# Set pad token if not set
if lm_tokenizer.pad_token is None:
    lm_tokenizer.pad_token = lm_tokenizer.eos_token

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

def build_template_prompt(classifier_output, inp):

    system_base = (
        "You are an information extraction system. "
        "You must return only a valid JSON object or the literal null. "
        "Do not return any extra text."
    )

    if classifier_output == "open_app":
        messages = [
            {
                "role": "system",
                "content": system_base + " Extract the application name from the user request."
            },
            {
                "role": "user",
                "content": "open Netflix"
            },
            {
                "role": "assistant",
                "content": '{"app_name":"Netflix"}'
            },
             {
                "role": "user",
                "content": "launch YouTube"
            },
            {
                "role": "assistant",
                "content": '{"app_name":"YouTube"}'
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
                "content": system_base + (
                    " Extract the settings action from the user request. "
                    "Valid values are: volume_up, volume_down, mute, unmute, "
                    "brightness_up, brightness_down, subtitles_on, subtitles_off."
                )
            },
            {
                "role": "user",
                "content": "increase volume"
            },
            {
                "role": "assistant",
                "content": '{"settings_action":"volume_up"}'
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


        
        
def llm_generate(user_input, classifier=None):
    """
    Generate entities extraction using the LLM with proper chat format.
    
    Args:
        user_input: The user's command or query
        
    Returns:
        JSON string with intent and entities
    """
    prompt = ""
    if classifier is not None:
        prompt_template = build_template_prompt(classifier, user_input)
       
       
    else:
        prompt_template = build_template_full_prompt(user_input)
        
    # Apply chat template manually
    prompt_template = lm_tokenizer.apply_chat_template(
        prompt_template, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = lm_tokenizer(prompt_template, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = lm_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=lm_tokenizer.pad_token_id,
            eos_token_id=lm_tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    result_text = lm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Post-process: try to return only the JSON object if extra text was generated
    import re
    if not result_text.startswith("{"):
        m = re.search(r"\{.*\}", result_text, re.DOTALL)
        if m:
            result_text = m.group(0)

    return result_text


if __name__ == "__main__":
    
    # Test the LLM-based intent and entity extraction
    print("Testing LLM Intent and Entity Extraction:\n")
    
    for user_input, true_intent, true_entity in zip(examples[:2], intents[:2], entities[:2]):
        llm_output = llm_generate(user_input)
        print(f"User Input: {user_input}")
        print(f"LLM Output: {llm_output}")
        print(f"True Intent: {true_intent}, True Entities: {true_entity}")
        print("-" * 70)