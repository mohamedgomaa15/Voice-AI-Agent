from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils import examples, intents, entities

lm_name_instruct = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

lm_tokenizer = AutoTokenizer.from_pretrained(lm_name_instruct)
lm_model = AutoModelForCausalLM.from_pretrained(
    lm_name_instruct, 
    dtype=torch.bfloat16, 
).to(device)

# Set pad token if not set
if lm_tokenizer.pad_token is None:
    lm_tokenizer.pad_token = lm_tokenizer.eos_token

def build_messages(user_input):
    """
    Builds messages in the proper chat format for Llama 3.2 Instruct.
    
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

def llm_generate(user_input):
    """
    Generate intent and entity extraction using the LLM with proper chat format.
    
    Args:
        user_input: The user's command or query
        
    Returns:
        JSON string with intent and entities
    """
    messages = build_messages(user_input)
    
    # Apply chat template manually
    prompt = lm_tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = lm_tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = lm_model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=lm_tokenizer.pad_token_id,
            eos_token_id=lm_tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    result_text = lm_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
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