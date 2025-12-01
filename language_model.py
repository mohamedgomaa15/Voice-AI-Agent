from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

lm_name = "meta-llama/Llama-3.2-1B"
lm_name_instruct = "meta-llama/Llama-3.2-1B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

lm_tokenizer = AutoTokenizer.from_pretrained(lm_name)
lm_model = AutoModelForCausalLM.from_pretrained(
    lm_name_instruct, 
    torch_dtype=torch.bfloat16, 
    ).to(device)

llm_pipeline = pipeline(
    "text-generation",
    model=lm_name_instruct,
    tokenizer=lm_tokenizer,
    device=0 if device=="cuda" else -1,
    max_length=256,
)

def build_prompt(user_input):
    """
    Builds an optimized prompt for intent and entity extraction.
    
    Args:
        user_input: The user's command or query
        
    Returns:
        Formatted prompt string for the model
    """
    
    system_message = """You are a specialized intent and entity extraction system. Your task is to analyze user commands and extract:
1. The primary intent (what the user wants to do)
2. All relevant entities (key information needed to fulfill the intent)

Common intents include: open app, search content, settings.
Common entity types include: person, time, product, location, date, action, object.

Rules:
- Return ONLY valid JSON, no markdown, no explanations
- Use lowercase for intent names with underscores (e.g., "book_appointment")
- If no entities are found, return an empty array
- If intent is unclear, use "unknown"

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

User: Mute the TV
Answer: {"intent": "settings", "entities": []}
"""

    prompt = (
        f"{system_message}\n"
        f"{examples}\n\n"
        f"User: {user_input}\n"
        "Answer:"
    )
    
    return prompt

def llm_generate(user_input):
    prompt = build_prompt(user_input)
    result = llm_pipeline(prompt, return_full_text=False, max_new_tokens=100)
    result_text = result[0]['generated_text']
    return result_text


if __name__ == "__main__":

    user_commands = [
        "Open YouTube and search on Ronaldo goals",
        "Search for funny cat videos",
        "Volume up",
        "Set the alarm for tomorrow"
   ]
    for user_command in user_commands:
        result_text = llm_generate(user_command)
        print("LLM Output:\n", result_text)

    # Output:
    # {"intent": "open_app", "entities": [{"type": "app_name", "value": "YouTube"}, {"type": "search_query", "value": "Ronaldo goals"}]}   
    
    # 
    # Answer: {"intent": "date", "entities": []}

 
