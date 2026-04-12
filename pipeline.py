# from speech import speech_model
from classiefier import setting_classifier_model, intent_classifier_model
from language_model import LanguageModel
from utils import evaluate_perf_latency
from utils import extract_app_name, extract_settings_action
import torch

llm = LanguageModel(device='cuda')

def agent_system_with_class_llm(command):
  
    # speech_out = speech_model(command)

    classifier_out = intent_classifier_model(command)[0]
    
    llm_out = llm.generate(command, classifier=classifier_out) 

    return {"intent": classifier_out, "entity": llm_out}
    

def agent_system_setclass_appmatch(command):

    # speech_out = speech_model(command)
    
    classifier_out = intent_classifier_model(command)[0]
  
    if classifier_out == "open_app":
        entities = extract_app_name(command)

    elif classifier_out == "settings":
        entities = setting_classifier_model(command)

    elif classifier_out == "out_of_scope":
        entities = "I can help with searching for content, opening applications, and control settings."

    elif classifier_out == "open_app_and_search":
        entities = llm.generate(command, classifier=classifier_out)
        
    elif classifier_out == "search":
        entities = llm.generate(command, classifier="search")
        
     
    return {"intent": classifier_out, "entity": entities}


    


