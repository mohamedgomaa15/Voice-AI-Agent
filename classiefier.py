from transformers import pipeline
import json


INTENT_MODEL_PATH = "mohamedgomaaa/intent-classifier-multilingual-v2"
SETTING_MODEL_PATH = "mohamedgomaaa/setting-classifier"

setting_model = pipeline(
            "text-classification",
            model=SETTING_MODEL_PATH,
            tokenizer="distilbert-base-multilingual-cased",
            top_k=None
        )

intent_model = pipeline(
    "text-classification",
    model=INTENT_MODEL_PATH,
    top_k=None
    )
    

def intent_classifier_model(texts):
    outs = intent_model(texts)
    pred_label = [out[0]['label'] for out in outs]
    return pred_label

def setting_classifier_model(texts):
    outs = setting_model(texts)
    pred_label = [out[0]['label'] for out in outs]
    entity = None
    if pred_label[0] == "out_of_scope":
        entity = json.dumps({"settings_action": "unknown"}, separators=(",", ":"))
    else :
        entity = json.dumps({"settings_action": pred_label[0]}, separators=(",", ":"))
    return entity

# def cal_accuracy(model, texts, labels):
#     pred_label = intent_classifier_model(texts)
#     correct = sum(p == t for p, t in zip(pred_label, labels))
#     accuracy = correct / len(labels)
#     return accuracy