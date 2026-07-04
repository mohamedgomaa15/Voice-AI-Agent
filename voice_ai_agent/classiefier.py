from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
import torch
from transformers import pipeline
import json


INTENT_MODEL_PATH = "mohamedgomaaa/intent-classifier-multilingual-v2"
SETTING_MODEL_PATH = "mohamedgomaaa/setting-classifier"
INTENT_ONNX_MODEL_PATH = "models/intent_onnx"
SETTING_ONNX_MODEL_PATH = "models/setting_onnx"


class ONNXIntentClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(INTENT_ONNX_MODEL_PATH)
        self.model = ORTModelForSequenceClassification.from_pretrained(INTENT_ONNX_MODEL_PATH, file_name="model_int8.onnx")
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256, 
        )
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_id = probs.argmax().item()
        confidence = probs[0][predicted_id].item()
        return self.id2label[predicted_id], confidence


class ONNXSettingClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(SETTING_ONNX_MODEL_PATH)
        self.model = ORTModelForSequenceClassification.from_pretrained(SETTING_ONNX_MODEL_PATH, file_name="model_int8.onnx")
        self.id2label = self.model.config.id2label

    def predict(self, text: str) -> tuple[str, float]:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256, 
        )
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predicted_id = probs.argmax().item()
        confidence = probs[0][predicted_id].item()
        pred_label = self.id2label[predicted_id]
        if pred_label == "out_of_scope":
            entity = json.dumps({"settings_action": "unknown"}, separators=(",", ":"))
        else :
            entity = json.dumps({"settings_action": pred_label}, separators=(",", ":"))
        return entity, confidence

class IntentClassifier:
    def __init__(self):
        self.model = pipeline(
                    "text-classification",
                    model=INTENT_MODEL_PATH,
                    top_k=None
                )
    def predict(self, text):
        outs = self.model(text)
        pred_label = outs[0][0]['label']
        confidence = outs[0][0]['score']
        return pred_label, confidence  


class SettingClassifier:
    def __init__(self):
        self.model = pipeline(
                    "text-classification",
                    model=SETTING_MODEL_PATH,
                    tokenizer="distilbert-base-multilingual-cased",
                    top_k=None
                )
        
    def predict(self, text):
        outs = self.model(text)
        pred_label = outs[0][0]['label']
        confidence = outs[0][0]['score']
        entity = None
        if pred_label == "out_of_scope":
            entity = json.dumps({"settings_action": "unknown"}, separators=(",", ":"))
        else :
            entity = json.dumps({"settings_action": pred_label}, separators=(",", ":"))
        return entity, confidence


if __name__ == "__main__":
    intent_clf = IntentClassifier()
    setting_clf = SettingClassifier()

    # Test intent classifier
    intent_label, intent_confidence = intent_clf.predict("turn up the volume")
    print(f"Intent: {intent_label}({intent_confidence:.2%})")

    # # Test setting classifier
    setting_entity, setting_confidence = setting_clf.predict("turn up the volume")
    print(f"Setting Entity: {setting_entity} ({setting_confidence:.2%})")