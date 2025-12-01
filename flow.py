from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
import torch.nn as nn
import torch

# get real microphone audio
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

device = "cuda" if torch.cuda.is_available() else "cpu"

fs = 16000 # Sample rate
duration = 5 # seconds

# whisper --->[small 244 M params, medium 769 M params, large 1550 M params] 

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
stt_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# transcription task and language english
stt_model.config.forced_decoder_ids = None

####### load dummy dataset and read audio files #######
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
text = ds[0]["text"]


######## record audio from microphone ########
print("Recording...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
audio = np.squeeze(audio)

input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 

# generate token ids
predicted_ids = stt_model.generate(input_features)

# decode token ids to text 
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("real text: ", text, "\npredicted text: ", transcription)
#[' Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.']



# load LLM model and tokenizer ... extracts entities


# the goal is to extract the intent and entities from the user input
# {
#   "intent": "open_app",
#   "app": "YouTube",
#   "action": "search",
#   "query": "Ronaldo goals"
# }

# NLU prediction function
def nlu_predict(user_text):
    prompt = f"""
    You are an NLU engine explain the input and extrct the entities from the user input.

    User: "{user_text}"
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = llm_model.generate(**inputs, max_new_tokens=128)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return decoded.strip()

nlu_result = nlu_predict(transcription[0])
print("NLU result: ", nlu_result)



######## Intent classification #########
 
classifier = pipeline(
    "zero-shot-classification",
    model="distilbert/distilbert-base-multilingual-cased",  # 134M parameters 
)

INTENTS = [
    "open_app",
    "search",
    "open_app_search",
    "play_content",
    "control_settings",
    "ask_info",
    "navigate_ui"
]

def classify_intent(text):
    result = classifier(
        text,
        candidate_labels=INTENTS,
        hypothesis_template="This command is about {}."
    )
    
    return {
        "intent": result["labels"][0],
        "confidence": float(result["scores"][0])
    }

print(classify_intent("open youtube"))




##########  TTS model ##########   82 million parameters, English only

#!pip install -q kokoro>=0.9.2 soundfile
#!apt-get -qq -y install espeak-ng > /dev/null 2>&1
from kokoro import KPipeline
from IPython.display import display, Audio
 

pipeline = KPipeline(lang_code='a')
text =  nlu_result
generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    display(Audio(data=audio, rate=24000, autoplay=i==0))
    sf.write(f'{i}.wav', audio, 24000)

