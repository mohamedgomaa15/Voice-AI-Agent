import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from voice_ai_agent.english_stt_optimized2 import EnglishSTT
from voice_ai_agent.pipeline import agent_system_setclass_appmatch

stt = EnglishSTT(device='cpu')
audio_path = 'path/to/sample.wav'

start = time.perf_counter()
transcription = stt.transcribe_file(audio_path)
elapsed = time.perf_counter() - start

command = transcription['text'].strip()
print('STT time:', elapsed)
print('Transcription:', command)

result = agent_system_setclass_appmatch(command)
print('Intent:', result['intent'])
print('Entities:', result['entity'])