"""
Optimized Speech-to-Text (STT) System
- Uses distil-whisper (distil-large-v3) for fast, accurate English transcription
- GPU-optimized (falls back to CPU automatically)
- Efficient memory usage with low_cpu_mem_usage=True
"""

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
from typing import Dict, Optional
import librosa
import numpy as np
import warnings

# Optional (install if needed)
# pip install noisereduce
try:
    import noisereduce as nr
    NOISE_REDUCTION_AVAILABLE = True
except:
    NOISE_REDUCTION_AVAILABLE = False


# Suppress librosa deprecation warnings
warnings.filterwarnings("ignore", message=".*__audioread_load.*", category=FutureWarning)


# Global model cache
_PIPELINE_CACHE = {}


# ==========================================
# 🔊 Audio Preprocessing (VERY IMPORTANT)
# ==========================================
def preprocess_audio(audio_path: str, target_sr: int = 16000):
    """
    Load and preprocess audio:
    - Convert to mono
    - Resample to 16kHz
    - Normalize volume
    - Optional noise reduction
    """
    try:
        # Load audio with librosa (handles various formats)
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        # Ensure audio is not empty
        if len(audio) == 0:
            raise ValueError("Audio file appears to be empty")

        # Normalize audio to prevent clipping
        audio = librosa.util.normalize(audio)

        # Optional noise reduction
        if NOISE_REDUCTION_AVAILABLE:
            audio = nr.reduce_noise(y=audio, sr=sr)

        return audio, sr

    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {audio_path}: {str(e)}")


# ==========================================
# 🧠 Pipeline Loader (IMPROVED)
# ==========================================
def _get_pipeline(device: Optional[str] = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device not in _PIPELINE_CACHE:
        print(f"\n{'='*60}")
        print(f"Loading distil-whisper model on {device}...")
        print(f"{'='*60}")

        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        # model_id = "distil-whisper/distil-large-v3"
        # model_id = "openai/whisper-medium.en"
        model_id = "distil-whisper/distil-small.en"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=torch_dtype,  # Updated: torch_dtype -> dtype
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=25,   # slightly smaller chunks for stability
            # batch_size=8,        # safer batch size
            dtype=torch_dtype,   # Updated: torch_dtype -> dtype
            device=device,
        )

        _PIPELINE_CACHE[device] = pipe
        print(f"✓ Model loaded successfully!")
        print(f"{'='*60}\n")

    return _PIPELINE_CACHE[device]


# ==========================================
# 🎯 Transcription (STABLE VERSION)
# ==========================================
def transcribe_audio_to_english(audio_path: str, device: Optional[str] = None) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"Processing '{os.path.basename(audio_path)}'...")

    # 🔊 Preprocess audio
    audio, sr = preprocess_audio(audio_path)

    pipe = _get_pipeline(device)

    # 🔒 Deterministic decoding + anti-hallucination settings
    result = pipe(
        {"array": audio, "sampling_rate": sr},
        generate_kwargs={
            "temperature": 0.0,                     # 🔥 MOST IMPORTANT
            "do_sample": False,                     # no randomness
            "num_beams": 1,                         # deterministic
            # "condition_on_previous_text": False,    # reduce hallucination
        },
        return_timestamps=False
    )

    return result["text"].strip()


# ==========================================
# 📦 Wrapper Class (IMPROVED)
# ==========================================
class EnglishSTT:

    def __init__(self, device: Optional[str] = None):
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")

    def transcribe_file(
        self,
        audio_path: str,
        language: str = "en",
        temperature: float = 0.0,
        **options
    ) -> Dict[str, str]:

        text = transcribe_audio_to_english(audio_path, device=self.device)
        return {"text": text}


# ==========================================
# ✅ Example Usage
# ==========================================
if __name__ == "__main__":
    sample_audio = "sample_audio.mp3"

    if os.path.exists(sample_audio):
        print("\n--- Testing EnglishSTT (Stable Version) ---")
        stt = EnglishSTT(device="cpu")
        result = stt.transcribe_file(sample_audio)
        print(f"Transcription: {result['text']}")
    else:
        print(f"No sample audio found at '{sample_audio}'")


