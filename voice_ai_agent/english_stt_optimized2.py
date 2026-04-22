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
            batch_size=8,        # safer batch size
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


#             beam_size:   Beam search width — higher is more accurate but slower.
#                          5 is a good default; use 1 for lowest latency.
#             best_of:     Candidate count when sampling (higher = better quality).
#             verbose:     Print progress and results.
#             reference:   Optional reference transcript for WER/accuracy reporting.
#             **whisper_options: Additional Whisper keyword arguments (passed through).

#         Returns:
#             dict with 'text', 'segments', 'language', and metadata keys.
#             The 'text' value is always English (non-English output is re-transcribed
#             or flagged rather than silently returned).
#         """
#         if not Path(audio_path).exists():
#             raise FileNotFoundError(f"Audio file not found: {audio_path}")

#         start_time = time.time()

#         if verbose:
#             print(f"\n{'='*60}")
#             print(f"TRANSCRIBING: {Path(audio_path).name}")
#             print(f"{'='*60}")
#             print(f"Language: {(language or 'auto-detect').upper()}")
#             print(f"Model: {self.model_size}")
#             print(f"Device: {self.device.upper()}")

#         # -----------------------------------------------------------------
#         # Build Whisper options
#         # -----------------------------------------------------------------
#         options: Dict = {
#             "task": "transcribe",           # NEVER translate
#             "fp16": (self.device == "cuda"),
#             "temperature": temperature,
#             "beam_size": beam_size,
#             "best_of": best_of,
#             "patience": 1.0,
#             # Disable conditioning on previous text to prevent hallucination
#             # loops that repeat or drift into other languages.
#             "condition_on_previous_text": False,
#             # Tighter thresholds → fewer garbage / hallucinated segments
#             "compression_ratio_threshold": 2.0,
#             "logprob_threshold": -0.5,      # was -1.0; tighter = higher quality
#             "no_speech_threshold": 0.5,     # was 0.6; slightly more sensitive
#         }

#         # Always lock the language to English unless explicitly overridden.
#         # This is the single most important fix: without it Whisper may
#         # auto-detect a non-English language and transcribe in that language.
#         effective_language = language if language is not None else "en"
#         options["language"] = effective_language

#         # Strong English-biasing prompt
#         if effective_language == "en":
#             options["initial_prompt"] = _ENGLISH_INITIAL_PROMPT

#         # Caller can still override individual options if needed
#         options.update(whisper_options)
#         # But never let the caller accidentally remove the language lock
#         if "language" not in options or options["language"] is None:
#             options["language"] = "en"
#         # And never switch to a translation task
#         options["task"] = "transcribe"

#         # -----------------------------------------------------------------
#         # First-pass transcription
#         # -----------------------------------------------------------------
#         if verbose:
#             print("\nProcessing audio…")

#         result = self.model.transcribe(audio_path, **options)
#         detected_lang = result.get("language", "unknown")

#         # -----------------------------------------------------------------
#         # English enforcement — re-transcribe or translate if needed
#         # -----------------------------------------------------------------
#         raw_text: str = result.get("text", "").strip()

#         if _contains_non_english(raw_text) or _english_char_ratio(raw_text) < 0.85:
#             if verbose:
#                 print(
#                     f"\n⚠️  Non-English characters detected in output "
#                     f"(detected lang: {detected_lang.upper()}). "
#                     "Re-running with translation task to force English output…"
#                 )

#             # Use the "translate" task which always outputs English
#             translate_options = dict(options)
#             translate_options["task"] = "translate"
#             # Remove language lock for translate — Whisper handles it internally
#             translate_options.pop("language", None)

#             result = self.model.transcribe(audio_path, **translate_options)
#             result["_translated"] = True   # flag so callers know what happened
#         else:
#             result["_translated"] = False

#         # -----------------------------------------------------------------
#         # Metrics
#         # -----------------------------------------------------------------
#         elapsed = time.time() - start_time
#         audio_duration = (
#             result.get("segments", [{}])[-1].get("end", 0)
#             if result.get("segments")
#             else 0
#         )
#         rtf = elapsed / audio_duration if audio_duration > 0 else 0

#         result["transcription_time"] = elapsed
#         result["audio_duration"] = audio_duration
#         result["real_time_factor"] = rtf

#         if reference is not None:
#             wer_val = self._compute_wer(reference, result.get("text", "").strip())
#             result["wer"] = wer_val
#             result["accuracy"] = 1.0 - wer_val if wer_val <= 1.0 else 0.0

#         # -----------------------------------------------------------------
#         # Verbose output
#         # -----------------------------------------------------------------
#         if verbose:
#             print(f"\n{'='*60}")
#             print("RESULTS")
#             print(f"{'='*60}")
#             print(f"Detected language : {detected_lang.upper()}")
#             if result.get("_translated"):
#                 print("Output mode       : TRANSLATED → English (non-English audio detected)")
#             else:
#                 print("Output mode       : Transcribed in English")
#             print(f"Transcription time: {elapsed:.2f} s")
#             if audio_duration > 0:
#                 print(f"Audio duration    : {audio_duration:.2f} s")
#                 print(f"Real-time factor  : {rtf:.2f}×")
#                 print(f"  → {'Faster' if rtf < 1 else 'Slower'} than real-time")
#             if reference is not None:
#                 print(f"WER               : {result['wer']:.2%}")
#                 print(f"Accuracy          : {result['accuracy']:.2%}")
#             print(f"\nTranscription:")
#             print(f"{'-'*60}")
#             print(result["text"])
#             print(f"{'='*60}\n")

#         return result

#     # ------------------------------------------------------------------
#     # Convenience wrappers (same signatures as before)
#     # ------------------------------------------------------------------

#     def transcribe_file(
#         self,
#         audio_path: str,
#         language: Optional[str] = "en",   # ← was None; now defaults to English
#         task: str = "transcribe",
#         reference: Optional[str] = None,
#         **options,
#     ) -> Dict:
#         """API-compatible wrapper for web UI integration.

#         The web UI passes a flat set of keyword arguments (often including
#         Whisper-specific options).  This helper converts those into the
#         parameters expected by :meth:`transcribe`.

#         Changes vs original:
#         - ``language`` now defaults to ``"en"`` (was ``None``) so that
#           callers that omit the argument still get English-locked behaviour.
#         """
#         options.pop("task", None)   # always transcribe, never translate

#         beam = options.pop("beam_size", 5)
#         best = options.pop("best_of", 5)
#         temp = options.pop("temperature", 0.0)
#         verbose = options.pop("verbose", False)

#         return self.transcribe(
#             audio_path,
#             language=language,
#             temperature=temp,
#             beam_size=beam,
#             best_of=best,
#             verbose=verbose,
#             reference=reference,
#             **options,
#         )

#     def transcribe_fast(self, audio_path: str) -> str:
#         """Fast transcription (greedy) with minimal latency.

#         Always English — uses greedy search (beam_size=1) for speed.
#         """
#         return self.transcribe(
#             audio_path,
#             language="en",
#             temperature=0.0,
#             beam_size=1,
#             best_of=1,
#             verbose=False,
#         )["text"]

#     def transcribe_accurate(self, audio_path: str, language: str = "en") -> str:
#         """High-accuracy transcription (slower, larger beam search)."""
#         return self.transcribe(
#             audio_path,
#             language=language,
#             temperature=0.0,
#             beam_size=10,
#             best_of=10,
#             verbose=False,
#         )["text"]

#     def transcribe_with_timestamps(
#         self,
#         audio_path: str,
#         language: str = "en",
#         verbose: bool = True,
#     ) -> Dict:
#         """Transcribe audio and print segment timestamps."""
#         result = self.transcribe(audio_path, language=language, verbose=verbose)

#         if verbose:
#             print(f"\n{'='*60}")
#             print("TRANSCRIPTION WITH TIMESTAMPS")
#             print(f"{'='*60}")
#             for segment in result.get("segments", []):
#                 start = segment["start"]
#                 end = segment["end"]
#                 text = segment["text"].strip()
#                 print(f"[{start:6.2f}s -> {end:6.2f}s] {text}")
#             print(f"{'='*60}\n")

#         return result

#     def save_transcription(
#         self,
#         result: Dict,
#         output_path: str,
#         include_timestamps: bool = False,
#     ) -> None:
#         """Save transcription text (and optional timestamps) to a file."""
#         output_file = Path(output_path)

#         with open(output_file, "w", encoding="utf-8") as f:
#             if include_timestamps and "segments" in result:
#                 f.write("Transcription with Timestamps\n")
#                 f.write(f"{'='*60}\n\n")
#                 for segment in result["segments"]:
#                     start = segment["start"]
#                     end = segment["end"]
#                     text = segment["text"].strip()
#                     f.write(f"[{start:6.2f}s -> {end:6.2f}s] {text}\n")
#             else:
#                 f.write(result.get("text", ""))

#         print(f"✓ Transcription saved to: {output_file}")

#     # ------------------------------------------------------------------
#     # Evaluation helpers
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _compute_wer(reference: str, hypothesis: str) -> float:
#         """Compute Word-Error-Rate (WER) between reference and hypothesis."""
#         ref_words = reference.strip().split()
#         hyp_words = hypothesis.strip().split()

#         n = len(ref_words)
#         m = len(hyp_words)

#         dp = [[0] * (m + 1) for _ in range(n + 1)]
#         for i in range(1, n + 1):
#             dp[i][0] = i
#         for j in range(1, m + 1):
#             dp[0][j] = j

#         for i in range(1, n + 1):
#             for j in range(1, m + 1):
#                 if ref_words[i - 1] == hyp_words[j - 1]:
#                     dp[i][j] = dp[i - 1][j - 1]
#                 else:
#                     dp[i][j] = 1 + min(
#                         dp[i - 1][j],       # deletion
#                         dp[i][j - 1],       # insertion
#                         dp[i - 1][j - 1],   # substitution
#                     )

#         return dp[n][m] / float(n) if n > 0 else 0.0

#     def evaluate_dataset(
#         self,
#         dataset: "Dict[str, str]",
#         verbose: bool = True,
#     ) -> Dict[str, any]:
#         """Evaluate the STT system against a labelled dataset.

#         Args:
#             dataset: Mapping of {audio_path: reference_transcript}.
#             verbose: Print per-file results and a final summary.

#         Returns:
#             dict with 'results', 'average_accuracy', 'average_latency',
#             'total_latency', and 'aggregate_wer'.
#         """
#         summary = []
#         total_latency = 0.0
#         weighted_errors = 0.0
#         total_ref_words = 0

#         for audio_path, reference in dataset.items():
#             if not Path(audio_path).exists():
#                 raise FileNotFoundError(f"Audio file not found: {audio_path}")

#             if verbose:
#                 print(f"\nEvaluating {audio_path}…")

#             start = time.time()
#             result = self.transcribe(audio_path, verbose=False)
#             latency = time.time() - start
#             hypothesis = result.get("text", "").strip()

#             wer_val = self._compute_wer(reference, hypothesis)
#             accuracy = 1.0 - wer_val if wer_val <= 1.0 else 0.0

#             ref_word_count = len(reference.strip().split())
#             total_ref_words += ref_word_count
#             weighted_errors += wer_val * ref_word_count
#             total_latency += latency

#             summary.append(
#                 {
#                     "audio": audio_path,
#                     "reference": reference,
#                     "hypothesis": hypothesis,
#                     "wer": wer_val,
#                     "accuracy": accuracy,
#                     "latency": latency,
#                 }
#             )

#             if verbose:
#                 print(f"  Hypothesis : {hypothesis}")
#                 print(f"  Reference  : {reference}")
#                 print(f"  WER        : {wer_val:.2%}")
#                 print(f"  Accuracy   : {accuracy:.2%}")
#                 print(f"  Latency    : {latency:.2f}s")

#         avg_accuracy = (
#             1.0 - (weighted_errors / total_ref_words)
#             if total_ref_words > 0
#             else 0.0
#         )
#         avg_latency = total_latency / len(summary) if summary else 0.0
#         aggregate_wer = weighted_errors / total_ref_words if total_ref_words > 0 else 0.0

#         if verbose:
#             print("\n" + "=" * 60)
#             print("DATASET EVALUATION SUMMARY")
#             print("=" * 60)
#             print(f"Total files evaluated : {len(summary)}")
#             print(f"Average accuracy      : {avg_accuracy:.2%}")
#             print(f"Average latency       : {avg_latency:.2f}s")
#             print(f"Total latency         : {total_latency:.2f}s")
#             print(f"Aggregate WER         : {aggregate_wer:.2%}")
#             print("=" * 60 + "\n")

#         return {
#             "results": summary,
#             "average_accuracy": avg_accuracy,
#             "average_latency": avg_latency,
#             "total_latency": total_latency,
#             "aggregate_wer": aggregate_wer,
#         }


# # =============================================================================
# # FUTURE: ARABIC SUPPORT
# # =============================================================================
# # Uncomment and modify this section when you need Arabic support

# """
# class ArabicSTT(EnglishSTT):
#     '''
#     Extended STT with Arabic language support.
#     Inherits all English functionality.
#     '''

#     def transcribe_arabic(self, audio_path: str, dialect: str = "standard") -> str:
#         '''
#         Transcribe Arabic audio.

#         Args:
#             audio_path: Path to audio file.
#             dialect: "standard", "egyptian", "gulf", "levantine"

#         Returns:
#             Transcribed Arabic text.
#         '''
#         prompts = {
#             "standard": "مرحبا، كيف حالك، شكرا",
#             "egyptian": "إزيك، عامل إيه، ماشي، يعني، كده",
#             "gulf": "السلام عليكم، وش أخبارك، زين",
#             "levantine": "كيفك، شو أخبارك، منيح",
#         }

#         result = self.model.transcribe(
#             audio_path,
#             language="ar",
#             initial_prompt=prompts.get(dialect, prompts["standard"]),
#             temperature=0.0,
#             beam_size=5,
#             best_of=5,
#         )
#         return result["text"]
# """


# # =============================================================================
# # EXAMPLE USAGE & TESTING
# # =============================================================================

# def main():
#     """Example usage of the STT system.

#     Demonstrates standard, fast, accurate, and timestamped transcriptions
#     as well as a small dataset evaluation helper.
#     """
#     print("\n" + "=" * 60)
#     print("ENGLISH STT SYSTEM - EXAMPLE USAGE")
#     print("=" * 60)

#     # Using "small" model for better accuracy on CPU.
#     # Switch to "base" if speed is the priority.
#     stt = EnglishSTT(model_size="small")

#     audio_file = "LJ001-0001.wav"  # Change this to your audio file

#     if not Path(audio_file).exists():
#         print(f"\n⚠️  Audio file '{audio_file}' not found!")
#         print("\nTo use this script:")
#         print("1. Place your audio file in the same directory")
#         print("2. Update the 'audio_file' variable with your filename")
#         print("3. Run the script again")
#         return

#     print("\n" + "=" * 60)
#     print("METHOD 1: STANDARD TRANSCRIPTION")
#     print("=" * 60)
#     result = stt.transcribe(audio_file)

#     print("\n" + "=" * 60)
#     print("METHOD 2: FAST TRANSCRIPTION")
#     print("=" * 60)
#     text_fast = stt.transcribe_fast(audio_file)
#     print(f"Result: {text_fast}\n")

#     print("\n" + "=" * 60)
#     print("METHOD 3: HIGH ACCURACY TRANSCRIPTION")
#     print("=" * 60)
#     text_accurate = stt.transcribe_accurate(audio_file)
#     print(f"Result: {text_accurate}\n")

#     print("\n" + "=" * 60)
#     print("METHOD 4: WITH TIMESTAMPS")
#     print("=" * 60)
#     result_timestamps = stt.transcribe_with_timestamps(audio_file)

#     stt.save_transcription(result, "transcription.txt")
#     stt.save_transcription(
#         result_timestamps,
#         "transcription_timestamps.txt",
#         include_timestamps=True,
#     )

#     sample_dataset = {
#         # "audio_1.wav": "the reference transcript for audio 1",
#         # add your samples here
#     }

#     if sample_dataset:
#         print("\n" + "=" * 60)
#         print("RUNNING DATASET EVALUATION")
#         print("=" * 60)
#         metrics = stt.evaluate_dataset(sample_dataset)
#         print(f"Final average accuracy: {metrics['average_accuracy']:.2%}")


# if __name__ == "__main__":
#     main()

# import whisper
# import os

# _MODEL_CACHE = {}

# def _load_whisper_model(model_size="medium", device=None):
#     key = (model_size, device)
#     if key not in _MODEL_CACHE:
#         if device:
#             _MODEL_CACHE[key] = whisper.load_model(model_size, device=device)
#         else:
#             _MODEL_CACHE[key] = whisper.load_model(model_size)
#     return _MODEL_CACHE[key]


# def transcribe_audio_to_english(audio_path, model_size="medium", device=None):
#     """
#     Transcribes an audio file to English using Whisper.

#     Args:
#         audio_path (str): The exact path to your audio file.
#         model_size (str): The size of the AI model.
#         device (str|None): Optional device override, e.g. 'cpu' or 'cuda'.

#     Returns:
#         str: The transcribed English text.
#     """
#     print(f"Loading Whisper model '{model_size}' (this may take a minute on the first run)...")

#     model = _load_whisper_model(model_size, device=device)
#     print(f"Processing '{audio_path}'...")

#     result = model.transcribe(
#             audio_path, 
#             language="en", 
#             task="transcribe",
#             beam_size=7,                         # Makes the AI consider more options for better accuracy
#             condition_on_previous_text=False     # Prevents the AI from repeating itself
#         )
#     return result["text"].strip()


# class EnglishSTT:
#     def __init__(self, model_size="medium", device=None):
#         self.model_size = model_size
#         self.device = device

#     def transcribe_file(self, audio_path, language="en", task="transcribe", beam_size=7, best_of=7, temperature=0.0, **options):
#         return {"text": transcribe_audio_to_english(audio_path, model_size=self.model_size, device=self.device)}


# # ==========================================
# # Example Usage
# # ==========================================
# if __name__ == "__main__":
#     # Replace this string with the actual path to your audio file (e.g., .mp3, .wav, .m4a)
#     my_audio_file = "sample_audio.mp3" 
    
#     if os.path.exists(my_audio_file):
#         # We use the "large" model here to meet your requirement for the highest accuracy
#         transcription = transcribe_audio_to_english(my_audio_file, model_size="large")
        
#         print("\n--- Final English Transcription ---")
#         print(transcription)
#     else:
#         print(f"Error: Could not find the file '{my_audio_file}'. Please double-check the path.")