"""
Optimized Speech-to-Text (STT) System
- High accuracy English transcription
- CPU-optimized (no GPU required)
- Future-ready for Arabic support
"""

import whisper
import torch
import time
from pathlib import Path
from typing import Optional, Dict


class EnglishSTT:
    """
    High-accuracy Speech-to-Text for English
    Optimized for CPU performance
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper STT
        
        Args:
            model_size: Model size for accuracy/speed tradeoff
                - "tiny"   : ~40MB,  fastest, lowest accuracy (not recommended)
                - "base"   : ~75MB,  fast, good for CPU ⭐ RECOMMENDED for CPU
                - "small"  : ~250MB, balanced, better accuracy
                - "medium" : ~770MB, slower, high accuracy
                - "large"  : ~1.5GB, slowest, best accuracy (requires GPU)
            
            device: "cuda" or "cpu" (auto-detected if None)
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # CPU optimization warning
        if self.device == "cpu" and model_size in ["medium", "large"]:
            print("⚠️  Warning: Using large models on CPU will be very slow!")
            print("   Recommended: Use 'base' or 'small' for CPU")
        
        print(f"\n{'='*60}")
        print(f"Loading Whisper '{model_size}' model on {self.device.upper()}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        self.model = whisper.load_model(model_size, device=self.device)
        load_time = time.time() - start_time
        
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds!")
        
        # Show memory info
        if self.device == "cuda":
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM allocated: {vram:.2f} GB")
        else:
            print(f"  Running on CPU (no GPU detected)")
        
        self.model_size = model_size
        print(f"{'='*60}\n")
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en",
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5,
        verbose: bool = True,
        reference: Optional[str] = None,
        **whisper_options,
    ) -> Dict:
        """Transcribe audio file using Whisper with clean defaults.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, webm, etc.)
            language: Language code ("en" for English, "ar" for Arabic).
                      If ``None`` the model will auto-detect the language.
            temperature: Sampling temperature (0 = deterministic, more accurate)
            beam_size: Beam search size (higher = more accurate but slower)
                      - 5 is recommended for good balance
                      - Use 1 for fastest (greedy search)
            best_of: Number of candidates when sampling (higher = better)
            verbose: Print progress information
            reference: Optional reference transcript (adds WER/accuracy)
            **whisper_options: Additional Whisper settings (passed through).

        Returns:
            dict with 'text', 'segments', 'language', and metadata
        """
        # Validate file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRANSCRIBING: {Path(audio_path).name}")
            print(f"{'='*60}")
            print(f"Language: {(language or 'auto').upper()}")
            print(f"Model: {self.model_size}")
            print(f"Device: {self.device.upper()}")

        # Transcription options optimized for accuracy
        options = {
            "task": "transcribe",  # NEVER translate
            "fp16": (self.device == "cuda"),  # Use FP16 only on GPU
            "temperature": temperature,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": 1.0,  # Patience for beam search
            "condition_on_previous_text": True,  # Use context from previous segments
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        if language is not None:
            # Lock language to prevent translation when provided
            options["language"] = language

        # Add an initial prompt for English to reduce repetitions and improve quality
        if language == "en":
            options["initial_prompt"] = (
                "Okay. Hello. Thank you. Please. "
                "The following is a clear English transcription."
            )

        # Allow callers to override whisper-specific options
        options.update(whisper_options)

        # Perform transcription
        if verbose:
            print(f"\nProcessing audio...")

        result = self.model.transcribe(audio_path, **options)
        
        # Calculate metrics
        elapsed = time.time() - start_time
        audio_duration = result.get('segments', [{}])[-1].get('end', 0) if result.get('segments') else 0
        rtf = elapsed / audio_duration if audio_duration > 0 else 0  # Real-time factor
        # Attach timing info to result for downstream use
        result['transcription_time'] = elapsed
        result['audio_duration'] = audio_duration
        result['real_time_factor'] = rtf
        # compute accuracy if reference supplied
        if reference is not None:
            wer_val = self._compute_wer(reference, result.get('text', '').strip())
            result['wer'] = wer_val
            result['accuracy'] = 1.0 - wer_val if wer_val <= 1.0 else 0.0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"RESULTS")
            print(f"{'='*60}")
            print(f"Detected language: {result.get('language', 'unknown').upper()}")
            print(f"Transcription time: {elapsed:.2f} seconds")
            if audio_duration > 0:
                print(f"Audio duration: {audio_duration:.2f} seconds")
                print(f"Real-time factor: {rtf:.2f}x")
                if rtf < 1:
                    print(f"  → Faster than real-time! ✓")
                else:
                    print(f"  → Slower than real-time")
            print(f"\nTranscription:")
            print(f"{'-'*60}")
            print(f"{result['text']}")
            print(f"{'='*60}\n")
        
        return result
    
    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        reference: Optional[str] = None,
        **options
    ) -> Dict:
        """API-compatible wrapper for web UI integration.

        The web UI passes a flat set of keyword arguments (often including
        Whisper-specific options). This helper converts those into the
        parameters expected by :meth:`transcribe`.
        """

        # The web UI passes ``task`` explicitly; ignore it because this class
        # always transcribes (no translation).
        options.pop("task", None)

        # Extract the common transcription options that this class understands
        beam = options.pop("beam_size", 5)
        best = options.pop("best_of", 5)
        temp = options.pop("temperature", 0.0)
        verbose = options.pop("verbose", False)

        return self.transcribe(
            audio_path,
            language=language,
            temperature=temp,
            beam_size=beam,
            best_of=best,
            verbose=verbose,
            reference=reference,
            **options,
        )

    def transcribe_fast(self, audio_path: str) -> str:
        """Fast transcription (greedy) with minimal latency."""
        return self.transcribe(
            audio_path,
            language="en",
            temperature=0.0,
            beam_size=1,
            best_of=1,
            verbose=False,
        )["text"]

    def transcribe_accurate(self, audio_path: str, language: str = "en") -> str:
        """High-accuracy transcription (slower, larger beam search)."""
        return self.transcribe(
            audio_path,
            language=language,
            temperature=0.0,
            beam_size=10,
            best_of=10,
            verbose=False,
        )["text"]

    def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: str = "en",
        verbose: bool = True,
    ) -> Dict:
        """Transcribe audio and print segment timestamps."""
        result = self.transcribe(audio_path, language=language, verbose=verbose)

        if verbose:
            print(f"\n{'='*60}")
            print("TRANSCRIPTION WITH TIMESTAMPS")
            print(f"{'='*60}")

            for segment in result.get("segments", []):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"].strip()
                print(f"[{start:6.2f}s -> {end:6.2f}s] {text}")

            print(f"{'='*60}\n")

        return result

    def save_transcription(
        self,
        result: Dict,
        output_path: str,
        include_timestamps: bool = False,
    ) -> None:
        """Save transcription text (and optional timestamps) to a file."""
        output_file = Path(output_path)

        with open(output_file, "w", encoding="utf-8") as f:
            if include_timestamps and "segments" in result:
                f.write("Transcription with Timestamps\n")
                f.write(f"{'='*60}\n\n")

                for segment in result["segments"]:
                    start = segment["start"]
                    end = segment["end"]
                    text = segment["text"].strip()
                    f.write(f"[{start:6.2f}s -> {end:6.2f}s] {text}\n")
            else:
                f.write(result.get("text", ""))

        print(f"✓ Transcription saved to: {output_file}")

    @staticmethod
    def _compute_wer(reference: str, hypothesis: str) -> float:
        """Compute word-error-rate (WER) between reference and hypothesis."""
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()

        n = len(ref_words)
        m = len(hyp_words)

        # Dynamic programming for Levenshtein distance on word sequences
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = i
        for j in range(1, m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],    # deletion
                        dp[i][j - 1],    # insertion
                        dp[i - 1][j - 1],  # substitution
                    )

        return dp[n][m] / float(n) if n > 0 else 0.0

    def evaluate_dataset(
        self,
        dataset: "Dict[str, str]",
        verbose: bool = True,
    ) -> Dict[str, any]:
        """Evaluate the STT system against a small dataset."""
        summary = []
        total_latency = 0.0
        weighted_errors = 0.0
        total_ref_words = 0

        for audio_path, reference in dataset.items():
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            if verbose:
                print(f"\nEvaluating {audio_path}...")

            start = time.time()
            result = self.transcribe(audio_path, verbose=False)
            latency = time.time() - start
            hypothesis = result.get("text", "").strip()

            wer_val = self._compute_wer(reference, hypothesis)
            accuracy = 1.0 - wer_val if wer_val <= 1.0 else 0.0

            ref_word_count = len(reference.strip().split())
            total_ref_words += ref_word_count
            weighted_errors += wer_val * ref_word_count
            total_latency += latency

            summary.append(
                {
                    "audio": audio_path,
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": wer_val,
                    "accuracy": accuracy,
                    "latency": latency,
                }
            )

            if verbose:
                print(f"  Hypothesis : {hypothesis}")
                print(f"  Reference  : {reference}")
                print(f"  WER        : {wer_val:.2%}")
                print(f"  Accuracy   : {accuracy:.2%}")
                print(f"  Latency    : {latency:.2f}s")

        avg_accuracy = (
            1.0 - (weighted_errors / total_ref_words)
            if total_ref_words > 0
            else 0.0
        )
        avg_latency = total_latency / len(summary) if summary else 0.0
        aggregate_wer = weighted_errors / total_ref_words if total_ref_words > 0 else 0.0

        if verbose:
            print("\n" + "=" * 60)
            print("DATASET EVALUATION SUMMARY")
            print("=" * 60)
            print(f"Total files evaluated : {len(summary)}")
            print(f"Average accuracy      : {avg_accuracy:.2%}")
            print(f"Average latency       : {avg_latency:.2f}s")
            print(f"Total latency         : {total_latency:.2f}s")
            print(f"Aggregate WER         : {aggregate_wer:.2%}")
            print("=" * 60 + "\n")

        return {
            "results": summary,
            "average_accuracy": avg_accuracy,
            "average_latency": avg_latency,
            "total_latency": total_latency,
            "aggregate_wer": aggregate_wer,
        }


# =============================================================================
# FUTURE: ARABIC SUPPORT
# =============================================================================
# Uncomment and modify this section when you need Arabic support

"""
class ArabicSTT(EnglishSTT):
    '''
    Extended STT with Arabic language support
    Inherits all English functionality
    '''
    
    def transcribe_arabic(self, audio_path: str, dialect: str = "standard") -> str:
        '''
        Transcribe Arabic audio
        
        Args:
            audio_path: Path to audio file
            dialect: "standard", "egyptian", "gulf", "levantine"
        
        Returns:
            Transcribed Arabic text
        '''
        # Dialect-specific prompts
        prompts = {
            "standard": "مرحبا، كيف حالك، شكرا",
            "egyptian": "إزيك، عامل إيه، ماشي، يعني، كده",
            "gulf": "السلام عليكم، وش أخبارك، زين",
            "levantine": "كيفك، شو أخبارك، منيح"
        }
        
        # Use Arabic with dialect hint
        result = self.model.transcribe(
            audio_path,
            language="ar",
            initial_prompt=prompts.get(dialect, prompts["standard"]),
            temperature=0.0,
            beam_size=5,
            best_of=5
        )
        
        return result['text']
"""


# =============================================================================
# EXAMPLE USAGE & TESTING
# =============================================================================

def main():
    """Example usage of the STT system.

    Demonstrates the standard, fast, accurate, and timestamped transcriptions
    as well as a small dataset evaluation helper.
    """

    print("\n" + "=" * 60)
    print("ENGLISH STT SYSTEM - EXAMPLE USAGE")
    print("=" * 60)

    stt = EnglishSTT(model_size="base")

    audio_file = "LJ001-0001.wav"  # Change this to your audio file

    if not Path(audio_file).exists():
        print(f"\n⚠️  Audio file '{audio_file}' not found!")
        print("\nTo use this script:")
        print("1. Place your audio file in the same directory")
        print("2. Update the 'audio_file' variable with your filename")
        print("3. Run the script again")
        return

    print("\n" + "=" * 60)
    print("METHOD 1: STANDARD TRANSCRIPTION")
    print("=" * 60)
    result = stt.transcribe(audio_file)

    print("\n" + "=" * 60)
    print("METHOD 2: FAST TRANSCRIPTION")
    print("=" * 60)
    text_fast = stt.transcribe_fast(audio_file)
    print(f"Result: {text_fast}\n")

    print("\n" + "=" * 60)
    print("METHOD 3: HIGH ACCURACY TRANSCRIPTION")
    print("=" * 60)
    text_accurate = stt.transcribe_accurate(audio_file)
    print(f"Result: {text_accurate}\n")

    print("\n" + "=" * 60)
    print("METHOD 4: WITH TIMESTAMPS")
    print("=" * 60)
    result_timestamps = stt.transcribe_with_timestamps(audio_file)

    stt.save_transcription(result, "transcription.txt")
    stt.save_transcription(
        result_timestamps,
        "transcription_timestamps.txt",
        include_timestamps=True,
    )

    sample_dataset = {
        # "audio_1.wav": "the reference transcript for audio 1",
        # add your 10 or more samples here
    }

    if sample_dataset:
        print("\n" + "=" * 60)
        print("RUNNING DATASET EVALUATION")
        print("=" * 60)
