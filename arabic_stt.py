import whisper
import torch
import time
from pathlib import Path

class ArabicSTT:
    def __init__(self, model_size="small", device=None):
        """
        Initialize Whisper STT for Arabic and English
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large"
                       - tiny: fastest, lowest accuracy
                       - base: fast, decent accuracy
                       - small: balanced (RECOMMENDED)
                       - medium: slower, better accuracy
                       - large: slowest, best accuracy
            device: "cuda" or "cpu" (auto-detected if None)
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"Model loaded successfully!")
        
        # Check VRAM usage if using GPU
        if self.device == "cuda":
            vram_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"VRAM allocated: {vram_allocated:.2f} GB")
    
    def transcribe_file(self, audio_path, language=None, task="transcribe", 
                       initial_prompt=None, temperature=0.0, beam_size=5,
                       best_of=5, condition_on_previous_text=True):
        """
        Transcribe audio file to text with improved accuracy
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, webm, etc.)
            language: Language code ("ar", "en", or None for auto-detect)
                     IMPORTANT: Always specify language to avoid translation!
            task: "transcribe" (NOT "translate" - we never want translation)
            initial_prompt: Hint text to guide the model
            temperature: Sampling temperature (0 = deterministic)
            beam_size: Beam search size (higher = more accurate but slower)
            best_of: Number of candidates when sampling
            condition_on_previous_text: Use previous text as context
        
        Returns:
            dict with 'text', 'segments', and 'language'
        """
        start_time = time.time()
        
        # CRITICAL: Force task to "transcribe" - NEVER translate
        if task != "transcribe":
            print(f"⚠️ Warning: task was '{task}', forcing to 'transcribe'")
            task = "transcribe"
        
        print(f"\nTranscribing: {audio_path}")
        if language:
            print(f"Language: {language} (forced)")
        else:
            print("Language: Auto-detect")
        
        # Add Arabic hints if language is Arabic
        if language == 'ar' and not initial_prompt:
            # Common Arabic app names and commands to guide Whisper
            initial_prompt = "افتح يوتيوب نتفليكس سبوتيفاي انستقرام جوجل فيسبوك تويتر واتساب ابحث دور شغل تطبيق"
            print(f"Using Arabic prompt hints")
        
        # Transcription options for better accuracy
        options = {
            "task": task,  # Always "transcribe"
            "fp16": (self.device == "cuda"),
            "temperature": temperature,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": 1.0,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "word_timestamps": False,
        }
        
        # CRITICAL: Add language if specified
        # Without this, Whisper may translate instead of transcribe
        if language:
            options["language"] = language
            print(f"✓ Language locked to '{language}' - translation disabled")
        
        # Add initial prompt if provided
        if initial_prompt:
            options["initial_prompt"] = initial_prompt
        
        result = self.model.transcribe(audio_path, **options)
        
        elapsed = time.time() - start_time
        detected_lang = result.get('language', 'unknown')
        
        # Verify we didn't get a translation
        result_text = result['text']
        if language == 'ar':
            # Quick check: if result is all English, something went wrong
            arabic_chars = sum(1 for c in result_text if '\u0600' <= c <= '\u06FF')
            if arabic_chars == 0 and len(result_text) > 10:
                print(f"⚠️ WARNING: Expected Arabic but got: {result_text[:50]}")
                print(f"⚠️ This might be a translation! Check Whisper settings.")
        
        print(f"Detected language: {detected_lang}")
        print(f"Transcription completed in {elapsed:.2f} seconds")
        print(f"Result: {result['text'][:100]}...")
        
        return result
    
    def transcribe_auto(self, audio_path, temperature=0.0):
        """
        Auto-detect language and transcribe with best settings
        """
        return self.transcribe_file(
            audio_path, 
            language=None,  # Auto-detect
            temperature=temperature,
            beam_size=5,
            best_of=5
        )
    
    def transcribe_egyptian(self, audio_path):
        """
        Optimized transcription for Egyptian Arabic dialect
        """
        # Egyptian Arabic prompt to guide the model
        egyptian_prompt = "إزيك، عامل إيه، ماشي، يعني، كده، علشان، عايز، ازاي"
        
        return self.transcribe_file(
            audio_path, 
            language="ar",
            initial_prompt=egyptian_prompt,
            temperature=0.0,
            beam_size=5,
            best_of=5
        )
    
    def transcribe_english(self, audio_path):
        """
        Optimized transcription for English
        """
        return self.transcribe_file(
            audio_path,
            language="en",
            temperature=0.0,
            beam_size=5,
            best_of=5
        )
    
    def transcribe_with_timestamps(self, audio_path, language=None):
        """
        Transcribe with detailed timestamps for each segment
        """
        result = self.transcribe_file(audio_path, language=language)
        
        print("\n" + "="*60)
        print("TRANSCRIPTION WITH TIMESTAMPS")
        print("="*60)
        
        for segment in result['segments']:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            print(f"[{start:.2f}s -> {end:.2f}s] {text}")
        
        return result
    
    def save_transcription(self, result, output_path, include_timestamps=False):
        """
        Save transcription to text file with proper UTF-8 encoding
        
        Args:
            result: Transcription result from whisper
            output_path: Output file path
            include_timestamps: If True, include timestamps for each segment
        """
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            if include_timestamps and 'segments' in result:
                for segment in result['segments']:
                    start = segment['start']
                    end = segment['end']
                    text = segment['text'].strip()
                    f.write(f"[{start:.2f}s -> {end:.2f}s] {text}\n")
            else:
                f.write(result['text'])
        print(f"\nTranscription saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize STT
    # Use "medium" for better accuracy (slower)
    # Use "small" for balanced performance (recommended)
    # Use "base" for faster processing
    stt = ArabicSTT(model_size="small")
    
    # Example audio file
    audio_file = "001.mp3"  # Replace with your audio file
    
    if Path(audio_file).exists():
        print("\n" + "="*60)
        print("Testing different transcription methods")
        print("="*60)
        
        # Method 1: Auto-detect language (RECOMMENDED for mixed content)
        print("\n1. AUTO-DETECT LANGUAGE:")
        result = stt.transcribe_auto(audio_file)
        print(f"Detected: {result['language']}")
        print(f"Text: {result['text']}")
        
        # Method 2: Force Arabic (for Arabic-only content)
        # print("\n2. FORCE ARABIC:")
        # result = stt.transcribe_egyptian(audio_file)
        # print(f"Text: {result['text']}")
        
        # Method 3: Force English (for English-only content)
        # print("\n3. FORCE ENGLISH:")
        # result = stt.transcribe_english(audio_file)
        # print(f"Text: {result['text']}")
        
        # Save to file
        stt.save_transcription(result, "transcription.txt")
        stt.save_transcription(result, "transcription_with_timestamps.txt", include_timestamps=True)
        
    else:
        print(f"Audio file '{audio_file}' not found!")
        print("\nTo use this script:")
        print("1. Place your audio file in the same directory")
        print("2. Update the 'audio_file' variable with your filename")
        print("3. Run the script again")