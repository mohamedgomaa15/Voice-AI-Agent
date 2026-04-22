import json
from jiwer import wer, cer
from time import time
from pathlib import Path
import sys
import re

# =========================
# Fix import path
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voice_ai_agent.english_stt_optimized2 import transcribe_audio_to_english


class STTAccuracyBenchmark:
    def __init__(self, dataset_path, model_fn):
        """
        dataset_path: path to JSON file
        model_fn: function(audio_path) -> transcription
        """
        self.dataset_path = Path(dataset_path).resolve()
        self.dataset = self.load_dataset()
        self.model_fn = model_fn

    def load_dataset(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base_dir = self.dataset_path.parent

        # 🔥 FIX: convert relative → absolute paths
        for sample in data:
            audio_path = Path(sample["audio"])

            if not audio_path.is_absolute():
                audio_path = base_dir / audio_path

            sample["audio"] = str(audio_path.resolve())

        return data
    

    def normalize_text(self, text):
        text = text.lower()  # lowercase
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        text = text.strip()
        return text

    def run(self):
        total_wer = 0
        total_cer = 0
        results = []

        for sample in self.dataset:
            audio_path = sample["audio"]
            ground_truth = self.normalize_text(sample["text"])

            # Debug check (optional)
            if not Path(audio_path).exists():
                print(f"[ERROR] File not found: {audio_path}")
                continue

            start = time()
            prediction = self.normalize_text(self.model_fn(audio_path))
            end = time()

            sample_wer = wer(ground_truth, prediction)
            sample_cer = cer(ground_truth, prediction)

            total_wer += sample_wer
            total_cer += sample_cer

            results.append({
                "audio": audio_path,
                "gt": ground_truth,
                "pred": prediction,
                "wer": sample_wer,
                "cer": sample_cer,
                "latency": end - start
            })

            print(f"[{Path(audio_path).name}]")
            print(f"GT   : {ground_truth}")
            print(f"PRED : {prediction}")
            print(f"WER  : {sample_wer:.3f} | CER: {sample_cer:.3f}")
            print("-" * 50)

        # Avoid division by zero
        n = len(results)
        avg_wer = total_wer / n if n > 0 else 0
        avg_cer = total_cer / n if n > 0 else 0

        print("\n===== FINAL RESULTS =====")
        print(f"Average WER: {avg_wer:.4f}")
        print(f"Average CER: {avg_cer:.4f}")

        return {
            "avg_wer": avg_wer,
            "avg_cer": avg_cer,
            "details": results
        }


# =========================
# Main
# =========================
if __name__ == "__main__":
    dataset_path = PROJECT_ROOT / "data" / "audio_samples.json"

    benchmark = STTAccuracyBenchmark(
        dataset_path=dataset_path,
        model_fn=lambda audio_path: transcribe_audio_to_english(audio_path, device="cpu")
    )

    results = benchmark.run()