import time
import os
from statistics import mean
import sys
from pathlib import Path

# Add parent directory to path for package imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import your STT
from voice_ai_agent.english_stt_optimized2 import EnglishSTT

# ==========================================
# ⚙️ Config
# ==========================================
AUDIO_FOLDER = "./data/audio_samples"   # folder with your 10 audio files
N_RUNS = 4                       # number of runs per file


# ==========================================
# 🚀 Benchmark Function
# ==========================================
def benchmark_stt():
    stt = EnglishSTT(device="cpu")  # or "cuda:0" if you have GPU

    audio_files = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith((".wav", ".mp3", ".webm"))]

    if not audio_files:
        print("❌ No audio files found!")
        return

    overall_times = []

    print("\n" + "="*60)
    print("🎤 STT LATENCY BENCHMARK STARTED")
    print("="*60)

    for audio_file in audio_files:
        file_path = os.path.join(AUDIO_FOLDER, audio_file)

        print(f"\n📂 Testing: {audio_file}")

        run_times = []

        for i in range(N_RUNS):
            print(f"   ▶ Run {i+1}...")

            start_time = time.perf_counter()

            result = stt.transcribe_file(file_path)

            end_time = time.perf_counter()

            latency = end_time - start_time
            run_times.append(latency)

            print(f"      ⏱ Time: {latency:.3f} sec")
            print(f"      📝 Text: {result['text']}")

        avg_time = mean(run_times)
        overall_times.extend(run_times)

        print(f"   ✅ Average for {audio_file}: {avg_time:.3f} sec")

    # ==========================================
    # 📊 Final Results
    # ==========================================
    overall_avg = mean(overall_times)

    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"Total runs: {len(overall_times)}")
    print(f"Overall average latency: {overall_avg:.3f} sec")
    print("="*60)


# ==========================================
# ▶ Run
# ==========================================
if __name__ == "__main__":
    benchmark_stt()