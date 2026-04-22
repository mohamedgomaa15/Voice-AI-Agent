"""
Basic tests for EnglishSTT evaluation helper.
"""

import time
from voice_ai_agent.english_stt_optimized2 import EnglishSTT


def test_evaluate_dataset_simple(monkeypatch):
    """Verify evaluation metrics are computed correctly."""
    stt = EnglishSTT(model_size="tiny")  # model won't actually be used

    # prepare a simple dataset
    dataset = {
        "sample1.wav": "hello world",
        "sample2.wav": "test one two three"
    }

    # monkeypatch transcribe method to avoid real audio processing
    def fake_transcribe(self, audio_path, **kwargs):
        # return identical text for each sample
        ref = dataset[audio_path]
        # mimic structure of real result
        return {"text": ref}

    monkeypatch.setattr(EnglishSTT, "transcribe", fake_transcribe)

    # run evaluation
    results = stt.evaluate_dataset(dataset, verbose=False)

    # each entry should have perfect accuracy and small latency
    for entry in results["results"]:
        assert entry["accuracy"] == 1.0
        assert entry["wer"] == 0.0
        assert entry["latency"] >= 0.0

    assert results["average_accuracy"] == 1.0
    assert results["aggregate_wer"] == 0.0
    assert results["total_latency"] >= 0.0

    # average latency should equal total / number of samples
    assert abs(results["average_latency"] * len(dataset) - results["total_latency"]) < 1e-6

    # ------------------------------------------------------------------
    # test transcribe_file accuracy support
    fake_text = "hello world"
    single = stt.transcribe_file("sample1.wav", reference=fake_text)
    # monkeypatch returns identical text
    assert single.get("text") == fake_text
    assert single.get("accuracy") == 1.0
    assert single.get("wer") == 0.0
