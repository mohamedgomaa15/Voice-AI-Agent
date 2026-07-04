from faster_whisper import WhisperModel
import torch
import time 
import numpy as np
import torch
import warnings

# Optional (install if needed)
# pip install noisereduce
# try:
#     import noisereduce as nr
#     NOISE_REDUCTION_AVAILABLE = True
# except:
#     NOISE_REDUCTION_AVAILABLE = False


# Suppress librosa deprecation warnings
warnings.filterwarnings("ignore", message=".*__audioread_load.*", category=FutureWarning)




class STTModel:
    def __init__(self, model_path='small'):
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.compute_type = 'float16'
            print("CUDA is set (fp16)")
        else:
            self.device = 'cpu'
            self.compute_type = 'int8'  # fastest option on CPU for faster-whisper
            print("CPU is set (int8)")

        self.model = WhisperModel(model_path, device=self.device, compute_type=self.compute_type)

    def transcript(self, audio):

        if isinstance(audio, dict):
            audio = audio["array"]

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        segments, _ = self.model.transcribe(
            audio,
            language='en',
            task='transcribe'
        )
        transcription = " ".join([s.text.strip() for s in segments])
        return transcription
    


if __name__ == '__main__':
    stt_model = STTModel()
    from datasets import load_dataset
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = ds[0]["audio"]
    text = ds[0]['text']
    print("dataset Loaded!!!")
    
    start = time.time()
    transcription = stt_model.transcript(sample)

    print("Wasted Time: ", time.time()-start)
    print("Text audio: ", text)
    print("Generated Text audio: ", transcription)

    start = time.time()
    transcription = stt_model.transcript(sample)

    print("Wasted Time: ", time.time()-start)
    print("Text audio: ", text)
    print("Generated Text audio: ", transcription)