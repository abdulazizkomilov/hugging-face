import os
import uuid
import torch
import logging
import aiofiles
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from transformers import pipeline

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Set device and dtype
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Custom model path
MODEL_PATH = "./stt_model/medium-wav2vec-1"  # O'z modelingiz papkasiga yo'lni ko'rsating

# Load model and processor at startup
logging.info("Loading custom STT model...")
model = torch.load(os.path.join(MODEL_PATH, "model.pth"), map_location=device)
model.to(device)
model.eval()

# Model token va feature extractor agar kerak bo'lsa
# Bular sizda mavjud STT model turiga bog'liq, Whisper model emasligi uchun tokenizer va feature extractor farq qilishi mumkin.
# Agar kerak bo'lsa, quyidagi kabi kod orqali yuklab oling:
# tokenizer = <YOUR TOKENIZER>
# feature_extractor = <YOUR FEATURE EXTRACTOR>

# Initialize ASR pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    # tokenizer=tokenizer,  # Agar kerak bo'lsa
    # feature_extractor=feature_extractor,  # Agar kerak bo'lsa
    torch_dtype=torch_dtype,
    device=device,
)

# Audio directory
AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

# FastAPI app
app = FastAPI()


@app.get("/")
async def root():
    """Health check."""
    return {"message": "Custom STT Model is ready for inference"}


async def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file asynchronously."""
    unique_filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(AUDIO_DIR, unique_filename)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return file_path


def preprocess_audio(file_path: str) -> str:
    """Ensure audio is resampled to 16kHz."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        logging.info(f"Resampling {file_path} from {sample_rate}Hz to 16000Hz.")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

        resampled_path = f"{file_path}_resampled.wav"
        torchaudio.save(resampled_path, waveform, 16000)
        return resampled_path

    return file_path


def split_long_audio(file_path: str, max_duration: int = 30) -> List[str]:
    """Split audio into smaller chunks if it exceeds max_duration."""
    waveform, sample_rate = torchaudio.load(file_path)
    total_duration = waveform.shape[1] / sample_rate

    if total_duration <= max_duration:
        return [file_path]

    logging.info(f"Splitting {file_path} into {max_duration}-second chunks.")
    chunks = []
    for i in range(0, int(total_duration), max_duration):
        chunk = waveform[:, i * sample_rate:(i + max_duration) * sample_rate]
        chunk_path = f"{file_path}_chunk_{i}.wav"
        torchaudio.save(chunk_path, chunk, sample_rate)
        chunks.append(chunk_path)

    return chunks


@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    """Transcribe multiple audio files with proper batching."""
    file_paths = []

    try:
        # Save and preprocess audio files
        for file in files:
            file_path = await save_temp_file(file)
            processed_path = preprocess_audio(file_path)
            file_paths.extend(split_long_audio(processed_path))

        # Perform transcription in batches
        results = []
        for batch_start in range(0, len(file_paths), 2):  # Batch size = 2
            batch_files = file_paths[batch_start:batch_start + 2]
            batch_results = asr_pipeline(batch_files, generate_kwargs={"language": "en"})
            results.extend(batch_results)

        # Format the response
        transcriptions = [
            {"filename": os.path.basename(file), "transcription": result["text"]}
            for file, result in zip(file_paths, results)
        ]

        return {"transcriptions": transcriptions}

    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        # Cleanup temporary files
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
