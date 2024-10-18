import os
import uuid
import torch
import logging
import aiofiles
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Set device and dtype efficiently
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model ID
MODEL_ID = "openai/whisper-tiny"

# Load model and processor once at startup
logging.info("Loading model and processor...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Initialize the ASR pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Directory to store temporary audio files
AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Model is ready for inference"}


async def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file with a unique name asynchronously."""
    unique_filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(AUDIO_DIR, unique_filename)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return file_path


def preprocess_audio(file_path: str) -> str:
    """Resample audio to 16kHz if necessary."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

        # Save resampled audio
        resampled_path = f"{file_path}_resampled.wav"
        torchaudio.save(resampled_path, waveform, 16000)
        return resampled_path

    return file_path


@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    """Transcribe multiple audio files in parallel using batch processing."""
    file_paths = []

    try:
        # Save and preprocess all audio files
        for file in files:
            file_path = await save_temp_file(file)
            processed_path = preprocess_audio(file_path)
            file_paths.append(processed_path)

        # Perform transcription in batch
        results = asr_pipeline(file_paths, batch_size=len(file_paths), generate_kwargs={"language": "en"})

        # Format the response
        transcriptions = [
            {"filename": file.filename, "transcription": result["text"]}
            for file, result in zip(files, results)
        ]

        return {"transcriptions": transcriptions}

    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        # Cleanup temporary audio files
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
