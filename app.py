import os
import uuid
import torch
import logging
import aiofiles
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

app = FastAPI()

# Audio directory
AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Model setup
model_id = "./stt_model/medium-wav2vec-1"

try:
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.to("cuda")  # Move model to GPU
    model.eval()
    print("Model and processor loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model or processor: {e}")


async def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file asynchronously."""
    unique_filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(AUDIO_DIR, unique_filename)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return file_path


def preprocess_audio(file_path: str) -> torch.Tensor:
    """Load and resample audio to 16kHz."""
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    return waveform


def batch_transcribe_audio(audio_tensors: List[torch.Tensor], model, processor) -> List[str]:
    """Process and transcribe audio tensors in batches."""
    # Pad and stack audio tensors
    inputs = processor(audio_tensors, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}  # Move inputs to GPU

    # Model inference
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    # Decode predictions
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcriptions


@app.get("/")
async def root():
    """Health check."""
    return {"message": "ASR Model is ready for inference"}


@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    file_paths = []
    transcriptions = []

    try:
        # Save and preprocess audio files
        for file in files:
            file_path = await save_temp_file(file)
            file_paths.append(file_path)

        # Load and preprocess audio, then prepare batches for transcription
        audio_tensors = [preprocess_audio(file_path) for file_path in file_paths]
        batch_size = 2  # Adjust based on GPU memory
        for batch_start in range(0, len(audio_tensors), batch_size):
            batch_tensors = audio_tensors[batch_start:batch_start + batch_size]
            batch_transcriptions = batch_transcribe_audio(batch_tensors, model, processor)
            transcriptions.extend(batch_transcriptions)

        # Format the response
        response = [
            {"filename": os.path.basename(file_path), "transcription": transcription}
            for file_path, transcription in zip(file_paths, transcriptions)
        ]
        return {"transcriptions": response}

    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    finally:
        # Cleanup temporary files
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
