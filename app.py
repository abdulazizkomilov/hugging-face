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

# Set up directories and constants
AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)
MODEL_PATH = "./stt_model/medium-wav2vec-1"
SAMPLING_RATE = 16000
BATCH_SIZE = 2  # Adjust based on GPU memory

# Initialize model and processor
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
    model.to("cuda")  # Use GPU
    model.eval()
    print("Model and processor loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model or processor: {e}")

# Save file asynchronously
async def save_temp_file(file: UploadFile) -> str:
    unique_filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(AUDIO_DIR, unique_filename)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return file_path

# Preprocess audio by resampling to 16kHz
def preprocess_audio(file_path: str) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != SAMPLING_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLING_RATE)(waveform)
    return waveform

# Transcribe a batch of audio tensors
def batch_transcribe_audio(audio_tensors: List[torch.Tensor], model, processor) -> List[str]:
    # Prepare inputs for the processor
    inputs = processor(audio_tensors, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to("cuda")
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to("cuda")

    # Perform model inference
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    # Decode logits to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcriptions

@app.get("/")
async def root():
    return {"message": "ASR Model is ready for inference"}

@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    file_paths = []
    transcriptions = []

    try:
        # Save and preprocess each uploaded audio file
        for file in files:
            file_path = await save_temp_file(file)
            file_paths.append(file_path)

        # Load and preprocess audio, then create batches for transcription
        audio_tensors = [preprocess_audio(file_path) for file_path in file_paths]
        for batch_start in range(0, len(audio_tensors), BATCH_SIZE):
            batch_tensors = audio_tensors[batch_start:batch_start + BATCH_SIZE]
            batch_transcriptions = batch_transcribe_audio(batch_tensors, model, processor)
            transcriptions.extend(batch_transcriptions)

        # Compile response with file names and their transcriptions
        response = [
            {"filename": os.path.basename(file_path), "transcription": transcription}
            for file_path, transcription in zip(file_paths, transcriptions)
        ]
        return {"transcriptions": response}

    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    finally:
        # Clean up temporary files
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
