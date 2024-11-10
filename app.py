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

AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

model_id = "./stt_model/medium-wav2vec-1"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    model.eval()
    model.gradient_checkpointing_enable()  # Enable gradient checkpointing
    print("Model and processor loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model or processor: {e}")


def get_asr_result(audio_path, model, processor, sr=16000, chunk_duration=10):
    """Perform ASR on audio in chunks to avoid memory overload."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)(waveform)
    audio = waveform.squeeze().numpy()

    chunk_samples = sr * chunk_duration
    transcriptions = []

    for start in range(0, len(audio), chunk_samples):
        audio_chunk = audio[start: start + chunk_samples]
        inputs = processor(audio_chunk, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                logits = model(input_values, attention_mask=attention_mask).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        transcriptions.append(transcription)

    return " ".join(transcriptions)


async def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file asynchronously."""
    unique_filename = f"{uuid.uuid4()}.wav"
    file_path = os.path.join(AUDIO_DIR, unique_filename)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return file_path


@app.get("/")
async def root():
    """Health check."""
    return {"message": "ASR Model is ready for inference"}


@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    file_paths = []
    transcriptions = []

    try:
        for file in files:
            file_path = await save_temp_file(file)
            file_paths.append(file_path)

        for file_path in file_paths:
            transcription = get_asr_result(file_path, model, processor)
            transcriptions.append(transcription)
            torch.cuda.empty_cache()  # Free CUDA memory after each inference

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
