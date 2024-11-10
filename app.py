import os
import uuid
import torch
import logging
import aiofiles
import torchaudio
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Initialize FastAPI app
app = FastAPI()

# Audio directory
AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Model initialization
model_id = "./stt_model/medium-wav2vec-1"

try:
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
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


def transcribe_audio_file(file_path: str, model, processor) -> str:
    """Process and transcribe audio file using ASR model."""
    # Load audio data
    audio, _ = torchaudio.load(file_path)
    audio = audio.squeeze().numpy()  # Remove extra dimensions for processing

    # Process with the processor
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

    # Model inference
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


@app.get("/")
async def root():
    """Health check."""
    return {"message": "ASR Model is ready for inference"}


@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    """Transcribe multiple audio files asynchronously with batch processing."""
    file_paths = []
    transcriptions = []

    try:
        # Save and preprocess audio files
        for file in files:
            file_path = await save_temp_file(file)
            processed_path = preprocess_audio(file_path)
            file_paths.append(processed_path)

        # Transcribe in batches
        for batch_start in range(0, len(file_paths), 2):  # Batch size of 2
            batch_files = file_paths[batch_start:batch_start + 2]
            batch_results = [transcribe_audio_file(file, model, processor) for file in batch_files]
            transcriptions.extend(batch_results)

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
