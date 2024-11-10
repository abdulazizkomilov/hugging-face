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

# Set up device and load model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.to(device)
    model.eval()
    print("Model and processor loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model or processor: {e}")


def get_asr_result(audio_path, model, processor, sr=16000):
    """Perform ASR on an audio file."""
    # Load audio using torchaudio for faster performance
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)(waveform)
    audio = waveform.squeeze().numpy()

    # Prepare inputs and move them to the appropriate device
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference with mixed precision if available
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    # Decode the logits to obtain the transcription
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0]


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
    """Health check endpoint."""
    return {"message": "ASR Model is ready for inference"}


@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    file_paths = []
    transcriptions = []

    try:
        # Save files and get paths
        for file in files:
            file_path = await save_temp_file(file)
            file_paths.append(file_path)

        # Batch transcribe files
        for file_path in file_paths:
            transcription = get_asr_result(file_path, model, processor)
            transcriptions.append(transcription)

        # Prepare response
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

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)  # Run with multiple workers for better performance
