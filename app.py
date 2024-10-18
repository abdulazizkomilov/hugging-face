import os
import uuid
import torch
import logging
import aiofiles
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Set device and dtype efficiently
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model and processor IDs
MODEL_ID = "openai/whisper-tiny"

# Load model and processor once at startup
logging.info("Loading model and processor...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
).to(device)

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Initialize pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Create audio directory if not exists
AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()


@app.get("/")
async def root():
    """Check the model readiness."""
    return {"message": "Model is ready for inference"}


async def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file with a unique name asynchronously."""
    unique_filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(AUDIO_DIR, unique_filename)
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return file_path


@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    """Transcribe multiple audio files."""
    file_paths = []

    try:

        for file in files:
            file_path = await save_temp_file(file)
            file_paths.append(file_path)

        results = asr_pipeline(file_paths, batch_size=len(file_paths), generate_kwargs={"language": "en"})

        return {"results": results}

    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
