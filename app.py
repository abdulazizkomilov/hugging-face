import os
import uuid
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-tiny"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

AUDIO_DIR = "./audios"
os.makedirs(AUDIO_DIR, exist_ok=True)


def save_temp_file(file: UploadFile) -> str:
    """Save uploaded file with a unique name and return the path."""
    unique_filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join(AUDIO_DIR, unique_filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    return file_path


@app.post("/transcribe/")
async def transcribe_audio(files: List[UploadFile] = File(...)):
    """Handle multiple audio files and return their transcriptions."""
    transcriptions = []
    file_paths = []

    try:
        for file in files:
            file_path = save_temp_file(file)
            file_paths.append(file_path)

            result = pipe(file_path, generate_kwargs={"language": "uzbek"})
            transcriptions.append({
                "filename": file.filename,
                "transcription": result["text"]
            })

        return {"transcriptions": transcriptions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
