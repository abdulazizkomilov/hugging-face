import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import librosa
from tempfile import NamedTemporaryFile
import os

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"
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


async def convert_to_wav(upload_file: UploadFile) -> BytesIO:
    try:
        # Read the UploadFile asynchronously and convert it into a file-like object
        file_content = await upload_file.read()  # Read the contents of the file
        audio = AudioSegment.from_file(BytesIO(file_content))  # Convert to AudioSegment

        # Set the sample rate and channels
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Save to a BytesIO object
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process audio file: {str(e)}")


def convert_wav_to_numpy(wav_io: BytesIO) -> np.ndarray:
    try:
        wav_io.seek(0)  # Ensure the BytesIO pointer is at the beginning
        audio_data, _ = librosa.load(wav_io, sr=16000)
        return audio_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to convert wav to numpy array: {str(e)}")


def download(file: UploadFile) -> str:
    """Download the uploaded file and return its temporary file path."""
    try:
        temp_file = NamedTemporaryFile(delete=False, suffix=".mp3")  # Adjust suffix as needed
        with temp_file as f:
            f.write(file.file.read())  # Save the uploaded file to disk
        return temp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


def delete(file_path: str):
    """Delete the temporary file."""
    try:
        os.remove(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    """Handle audio transcription."""
    try:
        # Download the file to a temporary location
        file_path = download(file)

        # Perform transcription
        result = pipe(file_path, return_timestamps=True, generate_kwargs={"language": "Uzbek"})

        # Clean up the temporary file
        delete(file_path)

        return {"transcription": result["text"], "timestamps": result["chunks"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
