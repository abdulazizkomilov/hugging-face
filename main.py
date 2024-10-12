import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
from io import BytesIO
import numpy as np
import librosa

app = FastAPI()

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
# model_id = "openai/whisper-tiny"

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


def convert_to_wav(file_path):
    try:
        # Load the file as AudioSegment
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Convert to 16kHz, mono

        # Save the audio to a BytesIO object
        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    except Exception as e:
        print(f"Failed to process audio file: {str(e)}")
        return None


def convert_wav_to_numpy(wav_io: BytesIO) -> np.ndarray:
    try:
        wav_io.seek(0)  # Ensure the BytesIO pointer is at the beginning
        audio_data, sample_rate = librosa.load(wav_io, sr=16000)  # Convert audio to numpy array
        return audio_data
    except Exception as e:
        print(f"Failed to convert WAV to numpy: {str(e)}")
        return None


@app.get("/")
async def root():
    return {"message": "Model is ready for inference"}


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        wav_io = convert_to_wav(file)
        if wav_io:
            audio_data = convert_wav_to_numpy(wav_io)
            result = pipe(audio_data, generate_kwargs={"language": "uzbek"})

            return {"transcription": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during transcription: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
