from fastapi import FastAPI, UploadFile, File, HTTPException
from celery.result import AsyncResult
from tempfile import NamedTemporaryFile
import os

from celery_app import transcribe_audio_task

app = FastAPI()


def download(file: UploadFile) -> str:
    """Download the uploaded file and return its temporary file path."""
    try:
        temp_file = NamedTemporaryFile(delete=False, suffix=".mp3")
        with temp_file as f:
            f.write(file.file.read())
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
        file_path = download(file)

        # result = pipe(file_path, return_timestamps=True, generate_kwargs={"language": "Uzbek"})

        task = transcribe_audio_task.delay(file_path)

        return {"task_id": task.id, "status": "Processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.get("/transcribe/{task_id}")
async def get_transcription_result(task_id: str):
    task = AsyncResult(task_id)
    if task.state == "PENDING":
        return {"status": "Processing"}
    elif task.state == "SUCCESS":
        return task.result
    else:
        raise HTTPException(status_code=500, detail="Task failed")
