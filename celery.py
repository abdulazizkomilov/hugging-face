from celery import Celery

celery = Celery(
    "tasks", 
    broker="redis://localhost:6379/0", 
    backend="redis://localhost:6379/0"
)

@celery.task
def transcribe_audio_task(audio_data):
    result = pipe(audio_data, return_timestamps=True, generate_kwargs={"language": "uzbek"})
    return {"text": result["text"], "timestamps": result["chunks"]}
