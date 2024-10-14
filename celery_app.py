import torch
from celery import Celery
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# MODEL AI

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

# CELERY

celery_app = Celery(
    "hugging-face",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

celery_app.conf.update(
    broker_connection_retry=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)

celery_app.conf.broker_connection_retry_on_startup = True

celery_app.autodiscover_tasks()


@celery_app.task
def transcribe_audio_task(file_path):
    result = pipe(file_path, generate_kwargs={"language": "uzbek"})
    return result
