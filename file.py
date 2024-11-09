import gdown
import os

file_id = os.getenv("MODEL_ID")
output = "stt_model.zip"

gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)


