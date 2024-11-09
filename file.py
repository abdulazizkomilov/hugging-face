import gdown
import zipfile
import os

file_id = os.getenv("MODEL_ID")
output = "stt_model.zip"

gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)


with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("stt_model")

