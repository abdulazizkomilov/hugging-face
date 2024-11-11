import gdown
import os

file_id = os.getenv("FILE_ID")
output = "super-mixed-wav.zip"

gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)


