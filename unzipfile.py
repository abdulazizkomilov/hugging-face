import zipfile

output = "stt_model.zip"
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("stt_model")
