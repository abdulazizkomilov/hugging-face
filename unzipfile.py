import zipfile

output = "super-mixed-wav2vec2-1b-30-04-2024.zip"
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("super-mixed-wav")
