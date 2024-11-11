from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Initialize GoogleAuth and authorize with credentials
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Opens a web server to authenticate
drive = GoogleDrive(gauth)

# Example: Upload a file to Google Drive
file = drive.CreateFile({'title': 'example.txt'})
file.SetContentString('Hello, Google Drive!')
file.Upload()
print("Uploaded file with ID:", file['id'])
