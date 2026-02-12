
import os
import io
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service(credentials_path='client_secrets.json', token_path='token.json'):
    """Shows basic usage of the Drive v3 API. 
    Supports both OAuth Client ID and Service Account credentials.
    """
    creds = None
    
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Credentials file not found at {credentials_path}")

    # Detect credential type
    import json
    from google.oauth2 import service_account
    
    with open(credentials_path, 'r') as f:
        data = json.load(f)
        
    # Case 1: Service Account
    if 'type' in data and data['type'] == 'service_account':
        print("Detected Service Account credentials.")
        creds = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)

    # Case 2: OAuth Client ID (Installed App)
    if os.path.exists(token_path):
        if token_path.endswith('.json'):
            from google.oauth2.credentials import Credentials
            try:
                creds = Credentials.from_authorized_user_file(token_path, SCOPES)
            except ValueError:
                print("Error loading token.json, will regenerate.")
        else:
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
            
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        if token_path.endswith('.json'):
             with open(token_path, 'w') as token:
                token.write(creds.to_json())
        else:
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)

    service = build('drive', 'v3', credentials=creds)
    return service

def find_folder(service, folder_name, parent_id=None):
    """Finds a folder by name."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(q=query, fields="nextPageToken, files(id, name)").execute()
    items = results.get('files', [])
    
    if not items:
        return None
    return items[0]['id']

def traverse_drive_folder(service, folder_id, path=""):
    """Recursively traverses a drive folder and yields (item, relative_path)."""
    query = f"'{folder_id}' in parents and trashed=false"
    page_token = None
    while True:
        results = service.files().list(q=query, 
                                       fields="nextPageToken, files(id, name, mimeType)",
                                       pageToken=page_token).execute()
        items = results.get('files', [])
        
        for item in items:
            if item['mimeType'] == 'application/vnd.google-apps.folder':
                # Recurse into subfolder
                new_path = os.path.join(path, item['name'])
                yield from traverse_drive_folder(service, item['id'], new_path)
            else:
                yield item, os.path.join(path, item['name'])
                
        page_token = results.get('nextPageToken')
        if not page_token:
            break

def download_file(service, file_id, local_path):
    """Downloads a file from Drive to local path."""
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        # print(f"Download {int(status.progress() * 100)}%.")

def upload_file(service, local_path, parent_id, upload_name=None):
    """Uploads a file to a specific folder on Drive."""
    from googleapiclient.http import MediaFileUpload
    
    if not upload_name:
        upload_name = os.path.basename(local_path)
        
    file_metadata = {
        'name': upload_name,
        'parents': [parent_id]
    }
    media = MediaFileUpload(local_path, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded {upload_name} (ID: {file.get('id')})")
    return file.get('id')

def ensure_folder_exists(service, parent_id, folder_name):
    """Finds or creates a folder within a parent folder."""
    query = f"'{parent_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    if files:
        return files[0]['id']
    else:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        print(f"Created folder '{folder_name}' (ID: {folder.get('id')})")
        return folder.get('id')
