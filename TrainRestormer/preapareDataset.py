import requests
import zipfile
import io
from tqdm import tqdm

def download_and_extract(url, output_dir):
    # Download with progress bar
    print("Downloading...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    block_size = 1024
    downloaded_data = io.BytesIO()
    
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        downloaded_data.write(data)
    progress_bar.close()
    
    # Extract with progress bar
    print("\nExtracting...")
    downloaded_data.seek(0)
    with zipfile.ZipFile(downloaded_data) as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="Extracting"):
            zip_ref.extract(member=file, path=output_dir)

# Usage
url = 'https://huggingface.co/datasets/riponcs/ASU_Turbulence_Clean_100k_dataset_v1/resolve/main/ASU_Turbulence_Clean_100k_dataset_v1.zip'
output_dir = './data/'

download_and_extract(url, output_dir)