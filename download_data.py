#!/usr/bin/env python3
import urllib.request
import zipfile
import ssl
from pathlib import Path

ssl_context = ssl._create_unverified_context()
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)

data_dir = Path('./data')
data_dir.mkdir(exist_ok=True)
zip_path = data_dir / 'tiny-imagenet-200.zip'
extract_path = data_dir / 'tiny-imagenet-200'

if extract_path.exists():
    print(f"Dataset exists at {extract_path}")
else:
    print("Downloading Tiny ImageNet...")
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    zip_path.unlink()
    print("Done!")

