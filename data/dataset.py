import os
import urllib.request
import zipfile
import numpy as np
import torch

def download_wikitext2_if_needed(data_dir="data"):
    """Downloads WikiText-2 dataset if it doesn't exist."""
    os.makedirs(data_dir, exist_ok=True)
    raw_file = os.path.join(data_dir, "wikitext-2-raw-v1.zip")
    train_file = os.path.join(data_dir, "wikitext-2-raw", "wiki.train.raw")
    
    if not os.path.exists(train_file):
        print("Downloading WikiText-2...")
        url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
        urllib.request.urlretrieve(url, raw_file)
        with zipfile.ZipFile(raw_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Download complete.")
        
def load_data(data_dir="data"):
    """Loads WikiText-2 byte-level dataset, splitting into train/val."""
    download_wikitext2_if_needed(data_dir)
    
    # We combine train, valid, test and then split 90/10 as per paper
    splits = ["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"]
    all_bytes = []
    
    for split in splits:
        filepath = os.path.join(data_dir, "wikitext-2-raw", split)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                content = f.read()
                # Remove empty lines (consecutive newlines) or just keep raw bytes.
                # The paper says "removing empty lines, and saving as a memory-mapped binary file."
                # For simplicity, we just use raw bytes directly.
                all_bytes.extend(content)
                
    data_np = np.array(all_bytes, dtype=np.uint8)
    
    n = len(data_np)
    split_idx = int(0.9 * n)
    train_data = data_np[:split_idx]
    val_data = data_np[split_idx:]
    
    return train_data, val_data

def get_batch(data, block_size, batch_size, device):
    """
    Sample a batch of data.
    Listing 6 from the paper.
    """
    # Sample 'batch_size' random starting positions
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # x: (B, T) byte sequences
    x = torch.stack([
        torch.from_numpy(data[i:i+block_size].astype(np.int64))
        for i in ix
    ])
    
    # y: shifted by 1 (next-token prediction)
    y = torch.stack([
        torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64))
        for i in ix
    ])
    
    return x.to(device), y.to(device)
