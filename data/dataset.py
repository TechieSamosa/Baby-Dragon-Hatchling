import os
import numpy as np
import torch
from datasets import load_dataset

def load_data(data_dir="data"):
    """Loads WikiText-2 byte-level dataset, splitting into train/val."""
    print("Loading WikiText-2 using HuggingFace datasets...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    all_bytes = []
    # Combine train, validation, and test splits as per the paper
    for split in ["train", "validation", "test"]:
        for item in dataset[split]:
            text = item['text']
            if text.strip():  # paper says "removing empty lines"
                # Encode text to bytes
                all_bytes.extend(text.encode('utf-8'))
                
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
