import os
import torch
import random
import numpy as np
from configs.config import get_model_config, get_train_config
from data.dataset import load_data, get_batch
from experiments.variants import get_model, VARIANTS
from train.train import train_model

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    train_config = get_train_config()
    set_seed(train_config.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_config.device = device
    
    # Load dataset once
    train_data, val_data = load_data()
    print(f"Data loaded: {len(train_data)} train bytes, {len(val_data)} val bytes.")
    
    # Run all variants
    for model_name in VARIANTS.keys():
        print(f"\n--- Running {model_name} ---")
        
        # Reset seed before each model to ensure comparable initializations
        set_seed(train_config.seed)
        
        model_config = get_model_config(model_name)
        model = get_model(model_name, model_config)
        
        # Print parameter count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{model_name} parameters: {params/1e6:.2f}M")
        
        # Train
        train_model(
            model_name=model_name,
            model=model,
            train_data=train_data,
            val_data=val_data,
            model_config=model_config,
            train_config=train_config,
            get_batch_fn=get_batch
        )
        
if __name__ == "__main__":
    main()
