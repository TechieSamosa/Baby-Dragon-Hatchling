import torch
from .train_utils import log_step, estimate_loss

def train_model(model_name: str, model, train_data, val_data, model_config, train_config, get_batch_fn):
    """
    Training pipeline following Figure 4 of the paper.
    """
    model.to(train_config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_config.learning_rate, 
        weight_decay=train_config.weight_decay
    )
    
    print(f"Starting training for {model_name} on {train_config.device}...")
    
    model.train()
    for step in range(1, train_config.max_iters + 1):
        # 1. Get batch
        X, Y = get_batch_fn(train_data, model_config.block_size, train_config.batch_size, train_config.device)
        
        # 2. Forward pass
        logits, loss = model(X, Y)
        
        # 3. Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # 4. Optimizer step
        optimizer.step()
        
        # 5. Log and evaluate
        if step % train_config.log_freq == 0:
            val_loss = estimate_loss(model, val_data, model_config, train_config, get_batch_fn)
            # Log training batch loss for plotting, as per paper (logging batch loss)
            log_step(model_name, step, loss.item(), train_config.log_file)
            print(f"Step {step:4d} | {model_name} | Train Loss {loss.item():.4f} | Val Loss {val_loss:.4f}")
            
    return model
