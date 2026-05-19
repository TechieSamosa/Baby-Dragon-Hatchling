import os
import json
import torch

def log_step(model_name: str, step: int, loss: float, log_file: str):
    """
    Append a log entry in JSONL format.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    entry = {
        "model": model_name,
        "step": step,
        "loss": float(loss)
    }
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

@torch.no_grad()
def estimate_loss(model, data, config, train_config, get_batch_fn):
    """
    Estimate validation loss over multiple batches.
    """
    model.eval()
    losses = torch.zeros(train_config.eval_iters)
    for k in range(train_config.eval_iters):
        X, Y = get_batch_fn(data, config.block_size, train_config.batch_size, train_config.device)
        _, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean().item()
    model.train()
    return out
