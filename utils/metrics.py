import numpy as np

def compute_avg_last_n(losses, n=50):
    if len(losses) == 0:
        return float('nan')
    return float(np.mean(losses[-n:]))

def compute_std_last_n(losses, n=50):
    if len(losses) == 0:
        return float('nan')
    return float(np.std(losses[-n:]))

def compute_component_impact(base_loss, variant_loss):
    """
    Computes delta relative to Base. Positive means variant is worse (degradation).
    """
    return variant_loss - base_loss
