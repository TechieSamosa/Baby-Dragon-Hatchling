import os
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(data, save_dir):
    """
    Plot raw training curves (like Figure 5).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for model, logs in data.items():
        plt.plot(logs["steps"], logs["losses"], label=model, alpha=0.8)
        
    plt.xlabel("Training Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss Curves — All Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss.png"))
    plt.close()

def plot_smoothed_curves(data, save_dir, window=5):
    """
    Plot smoothed training curves.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for model, logs in data.items():
        if len(logs["losses"]) > window:
            smoothed = np.convolve(logs["losses"], np.ones(window)/window, mode='valid')
            steps = logs["steps"][(window-1):]
            plt.plot(steps, smoothed, label=model, alpha=0.8)
            
    plt.xlabel("Training Step")
    plt.ylabel("Smoothed Loss")
    plt.title("Smoothed Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "smooth_loss.png"))
    plt.close()

def plot_component_impact(impacts, save_dir):
    """
    Plot bar chart of component impacts (like Figure 6).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    labels = list(impacts.keys())
    values = list(impacts.values())
    
    colors = ['#ff9999' if v > 0 else '#99ff99' for v in values]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (0.002 if yval > 0 else -0.005), 
                 f"{yval:+.3f}", ha='center', va='bottom' if yval > 0 else 'top')
                 
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("Δ Loss vs. BDH Base")
    plt.title("Component Impact (Δ Avg Last-50 Loss)")
    plt.savefig(os.path.join(save_dir, "component_impact.png"))
    plt.close()

def plot_final_comparison(avg_losses, save_dir):
    """
    Plot bar chart comparing final performance of all models.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    labels = list(avg_losses.keys())
    values = list(avg_losses.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='skyblue')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, 
                 f"{yval:.3f}", ha='center', va='bottom')
                 
    plt.ylabel("Avg Last-50 Loss")
    plt.title("Final Performance Comparison")
    
    # Set y-axis to start slightly below the min value to emphasize differences
    min_val = min(values)
    plt.ylim(max(0, min_val - 0.5), max(values) + 0.2)
    
    plt.savefig(os.path.join(save_dir, "comparison.png"))
    plt.close()
