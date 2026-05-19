import os
import argparse
from utils.logger import load_logs
from utils.metrics import compute_avg_last_n, compute_component_impact
from utils.plot import plot_training_curves, plot_smoothed_curves, plot_component_impact, plot_final_comparison

def main():
    parser = argparse.ArgumentParser(description="Analyze BDH training logs and generate plots.")
    parser.add_argument("--log_file", type=str, default="results/log.jsonl", help="Path to the training log file")
    parser.add_argument("--plot_dir", type=str, default="results/plots", help="Directory to save plots")
    parser.add_argument("--n_last", type=int, default=50, help="Number of last steps to average for metrics")
    args = parser.parse_args()
    
    print(f"Loading logs from {args.log_file}...")
    data = load_logs(args.log_file)
    
    if not data:
        print(f"No data found in {args.log_file}. Run experiments first.")
        return
        
    print(f"Loaded data for models: {list(data.keys())}")
    
    # Calculate average last-50 loss
    avg_losses = {}
    print("\n--- Final Metrics ---")
    for model, logs in data.items():
        avg_loss = compute_avg_last_n(logs["losses"], n=args.n_last)
        avg_losses[model] = avg_loss
        
        final_loss = logs["losses"][-1] if logs["losses"] else float('nan')
        print(f"{model:15s} | Final: {final_loss:.4f} | Avg Last-{args.n_last}: {avg_loss:.4f}")
        
    # Calculate component impact (relative to bdh_base)
    if "bdh_base" in avg_losses:
        base_loss = avg_losses["bdh_base"]
        impacts = {}
        
        # Define pretty labels for the paper's ablation plot
        impact_labels = {
            "bdh_nomul": "Multiplication (vs Add)",
            "bdh_improved": "Activation (GELU vs ReLU)",
            "bdh_lowdim": "Latent Dim (32 vs 128)"
        }
        
        for model in ["bdh_nomul", "bdh_improved", "bdh_lowdim"]:
            if model in avg_losses:
                impact = compute_component_impact(base_loss, avg_losses[model])
                impacts[impact_labels[model]] = impact
                
        if impacts:
            plot_component_impact(impacts, args.plot_dir)
            print("\nGenerated component impact plot.")
            
    # Generate all plots
    print(f"\nGenerating plots in {args.plot_dir}...")
    plot_training_curves(data, args.plot_dir)
    plot_smoothed_curves(data, args.plot_dir)
    plot_final_comparison(avg_losses, args.plot_dir)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
