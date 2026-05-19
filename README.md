# The Baby Dragon Hatchling (BDH) Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

This repository contains the official PyTorch implementation and component-level ablation study of the **Baby Dragon Hatchling (BDH)** architecture, as presented in:

**"BDH Architecture Analysis: A Controlled Component-Level Study of the Dragon Hatchling Language Model"**
*(Aditya Khamitkar, Tushar Jagatap, Nitin Saini)*

The Dragon Hatchling (BDH) is a biologically-inspired large language model architecture that combines graph-theoretic foundations, Hebbian synaptic plasticity, sparse positive activations, and linear attention into a single state-space framework.

---

## 📖 Abstract

The Dragon Hatchling (BDH) architecture combines graph-theoretic foundations, Hebbian synaptic plasticity, sparse positive activations, and linear attention into a single state-space framework. While the original architecture rivals GPT-2 at parameter parity (10 M–1 B), the relative contribution of each design decision remains under-studied.

This repository presents a controlled, component-level ablation study of BDH on byte-level WikiText-2 language modelling. We isolate and evaluate four design dimensions: 
1. **Multiplicative interaction** between primal and dual latent branches.
2. **Latent dimensionality** governed by the internal multiplier.
3. **Sparse activation function** (ReLU vs. GELU).
4. **The core attention mechanism**.

Our experimental results across 2,900 training steps reveal that **multiplicative interaction is the most critical component** (removing it raises loss by +0.059), while switching from ReLU to GELU yields modest but consistent improvement, and reducing latent dimensionality incurs only a marginal penalty.

## 🧬 Architecture Overview

BDH is formally a local, distributed computation on a graph of $n$ neuron particles. It features:
- **Hebbian Synaptic Plasticity:** Working memory is maintained entirely through edge-reweighting governed by Hebbian learning rules.
- **Conjunctive Concept Binding:** Element-wise multiplication between a sparse primal branch and an attention-modulated dual branch acts as a coincidence detector (logical AND), naturally yielding monosemantic representations.
- **Integrate-and-Fire Dynamics:** Modeled via ReLU thresholding.

## 🗂️ Project Structure

The codebase is modular and designed for easy extension of the architecture.

```text
├── configs/            # Architecture and training hyperparameters
├── data/               # WikiText-2 (byte-level) automated downloader and batching
├── experiments/        # Model registry and training runner
├── models/             # PyTorch architecture implementations
│   ├── bdh_base.py     # Full BDH-GPU architecture (Base)
│   ├── bdh_nomul.py    # Ablation: Additive instead of multiplicative gate
│   ├── bdh_lowdim.py   # Ablation: Reduced latent dimensionality (m=32)
│   ├── bdh_improved.py # Ablation: GELU instead of ReLU activations
│   └── transformer.py  # Baseline: Standard Decoder-only Transformer
├── train/              # Training loop and validation estimation
├── utils/              # Plotting, logging, and metrics utilities
├── analyze.py          # Script to generate plots and compute summary statistics
└── README.md           # This file
```

## 🚀 Reproducing the Experiments

### 1. Requirements

Install the necessary dependencies:
```bash
pip install torch numpy matplotlib
```

### 2. Run the Training Pipeline

To train all 5 model variants (Transformer, BDH Base, BDH-NoMul, BDH-LowDim, BDH-Improved) on the byte-level WikiText-2 dataset for 2,900 steps (as specified in the paper):

```bash
python -m experiments.runner
```
*Note: The script will automatically download the WikiText-2 dataset on the first run.*

### 3. Analyze Results and Generate Plots

Once training is complete, the logs will be saved to `results/log.jsonl`. Generate the performance metrics and paper figures by running:

```bash
python analyze.py
```
This will output the component impact deltas and generate the following plots in `results/plots/`:
- `loss.png`: Training loss curves for all models.
- `smooth_loss.png`: Smoothed training trajectories.
- `component_impact.png`: Bar chart of the ablation impacts.
- `comparison.png`: Final performance comparison.

## 📊 Key Findings

1. **Multiplicative gates are non-negotiable:** Replacing the `xs ⊙ ys` product with addition `xs + ys` causes the largest performance degradation, confirming that conjunctive feature binding is central to BDH's representational power.
2. **Latent dimension is over-parameterized at small scales:** Reducing the internal multiplier from $m=128$ to $m=32$ marginally improves performance while reducing parameter count by 4x.
3. **GELU vs. ReLU:** GELU offers a small perplexity improvement (better gradient flow) over ReLU, but sacrifices strict integrate-and-fire biological analogies and potential interpretability benefits.

## 📝 Citation

If you use this code or findings in your research, please consider referencing:

```bibtex
@article{khamitkar2026bdh,
  title={BDH Architecture Analysis: A Controlled Component-Level Study of the Dragon Hatchling Language Model},
  author={Khamitkar, Aditya and Jagatap, Tushar and Saini, Nitin},
  year={2026},
  institution={SCAI, VIT Bhopal University}
}
```

## 📄 License
This project is licensed under the MIT License.
