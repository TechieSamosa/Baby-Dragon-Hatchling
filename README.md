# BDH-Ablations: Component-Level Analysis of the Dragon Hatchling Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

This repository presents a controlled, component-level ablation study of the **Baby Dragon Hatchling (BDH)** architecture on byte-level WikiText-2 language modeling. The goal of this research is to isolate and quantify the representational importance of BDH’s core bio-inspired features.

---

## 📖 Research Abstract

The Dragon Hatchling (BDH) (Kosowski et al., arXiv:2509.26507) is a biologically-inspired sequence modeling architecture. It replaces standard Multi-Head Attention (MHA) and Feed-Forward Networks (FFN) with a dual-circuit state-space model that implements integrate-and-fire activation thresholding, Hebbian synaptic learning rules, and multiplicative concept binding.

While the original paper demonstrates that BDH scales competitively with GPT-2, the individual contribution of each core component has not been systematically isolated. This project addresses this gap through a controlled ablation study across four key dimensions:
1. **Multiplicative Conjunction Gate ($\odot$):** Evaluated against an additive alternative ($+$).
2. **Latent Space Dimensionality ($m$):** Evaluated at the default $m=128$ vs. a compressed $m=32$.
3. **Activation Non-Linearity:** Comparing biologically-plausible, exact-sparsity ReLU vs. smooth GELU.
4. **Attention Loop Representation:** Checked against a parameter-matched Transformer baseline.

---

## 🎨 Architectural Framework & Diagrams

### 1. Single Layer BDH Forward Pass
The following diagram maps the computational graph of one BDH-GPU layer. The central innovation is the **Multiplicative Gate** (logical AND) combining the primal sparse pathway (content encoding) and the dual sparse pathway (context/attention modulated).

```mermaid
graph TD
    Input[Token Input: idx] --> Embed[Token Embedding E]
    Embed --> LN1[Layer Norm]
    
    subgraph BDH_Layer ["BDH Layer (Repeated L = 6 times)"]
        LN1 --> EncPrimal[Primal Encoder W_enc]
        EncPrimal --> ReLUPrimal["Primal Activation: ReLU(·)"]
        
        ReLUPrimal --> Attn["Causal Softmax Attention <br> Q = K = xs, V = x"]
        LN1 --> Attn
        
        Attn --> LN2[Layer Norm]
        LN2 --> EncDual[Dual Encoder W_enc_v]
        EncDual --> ReLUDual["Dual Activation: ReLU(·)"]
        
        ReLUPrimal -- xs --> Gate{{"Multiplicative Gate (xs ⊙ ys)"}}
        ReLUDual -- ys --> Gate
        
        Gate --> Drop[Dropout p=0.1]
        Drop --> Dec[Decoder W_dec]
        Dec --> LN3[Layer Norm]
        
        LN1 -- Residual Skip --x LN4[Layer Norm]
        LN3 --> LN4
    end
    
    LN4 --> LMHead[LM Head: Linear]
    LMHead --> Logits[Output Logits]
```

---

### 2. Biological Circuits Analogy
BDH translates classical biological neural behavior into matrix operations. 
*   **Primal branch ($x_s$):** Excitatory neural assemblies responding directly to the stimulus.
*   **Dual branch ($y_s$):** Attention-gated inhibitory neural assemblies providing context-specific feedback.
*   **Product ($x_s \odot y_s$):** Co-activation of both circuits representing conjunctive concept binding (logical AND).
*   **Synaptic Matrix ($S_t$):** The linear attention matrix mimics Hebbian long-term potentiation: connections strengthen when neurons fire together.

```mermaid
graph LR
    subgraph Excitatory_Circuit ["Excitatory Circuit (Primal xs)"]
        A((Neuron A)) -->|Synapse| B((Neuron B))
    end
    
    subgraph Inhibitory_Circuit ["Inhibitory Circuit (Dual ys)"]
        P((Neuron P)) -->|Inhibit| Q((Neuron Q))
    end
    
    Excitatory_Circuit -- Activation --> Gate(Conjunctive Binding: xs ⊙ ys)
    Inhibitory_Circuit -- Gated Attention --> Gate
    
    classDef excitatory fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef inhibitory fill:#ffebee,stroke:#d32f2f,stroke-width:2px;
    classDef gate fill:#fff8e1,stroke:#fbc02d,stroke-width:2px;
    
    class A,B excitatory;
    class P,Q inhibitory;
    class Gate gate;
```

---

### 3. Training & Evaluation Pipeline Flowchart
The following diagram maps the execution pipeline of the experiment runner:

```mermaid
flowchart TD
    Start([Start Experiment]) --> InitConfigs[Initialize Configs & Seed 42]
    InitConfigs --> LoadData[Load WikiText-2 Byte Stream]
    LoadData --> SplitData[Split: 90% Train / 10% Validation]
    
    subgraph Loop ["Iterate over all 5 Model Variants"]
        Instantiate[Instantiate Model on Device] --> Optim[AdamW Optimizer: lr=1e-3, wd=0.1]
        Optim --> StepLoop{Step < 2900?}
        
        StepLoop -- Yes --> GetBatch[Sample Batch B=8, T=128]
        GetBatch --> Forward[Forward Pass: Cross-Entropy Loss]
        Forward --> Backward[Backward Pass & Gradients]
        Backward --> Step[Optimizer Step]
        
        Step --> CheckLog{Step % 100 == 0?}
        CheckLog -- Yes --> Eval[Estimate Val Loss on 20 batches]
        Eval --> Log[Log to results/log.jsonl]
        Log --> Increment[Step += 1]
        CheckLog -- No --> Increment
        Increment --> StepLoop
    end
    
    StepLoop -- No --> SaveResults[Save Final Weights & Run analyze.py]
    SaveResults --> Finish([End Experiment])
```

---

## 🧪 Experimental Configurations

| Architecture | Interaction | Activation | Mult. ($m$) | Latent Dim ($N$) | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Transformer** | — | GELU | — | — | Standard Multi-Head Attention & FFN |
| **BDH Base** | Multiplication ($\odot$) | ReLU | 128 | 8,192 | Standard BDH-GPU implementation |
| **BDH-NoMul** | Addition ($+$) | ReLU | 128 | 8,192 | Ablation: Replaces $\odot$ with linear blending |
| **BDH-LowDim** | Multiplication ($\odot$) | ReLU | 32 | 2,048 | Ablation: Compresses latent dimension by 4x |
| **BDH-Improved**| Multiplication ($\odot$) | GELU | 128 | 8,192 | Ablation: Soft non-linearity replaces hard ReLU |

---

## 📈 Quantitative Findings & Visualizations

After training completed, running `python analyze.py` compiled the training trajectory logs into the following results.

### 1. Training Curves (Loss Trajectories)
*Compare the convergence speed and final cross-entropy loss across all 5 models.*

![Training Loss Curves](results/plots/loss.png)

*   **The Transformer baseline** converges the fastest initially, due to the representational expressiveness of unshared parameters per layer.
*   **BDH-NoMul** (additive interaction) converges the slowest and plateaus at the highest final loss, illustrating that multiplication is vital to its performance.

### 2. Component Impact Study (Ablations)
*Signed $\Delta$ in Average Last-50 Loss compared to BDH Base. Positive values represent performance degradation (hurts the model), while negative values represent improvements.*

![Component Impact Chart](results/plots/component_impact.png)

*   **Multiplicative Interaction is crucial (+0.059):** Replacing the gate with addition leads to a massive drop in performance. Without multiplication, conjunctive binding collapses.
*   **Latent space compression helps (-0.052):** Reducing the multiplier from $m=128$ to $m=32$ slightly improves perplexity, showing that the base latent space is heavily over-parameterized at this scale.
*   **GELU activation yields minor gains (-0.039):** Replacing ReLU with GELU improves loss by providing smoother gradients, though it forfeits biological interpretability.

---

## 🛠️ Reproduction Guide

### Local Installation
```bash
pip install torch numpy matplotlib datasets
```

### Running the Experiments
```bash
# 1. Run all 5 training runs in sequence (saves to results/log.jsonl)
python -m experiments.runner

# 2. Compile metrics and generate plots
python analyze.py
```

## 📝 BibTeX Citation

```bibtex
@article{khamitkar2026bdh,
  title={BDH Architecture Analysis: A Controlled Component-Level Study of the Dragon Hatchling Language Model},
  author={Khamitkar, Aditya and Jagatap, Tushar and Saini, Nitin},
  year={2026},
  institution={SCAI, VIT Bhopal University}
}
```
