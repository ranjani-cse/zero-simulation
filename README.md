
# ZeRO Optimizer Simulation: 32 Virtual GPUs

## What I Built

A simulation of Microsoft DeepSpeed's ZeRO optimizer stages (1, 2, 3) on 32 virtual GPUs. The simulation shows how memory and communication change across ZeRO stages.

## What I Understood

### The Problem ZeRO Solves

In standard data-parallel training (DDP), every GPU stores a **full copy** of:
- Model parameters
- Gradients
- Optimizer states (Adam's m and v)

For my demo model (2.1M parameters), this means **33.60 MB per GPU** — and across 32 GPUs, that's **1,075 MB (1.05 GB)** of redundant memory.

### How ZeRO Fixes It

ZeRO partitions different components across GPUs:

| Stage | What's Partitioned | Memory per GPU | Reduction |
|-------|-------------------|----------------|-----------|
| Baseline | Nothing | 33.60 MB | 0% |
| ZeRO-1 | Optimizer states | 17.32 MB | 48.4% |
| ZeRO-2 | Optimizer + Gradients | 9.19 MB | 72.7% |
| ZeRO-3 | Optimizer + Gradients + Parameters | 1.05 MB | **96.9%** |

### The Memory vs Communication Trade-off

| Stage | Memory (MB) | Communication (MB) |
|-------|-------------|---------------------|
| Baseline | 33.60 | 16.80 |
| ZeRO-1 | 17.32 | 25.20 |
| ZeRO-2 | 9.19 | 16.80 |
| ZeRO-3 | 1.05 | 25.20 |

**Key insight:** ZeRO-3 saves the most memory (97%) but requires more communication (50% more). This is the fundamental trade-off in distributed training.

### My Key Takeaways

1. **Redundancy is the problem.** In DDP, every GPU holds the same data — wasteful.
2. **ZeRO partitions progressively.** Each stage partitions more: optimizer → gradients → parameters.
3. **Memory savings scale with GPUs.** With 32 GPUs, ZeRO-3 gives ~32× memory reduction.
4. **Communication is the cost.** More partitioning = more communication.
5. **ZeRO-3 is essential for huge models.** When a model doesn't fit on one GPU, ZeRO-3 makes training possible.

## Files

- `zero_simulation.ipynb` — Full simulation notebook
- `zero_comparison.png` — Memory breakdown plot
- `memory_vs_communication.png` — Trade-off plot

## How to Run

Open the notebook in Google Colab and run all cells top to bottom.
