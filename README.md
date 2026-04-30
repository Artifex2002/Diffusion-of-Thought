# Diffusion-of-Thought

## Improving Diffusion LMs via Iterative Supervised Trajectory-Aware RL

## Getting Started

### Prerequisites

- macOS 12.3 or later (required for MPS support)
- [Homebrew](https://brew.sh)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) + [Mamba](https://mamba.readthedocs.io)

### Step 1: Install System Dependencies

```bash
# Install Conda if you don't have it. Follow the guide below:
# https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
# Install Mamba into your base conda env if you haven't already
conda install mamba -n base -c conda-forge
```

### Step 2: Clone This Repository

```bash
git clone https://github.com/Artifex2002/Diffusion-of-Thought.git

# Feel free to rename this to your desired project name
cd Diffusion-of-Thought
```

### Step 3: Create & Activate the Conda Environment

```bash
# Creates the environment with all pinned dependencies
mamba env create -f environment.yml -n DoT

# Activate it
conda activate DoT
```


🎉 **Environment is now good to go!**

## Planned Repository Structure

```
├── README.md                  # This file
├── checklist.md               # Step-by-step experiment checklist with deliverables
├── proposal.docx              # Full research proposal document
├── src/
│   ├── generate.py            # Rollout generation with trajectory logging
│   ├── verify.py              # Programmatic correctness verifiers (math + code)
│   ├── reward.py              # SAPO trajectory interval reward implementation
│   ├── train_sft.py           # Round 1: STaR-style supervised fine-tuning
│   ├── train_rl.py            # Rounds 2+: Trajectory-aware RL (SAPO reward)
│   └── eval.py                # Evaluation on GSM8K, MATH, HumanEval, MBPP
├── experiments/
│   └── configs/               # Config files for each experiment (EXP-0A through EXP-6B)
└── analysis/
    └── trajectory_viz.py      # Visualise denoising trajectories across rounds
```

## Key References

| Paper | arXiv | Role |
|---|---|---|
| LLaDA | [2502.09992](https://arxiv.org/abs/2502.09992) | Base model |
| Diffusion Beats AR | [2507.15857](https://arxiv.org/abs/2507.15857) | Data efficiency motivation |
| d1 / diffu-GRPO | [2504.12216](https://arxiv.org/abs/2504.12216) | Baseline RL algorithm |
| SAPO | [2510.01544](https://arxiv.org/abs/2510.01544) | Trajectory reward design |
| ATPO | [2511.15208](https://arxiv.org/abs/2511.15208) | Trajectory zone analysis |
| Latent Tokens (He et al.) | [2602.03769](https://arxiv.org/abs/2602.03769) | Theoretical motivation + Phase 2 |
| LogicDiff | [2603.26771](https://arxiv.org/abs/2603.26771) | Unmasking order evidence |
| STaR | — | Round 1 cold start loop |

## Status

🔬 **Phase 1 in progress** — Environment setup and baseline evaluation (Stage 0–1 per checklist).
