# Improving Masked Diffusion LMs via Post Training Loops of Iterative Supervised Trajectory-Aware RL

**Ashutosh Panda** · apanda2@andrew.cmu.edu · April 2026

---

## Why Diffusion Reasoning, and Why Now

Autoregressive (AR) models generate text token-by-token, left-to-right. Masked diffusion language models (MDLMs) take a different approach: they begin from a fully masked sequence and iteratively unmask tokens over T denoising steps with bidirectional attention at every step, enabling global revision and coherence that AR models structurally cannot achieve.

Two recent results make this timely. He et al. [1] show that MDLMs implicitly maintain joint predictions over all still-masked positions — latent tokens — and that these are causally responsible for MDLMs' reasoning advantage: ablating them degrades performance substantially. Critically, this mechanism is emergent; standard training produces it incidentally without any explicit reward for making those predictions useful. Prabhudesai et al. [2] show that MDLMs tolerate data repetition far better than AR models (R* ≈ 500 vs. ~15 epochs) because randomised masking acts as implicit data augmentation. For any self-improvement loop cycling repeatedly through a fixed problem set, this structural advantage is decisive.

---

## The Problem with Existing MDLM RL

diffu-GRPO [3] first applied policy gradient RL to MDLMs using a flat outcome advantage A(o): every token receives the same gradient multiplier regardless of which denoising step produced it. SAPO [4] identifies the resulting failure: with only outcome reward, denoising steps degenerate into unstructured refinement, shuffling tokens without contributing to the solution. ATPO [5] shows that within trajectories, only transient zones of confusion — uncertainty spikes that predict success or failure — deserve gradient attention. Critically, all three are single-round: no prior work asks whether MDLMs can improve iteratively through self-generated signal.

---

## Our Approach: Compositional Objective

We compose SAPO's step-level rewards with ATPO's adaptive gradient weighting inside a multi-round iterative loop, building the objective in three steps.

**Step 1 (diffu-GRPO baseline).** With outcome advantage A(o) and bidirectional token predictions:

$$\mathcal{L}_{\text{GRPO}} = \mathbb{E}\left[A(o) \sum_{i=1}^{N} \log \pi_\theta(x_i \mid x_{\setminus i})\right] \tag{1}$$

Every token gets the same flat signal; no step structure is modelled.

**Step 2 (SAPO interval rewards).** Partition the trajectory into K intervals; let S_k be tokens unmasked during interval k and z_{t_k} the partial state at its boundary. Define:

$$r_k = \log p(\text{correct} \mid z_{t_{k+1}}) - \log p(\text{correct} \mid z_{t_k}) \tag{2}$$

Replace the flat advantage with per-interval rewards:

$$\mathcal{L}_{\text{SAPO}} = \mathbb{E}\left[\sum_{k=1}^{K} r_k \sum_{i \in S_k} \log \pi_\theta(x_i \mid x_{\setminus i})\right] \tag{3}$$

This directly trains the latent token mechanism He et al. identified: interval k is rewarded precisely when its latent predictions are more concentrated on the correct solution than in the previous step, aiming to make an emergent property deliberate.

**Step 3 (ATPO soft weighting).** Score each interval by rate of entropy change RoEC_k and confidence margin CM_k:

$$w_k = \sigma(\alpha \cdot \text{RoEC}_k + \beta \cdot (1 - \text{CM}_k)) \tag{4}$$

The combined objective concentrates both reward and gradient on the intervals where meaningful decisions are made:

$$\mathcal{L} = \mathbb{E}\left[\sum_{k=1}^{K} w_k \cdot r_k \sum_{i \in S_k} \log \pi_\theta(x_i \mid x_{\setminus i})\right] \tag{5}$$

r_k defines what each step is rewarded for; w_k defines where the gradient flows. Computing w_k requires no extra forward passes — entropy and CM are read from logits already produced during denoising.

---

## The Iterative Loop and the DLM Advantage

We run L inside a multi-round loop.

- **Round 0 (STaR cold start):** Generate rollouts from base LLaDA-8B [6], filter to correct traces, and SFT. This raises the floor so Rounds 1+ have contrastive signal — a prerequisite for r_k to be informative.
- **Rounds 1–4 (RL):** Apply L, re-calibrate thresholds each round by maximising Corr(w_k, |r_k|) on held-out trajectories, regenerate rollouts.

The Prabhudesai et al. result is the engine: AR models overfit after ~4 repetitions, capping iterative benefit. MDLMs sustain signal across ~500. We predict the DLM–AR performance gap widens monotonically with rounds.

---

## Diagnostic: Is the Trajectory a Policy?

The central question is whether denoising steps are a genuine policy rollout or merely iterative sampling. We test this by training two Round-2 variants: free vs. left-to-right-fixed unmasking order. If the free-order model develops sharper w_k distributions and higher r_k at specific intervals, the trajectory is acquiring policy structure. If not, architectural changes are needed. This ablation gates whether Phase 2 — a learned critic over latent denoising states — is worth pursuing.

---

## References

[1] He et al., *Reasoning with Latent Tokens in Diffusion LMs*, arXiv:2602.03769.  
[2] Prabhudesai et al., *Diffusion Beats AR in Data-Constrained Settings*, NeurIPS 2025.  
[3] Zhao et al., *d1: Scaling Reasoning in Diffusion LLMs via RL*, arXiv:2504.12216.  
[4] Xie et al., *SAPO: Step-Aware Policy Optimization*, arXiv:2510.01544.  
[5] Chen et al., *Reasoning in Diffusion LLMs: Confusion Zones*, arXiv:2511.15208.  
[6] Nie et al., *LLaDA: Large Language Diffusion Models*, arXiv:2502.09992.
