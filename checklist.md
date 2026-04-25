# Phase 1 Research Checklist
## Improving Diffusion LMs via Iterative Supervised Trajectory-Aware RL

**Base Model:** LLaDA-8B ([huggingface.co/GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct))  
**Goal:** Show that a masked diffusion LM improves iteratively across iterative supervised learning rounds, with SAPO trajectory rewards inducing meaningful step-level reasoning structure.

---

## Stage 0 — Environment Setup

- [ ] Clone and install LLaDA-8B inference + fine-tuning codebase
- [ ] Verify LLaDA-8B generates correct outputs on 5 GSM8K examples (sanity check)
- [ ] Set up programmatic correctness verifier for math (sympy-based answer checker)
- [ ] Set up programmatic correctness verifier for code (unit test execution sandbox)
- [ ] Implement trajectory logger: records full denoising trajectory (all intermediate partially-unmasked states) for every generation
- [ ] Confirm training loop runs on a small batch (10 examples, 1 gradient step) without errors

**Deliverable:** Reproducible environment with working generation, verification, and trajectory logging.

---

## Stage 1 — Baseline Evaluation (Pre-Training)

**Datasets:**
- Math: GSM8K test set (1,319 problems), MATH Level 1–3 (accessible difficulty for 8B model)
- Code: HumanEval (164 problems), MBPP (374 problems)

**Experiments:**
- [ ] **EXP-0A:** Evaluate LLaDA-8B-Instruct zero-shot on GSM8K → record Pass@1
- [ ] **EXP-0B:** Evaluate LLaDA-8B-Instruct zero-shot on MATH Level 1–3 → record Pass@1
- [ ] **EXP-0C:** Evaluate LLaDA-8B-Instruct zero-shot on HumanEval → record Pass@1
- [ ] **EXP-0D:** Evaluate LLaDA-8B-Instruct zero-shot on MBPP → record Pass@1
- [ ] Record baseline SAPO intermediate reward scores on 100 random trajectories (establishes pre-training trajectory quality baseline)

**Deliverable:** Baseline numbers table. All subsequent experiments compared against this.

---

## Stage 2 — Round 1: STaR Cold Start (Supervised)

**Dataset:** GSM8K train set (7,473 problems) + MATH train set (7,500 problems, Level 1–3 only)

**Procedure:**
- [ ] Generate 8 rollouts per problem using LLaDA-8B with temperature sampling
- [ ] Filter: keep only rollouts where final answer is verified correct
- [ ] Log yield rate (what % of problems produce at least 1 correct rollout — if <15%, the cold start will be weak, flag this)
- [ ] **EXP-1A:** Fine-tune LLaDA-8B on filtered correct traces (standard cross-entropy, next-token style on unmasked positions)
  - Batch size: 32, LR: 1e-5, epochs: 2, checkpoint every 500 steps
- [ ] Evaluate Round 1 model on GSM8K test + MATH test → record Pass@1
- [ ] Compare against baseline (EXP-0A/0B) — expect modest but nonzero gain

**Deliverable:** Round 1 checkpoint. Confirmed that STaR cold start produces measurable improvement over zero-shot baseline.

---

## Stage 3 — Round 2+: SAPO Trajectory-Aware RL

**This is the novel contribution. Rounds 2, 3, 4 follow the same loop.**

### SAPO Reward Implementation
- [ ] Implement interval reward function: for each denoising interval [t_i, t_{i+1}], compute Δ = log p(correct answer | state at t_{i+1}) − log p(correct answer | state at t_i)
- [ ] Validate reward implementation on 20 hand-checked trajectories (confirm signal direction is correct)
- [ ] Confirm reward is computationally tractable per batch (profile time per step)

### Round 2
- [ ] Generate fresh rollouts from Round 1 model (8 per problem, same train set)
- [ ] **EXP-2A:** Apply SAPO policy gradient update on full trajectory with interval rewards
  - Use PPO-clip or REINFORCE with baseline; clip reward at ±3 to prevent instability
- [ ] Evaluate on GSM8K test + MATH test → record Pass@1
- [ ] Record SAPO intermediate reward scores on 100 trajectories → compare to Stage 1 baseline scores
- [ ] **Ablation EXP-2B:** Run identical setup but with outcome-only reward (no trajectory reward) → compare final accuracy and intermediate reward scores to EXP-2A

### Round 3
- [ ] Generate rollouts from Round 2 model
- [ ] **EXP-3A:** Apply SAPO trajectory RL (same procedure)
- [ ] Evaluate on test sets → record Pass@1
- [ ] Record SAPO intermediate reward scores

### Round 4
- [ ] Generate rollouts from Round 3 model
- [ ] **EXP-4A:** Apply SAPO trajectory RL
- [ ] Evaluate on test sets → record Pass@1
- [ ] Record SAPO intermediate reward scores

**Deliverable:** Accuracy curve across 4 rounds for both EXP trajectory-RL and EXP-2B outcome-only baseline. Intermediate reward trajectory across rounds (this is the key structural result).

---

## Stage 4 — Key Ablation: Fixed vs. Free Unmasking Order

**This tests the core theoretical question: is the denoising trajectory becoming a genuine policy rollout?**

- [ ] **EXP-5A:** Train Round 2 model variant with unmasking order fixed to left-to-right (eliminates ordering as a degree of freedom)
- [ ] **EXP-5B:** Train Round 2 model variant with free unmasking order (current setup, model can unmask any position)
- [ ] Compare: (a) final accuracy, (b) SAPO intermediate reward scores, (c) qualitative inspection of which tokens are unmasked first in each case
- [ ] If EXP-5B > EXP-5A on intermediate rewards, the model is learning to use unmasking order as a reasoning structure → supports trajectory-as-policy hypothesis
- [ ] If no difference, note this as evidence against the hypothesis — the ordering degree of freedom may not be exploitable without architectural changes

**Deliverable:** Single table comparing EXP-5A vs EXP-5B on accuracy and intermediate reward scores. This determines whether Phase 2 (learned critic / value function) is worth pursuing.

---

## Stage 5 — AR Baseline Comparison

**Tests the data efficiency claim from Prabhudesai et al. — the strongest novel angle in the proposal.**

- [ ] Run equivalent STaR + outcome RL loop for 4 rounds using a comparable AR model (Llama-3-8B or Mistral-7B-Instruct)
  - Same datasets, same number of training problems, same number of rounds
- [ ] **EXP-6A:** Plot accuracy vs. round for DLM (our model) vs. AR model
- [ ] **EXP-6B:** Plot training loss across rounds — does AR model show signs of overfitting to training problems faster than DLM?
- [ ] If DLM self-play gap over AR *widens* with more rounds, this is the headline result

**Deliverable:** DLM vs. AR self-play comparison plot across 4 rounds. This is the result that directly validates the core hypothesis motivated by Prabhudesai et al.

---

## Stage 6 — Code Domain Transfer (if time permits)

- [ ] Repeat Rounds 1–2 on HumanEval + MBPP using execution-based verifier
- [ ] Compare gain across rounds on code vs. math — does the self-play loop work similarly across domains?
- [ ] Note: code has a harder cold start (binary pass/fail with no partial credit) — document yield rate in Round 1

**Deliverable:** Code results table. Confirms or limits the generality of findings.

---

## Final Deliverables Checklist

- [ ] Accuracy table: all models (baseline, R1–R4, AR baseline) on all datasets
- [ ] Intermediate reward curve across rounds (the structural evidence for trajectory quality improvement)
- [ ] Ablation table: trajectory reward vs. outcome-only reward vs. fixed unmasking order
- [ ] DLM vs. AR self-play comparison plot
- [ ] Qualitative trajectory examples: show a Round 1 vs. Round 4 denoising trajectory on the same problem — what changed?
- [ ] Write-up of negative results if any (cold start collapse, no improvement across rounds, no ordering effect) — these are publishable findings given the novelty of the setup

---

## Risk Flags — Check These Early

| Risk | Early Signal | Mitigation |
|---|---|---|
| Cold start collapse (Round 1 yield < 15%) | Check after generating 500 rollouts | Use hint-augmented prompting or easier problem subset to bootstrap |
| SAPO reward compute too slow | Profile on 10 examples before full run | Score every 2nd denoising interval instead of all intervals |
| No improvement across rounds | Flat accuracy after Round 2 | Check if training distribution has saturated; add harder problems |
| AR baseline stronger than DLM | EXP-6A shows AR wins every round | Report honestly; investigate whether Prabhudesai result transfers to fine-tuning regime |
