# Diffusion RLVR Experiment Checklist

## Phase 0 — Setup & Baselines
> Before any RL training

- [ ] Set up LLaDA-8B inference environment
  - Confirm generation, logit access, and partial-state conditioning all work
- [ ] Select benchmarks
  - Math: MATH500, AMC/AIME subset
  - Code: HumanEval, MBPP
  - Keep math and code results separate throughout all phases
- [ ] Run baseline LLaDA-8B (no RL) on all benchmarks
  - Record pass@1 — this is the absolute floor
- [ ] Run diffu-GRPO on all benchmarks
  - Flat outcome reward, uniform gradient
  - First RL condition in the ablation table
- [ ] Profile rk estimation cost
  - Measure time for K=5 intervals × N=16 rollouts per example
  - Determines feasibility of the full iterative loop

---

## Phase 1 — Signal vs. Gradient Decomposition
> Core ablation table: tests the 2×2 (signal × gradient) decomposition

- [ ] Run diffu-GRPO + ATPO weighting (flat signal, structured gradient)
  - Applies wk to flat outcome A(o)
  - Tests whether gradient structure alone helps, independent of interval rewards
- [ ] Run SAPO alone (interval signal, uniform gradient)
  - Replicate SAPO on your benchmarks as the structured-signal baseline
- [ ] Run SAPO + ATPO (interval signal, structured gradient)
  - Main single-round method — both axes of the 2×2 active
- [ ] Ablation: compare all four conditions on math and code separately

| Condition | Signal | Gradient |
|---|---|---|
| diffu-GRPO | Flat | Uniform |
| diffu-GRPO + ATPO | Flat | Structured |
| SAPO | Interval | Uniform |
| SAPO + ATPO | Interval | Structured |

- [ ] Mechanistic check: identify examples falling in each 2×2 cell
  - For a held-out subset, log (rk, wk) per interval and tag each cell
  - Verify SAPO+ATPO outperforms ablations specifically on the middle cells:
    - High rk / low wk — productive step, model already confident
    - Low rk / high wk — unproductive step, model uncertain

---

## Phase 2 — Policy Structure Diagnostic
> Gates Phase 3. Run before committing compute to the iterative loop.

- [ ] Train Round-2 variant: free unmasking order (standard MDLM)
  - Control condition
- [ ] Train Round-2 variant: left-to-right fixed unmasking order
  - Forces sequential structure
- [ ] Compare wk distribution entropy between the two variants
  - If free-order develops sharper wk distributions and higher rk at specific intervals → trajectory is acquiring genuine policy structure → proceed to Phase 3
  - If not → diagnose before iterating. Architectural changes may be needed.
- [ ] **Decision gate: proceed to Phase 3 only if free-order shows sharper wk**

---

## Phase 3 — Iterative Loop
> Central empirical thesis: DLM–AR performance gap widens monotonically with rounds

- [ ] Run diffu-GRPO + loop (R = 1..4)
  - Control: does iterating with flat reward help?
  - Tests whether loop benefit is objective-agnostic
- [ ] Run SAPO + ATPO + loop (R = 1..4)
  - Full method. Log per-round performance curves on math and code separately.
- [ ] Plot DLM vs. AR performance gap per round
  - The monotonic widening claim is the central thesis — must be shown explicitly, not asserted
- [ ] Re-calibrate α, β thresholds each round via Corr(wk, |rk|) on held-out set
  - Track calibration stability across rounds — flag degradation as a limitation if observed
- [ ] Compare SAPO+ATPO+loop vs. diffu-GRPO+loop
  - If SAPO+ATPO+loop >> diffu-GRPO+loop: the objective is doing real work
  - If similar: the loop is the story, not the objective

---

## Phase 4 — Analysis & Reporting
> After all runs complete

- [ ] Report math and code results separately throughout
  - Do not average across domains
  - Expect slower loop convergence on code due to higher rk variance from partial denoising states
- [ ] Verify "no prior work on iterative MDLM self-improvement" novelty claim
  - Search concurrent preprints before submitting — this is a central novelty claim
- [ ] Write up Phase 2 diagnostic as a standalone finding
  - Informs the community whether denoising trajectories are genuine policy rollouts, independent of the loop results

---

## Notes

- **Phase 2 is a hard gate.** If the trajectory diagnostic fails, running Phase 3 is expensive and hard to interpret.
- **The loop control (diffu-GRPO + loop) is the most important single experiment.** It cleanly separates the contribution of the objective from the contribution of iteration.
- **Start Phase 4 writing during Phase 1.** The ablation decomposition is a standalone contribution and does not depend on loop results.
