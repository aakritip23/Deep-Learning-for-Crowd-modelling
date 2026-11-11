# Research notes – beating the Social LSTM baseline on TrajNet

_Round 1, 2026-11-11. Numbers marked (paper) were read from the papers; (search) come from search snippets and are unverified._

_These notes were written before the model was built (research and decision phase). What was built and measured is in [`../03_improved_model/RESULTS.md`](../03_improved_model/RESULTS.md) and [`../04_results/`](../04_results/)._

## 1. How the baseline score is defined
- TrajNet (2018 challenge) ranks by **Avg = (MAD + FAD) / 2**; MAD = ADE (mean over 12 steps), FAD = FDE (last step), in metres.
- The professor's README row "Social LSTM: average 1.3865, final 2.098, mean 0.675" fits this exactly:
  (2.098 + 0.675) / 2 = 1.3865. So **Final = FAD/FDE, Mean = MAD/ADE, Average = ranking score.**
- Target to beat: **Avg 1.3865 / FAD 2.098 / MAD 0.675**.
- TrajNet scores a **single deterministic prediction** per pedestrian (no best-of-K) - the server cannot score K samples.

## 2. Where the baseline sits on the TrajNet leaderboard (Table I of the Transformer paper, accessed 2020; Avg / FAD / MAD)
| Method | Avg | FAD | MAD | Uses neighbours? |
|---|---|---|---|---|
| Transformer (TF, Giuliari 2020) | 0.776 | 1.197 | 0.356 | no |
| REDv3 (RNN-encoder + MLP) | 0.781 | 1.201 | 0.360 | no |
| RED | 0.798 | 1.229 | 0.366 | no |
| SR-LSTM | 0.816 | 1.261 | 0.370 | yes |
| Social Force (EWAP) | 0.819 | 1.266 | 0.371 | yes |
| MX-LSTM | 0.887 | 1.374 | 0.399 | yes |
| Linear interpolation (constant velocity) | 0.894 | 1.359 | 0.429 | no |
| LSTM | 1.140 | 1.793 | 0.491 | no |
| S-GAN | 1.334 | 2.107 | 0.561 | yes |
| **Prof's Social LSTM (README)** | **1.3865** | **2.098** | **0.675** | yes |
| "social lstm v2" / original (RED paper) | 1.563 / 2.874 | 2.299 / 4.323 | 0.826 / 1.424 | yes |
- AMENet (CVAE + attentive neighbour maps) reports FDE 1.183 / MAD 0.356 on TrajNet (search; unverified) -> Avg ~0.77.
- Top-4 entries at the time were all **individual** (no social context) models.

## 3. Findings that shape the project
1. **Headroom is large**: even constant velocity (Avg 0.894) beats the baseline's 1.3865. Realistic target = **~0.78 (best published)**, i.e. roughly -45 % vs baseline. The baseline is probably under-performing because of its pipeline (data transform, short training, NLL sampling) more than its idea.
2. **Input normalisation matters most.** RED paper: absolute coordinates give FAD 4.32, relative/velocity gives 2.10 (same as what the professor found); standardising speeds (mean/std from train) helps; orientation normalisation used by RED.
3. **Deterministic, point-estimate decoding** (L2 / Huber loss, predict 12 steps directly - MLP head or Transformer) is the winning recipe for this benchmark. The baseline samples from a Gaussian step by step, which accumulates error and is noisy for a single-shot metric.
4. **Social modelling gave little on this metric.** Social models rank below the best individual models. So: good encoder + good normalisation + augmentation first, then add a light social module and *measure* whether it helps (ablation).
5. **Augmentation**: random scaling (0.5-2x), rotation, flips, noise; the Transformer paper used scaling aug for its quantised variant.
6. **Multimodal heads (CVAE/diffusion)** are the 2023-2026 trend (Social-STGCNN, Trajectron++, PECNet, MID, Mamba variants; self-reported minADE/minFDE@20 on ETH/UCY around 0.2/0.3), but they are evaluated best-of-20 and **do not directly help a single-prediction TrajNet score**. Use them as a secondary track (report minADE/minFDE@20 + a mean/medoid single prediction for TrajNet).
7. **Dataset caveats** (from SDD analysis literature + our EDA): Stanford Drone makes up most of the data (60/74 train files), mixing pedestrians, bikes, cars -> speeds differ greatly by source. Source/scene-specific normalisation or a source embedding is worth testing.

## 4. Candidate designs (to choose from)
| Option | Idea | Cost | Expected on TrajNet | Notes |
|---|---|---|---|---|
| A. RED-style / GRU encoder + MLP decoder | Standardised relative velocities, predict all 12 offsets at once | low | ~0.78-0.80 (published) | strong, quick baseline of "ours" |
| B. Transformer encoder (TF) | Self-attention over the 8 observed steps, 12-step output, warm-up + Adam | low-med | ~0.776 (published) | also copes with missing frames |
| C. B + social attention | Cross-attention over neighbours' encodings (replaces grid pooling) | med | maybe 0.75-0.78 (to test) | the "improved Social" model; needs ablation |
| D. C + multimodal head | CVAE / goal-conditioned (PECNet-style) or diffusion | med-high | better minADE@20, single-pred similar | secondary track |
| E. Ensemble / test-time augmentation of B/C | Average predictions over rotations/seeds | low | typically -2-5 % error | cheap final boost |

## 5. Recommended plan
1. Build the evaluation harness + constant-velocity and vanilla baselines (task 1).
2. Reproduce the Social LSTM number, then implement **A -> B** with proper normalisation and augmentation; expect to beat 1.3865 by a wide margin already here.
3. Add **C** (neighbour attention) and compare against B; keep it only if it improves held-out ADE/FDE.
4. Optional **D/E** as extra tracks.

## 5b. Round 2 – measured on our validation split (script: constant velocity, 11,403 trajectories)
| Source | n | CV last-step velocity (ADE / FDE / Avg) | CV mean-observed velocity (ADE / FDE / Avg) |
|---|---|---|---|
| ALL | 11403 | 1.540 / 2.842 / 2.191 | 0.843 / 1.569 / **1.206** |
| biwi | 145 | 1.404 / 2.539 / 1.971 | 0.513 / 0.896 / 0.705 |
| crowds | 2211 | 1.421 / 2.628 / 2.024 | 0.780 / 1.480 / 1.130 |
| mot | 107 | 1.367 / 2.559 / 1.963 | 0.827 / 1.639 / 1.233 |
| stanford | 8940 | 1.574 / 2.903 / 2.238 | 0.864 / 1.601 / 1.232 |
- Our validation split is **not comparable to the TrajNet test leaderboard** (constant velocity scores 0.894 there, 1.206 here; 78 % of validation is Stanford Drone). The professor's baseline must be re-scored on the same split before any claim of improvement.
- The noisy last-step velocity is much worse than the mean velocity -> smoothing/standardising the observed motion is important for any learned model.

## 5c. Decision: improve the existing model or build a newer one?
| Path | What | Ceiling on TrajNet-style single-prediction score | Cost / risk | Verdict |
|---|---|---|---|---|
| 1. Refine Social LSTM | velocity+rotation inputs (already done), standardised inputs, L2/Huber multi-step head instead of Gaussian sampling, directional or attention pooling | ~0.80-0.85 (SR-LSTM, best LSTM-social on leaderboard: 0.816) | low | good *ablation / intermediate*, not the final model |
| 2. Individual Transformer (TF) | self-attention over 8 observed steps, 12 outputs | ~0.776 (published best on TrajNet) | low-med | strong core |
| 3. **Social Transformer (TF + neighbour cross-attention + TTA/ensemble)** | path 2 plus learned attention over neighbours replacing grid pooling | ~0.75-0.78 expected, interaction gain must be proven by ablation | med | **recommended** |
| 4. CVAE / goal-conditioned (PECNet, AMENet) | latent futures, endpoint conditioning; AMENet also needs semantic maps (we have none) | similar single-pred; better best-of-K | med-high | secondary track |
| 5. Diffusion / Mamba (MID, LED, GeoTDM, MMFlow, SGM) | multimodal generative; best-of-20 minADE/FDE | no clear gain on a single-prediction metric; slow sampling (LED needs a special initializer to be fast) | high | out of scope for main track |

## 6. Sources
- Giuliari et al., Transformer Networks for Trajectory Forecasting: https://arxiv.org/abs/2003.08111
- Becker et al., RED: a simple but effective baseline predictor for the TrajNet benchmark: https://arxiv.org/abs/1805.07663
- Kothari et al., Human Trajectory Forecasting in Crowds (TrajNet++): https://arxiv.org/abs/2007.03639
- AMENet (Cheng et al.): https://arxiv.org/abs/2006.08264
- Social-STGCNN: https://arxiv.org/abs/2002.11927 ; Social GAN: https://arxiv.org/abs/1803.10892
- 2025-26 trends (self-reported, unverified): TOTP (ICCV 2025), MMFlow, Social Graph Mamba (arXiv 2608.21411), GeoTDM (arXiv 2410.13027), review arXiv 2506.14831
- TrajNet challenge: https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge
