<div align="center">

# OLP-H + Gaussian Reliability

### Floater-aware, budget-robust Gaussians for pose-free feed-forward 3D Gaussian Splatting

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#getting-started)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch&logoColor=white)](#getting-started)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](#getting-started)
[![Base](https://img.shields.io/badge/built%20on-NoPoSplat-6E56CF)](https://github.com/cvg/NoPoSplat)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**English** · [한국어](README_KO.md)

Capstone Design project · SeoulTech · 2026<br>
Junyoung Kwon · Kyoosuk Hwang

</div>

---

<p align="center">
  <img src="docs/figures/teaser.jpg" alt="Novel views on RealEstate10K: context view, ground truth, NoPoSplat, OLP-H, and OLP-H with the reliability head, with per-image PSNR" width="100%">
  <br>
  <sub>Novel views from two unposed RealEstate10K images. Per-image PSNR below each render; the best of the three models is highlighted.</sub>
</p>

Feed-forward 3DGS models such as [NoPoSplat](https://github.com/cvg/NoPoSplat) reconstruct a scene from
two **unposed** images in a single forward pass. They predict one Gaussian per pixel, so every Gaussian is
rendered with the same trust — including the floaters and low-confidence primitives that hurt novel views.

This project adds two lightweight modules on top of NoPoSplat's Gaussian head:

- **OLP-H (Orthogonal Latent Preconditioner, Hadamard)** — a fixed randomized Hadamard rotation of the
  head feature that reshapes the predicted Gaussian distribution toward fewer high-opacity primitives.
- **Gaussian Reliability Head** — a per-Gaussian gate `r ∈ [r_min, 1]` that softly scales opacity, trained
  with a weak *context-residual* signal so unreliable Gaussians fade out instead of being hard-pruned.

## Highlights

| | |
| --- | --- |
| 🎯 **3.6× lower floater proxy** | OLP-H reduces the high-opacity × large-scale Gaussian ratio from 2.95e-4 to 8.16e-5 (300 matched scenes). |
| 📉 **Fewer active Gaussians** | Active ratio at opacity > 0.3 drops from 0.621 to 0.534 at the same generated count. |
| 🪶 **Budget-robust** | Keeping only the top-75% Gaussians costs NoPoSplat −6.9 dB PSNR but OLP-H only −4.5 dB. |
| 🔁 **Reliability recovers quality** | The reliability head lifts OLP-H by **+1.04 dB PSNR** and lowers LPIPS 0.181 → 0.163 on the full RE10K test set. |

## Method

<p align="center">
  <img src="docs/figures/pipeline.svg" alt="Pipeline: NoPoSplat backbone, OLP-H preconditioner, Gaussian and reliability heads, opacity gating, and the training-only context residual loss" width="100%">
</p>

**OLP-H.** Before the Gaussian parameter head, each 256-channel DPT feature vector `x` is rotated by a fixed
orthogonal matrix

```text
Q x = H · D · P · x / √C        (C = 256)
```

where `P` is a seeded channel permutation, `D` a random ±1 diagonal, and `H` the Walsh–Hadamard transform.
`Q` is norm-preserving and has no learnable parameters; during fine-tuning it is blended in from identity
(`mix: 0 → 1` over a warm-up) so the pretrained head can adapt.
→ [`src/model/encoder/heads/latent_preconditioner.py`](src/model/encoder/heads/latent_preconditioner.py)

**Gaussian Reliability Head.** A two-layer conv branch (3×3 → ReLU → 1×1) on the same preconditioned feature
predicts a logit `a_g` per Gaussian:

```text
r_g      = r_min + (1 − r_min) · σ(a_g)        r_min = 0.05, initialised so r_g ≈ 0.95
α′_g     = α_g · r_g
```

Nothing is deleted — the gate only changes each Gaussian's rendering contribution, so the model stays
fully feed-forward. → [`src/model/encoder/heads/dpt_gs_head.py`](src/model/encoder/heads/dpt_gs_head.py)

**Context-residual reliability loss.** During training, the predicted Gaussians are also rendered at the
*input* cameras. The per-pixel residual `|Î_ctx − I_ctx|` is sampled at each projected Gaussian center as a
weak risk signal, and

```text
L_rel = mean_g [ r_g · sg(α_g) · sg(risk_g) ]          λ = 1e-4, linear warm-up 3k → 8k steps
```

Stop-gradients keep the original opacity / scale / color predictions intact, so the gradient flows only
into the reliability gate. → [`src/loss/loss_reliability.py`](src/loss/loss_reliability.py)

## Results

All experiments use RealEstate10K at 256×256 with two context views.

### Full RE10K evaluation — 5,601 test pairs, pose-aligned

| Model | Training | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | ---: | ---: | ---: |
| NoPoSplat (official checkpoint) | — | **25.06** | **0.839** | **0.161** |
| OLP-H | 300k-step fine-tune from NoPoSplat | 23.80 | 0.794 | 0.181 |
| **OLP-H + Reliability** | +75k steps with `L_rel` | 24.84 | 0.824 | 0.163 |
| NoPoSplat + transferred reliability head | no fine-tuning | 24.76 | 0.837 | 0.164 |

The reliability head recovers most of the quality lost by OLP-H (+1.04 dB, LPIPS within 0.002 of NoPoSplat)
while keeping the OLP-H Gaussian distribution.
Raw metrics and run configs: [`results/reliability_residual/`](results/reliability_residual/README.md)

### Gaussian compactness — 300 matched scenes

<p align="center">
  <img src="experiment_records/home_data_olp_h_300_20260611/extra/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/labmeeting_report_assets/01_active_gaussian_compactness.png" alt="Active Gaussian ratios and floater proxy, NoPoSplat vs OLP-H" width="85%">
</p>

| Metric (mean over scenes) | NoPoSplat | OLP-H |
| --- | ---: | ---: |
| Opacity mean | 0.391 | **0.347** |
| Active ratio @ α > 0.10 | 0.897 | **0.842** |
| Active ratio @ α > 0.30 | 0.621 | **0.534** |
| Active ratio @ α > 0.50 | 0.328 | **0.269** |
| Floater proxy (high α ∧ large scale) | 2.95e-4 | **8.16e-5** |

### Robustness under a Gaussian budget

Render-time top-K pruning by opacity (no retraining, no pose alignment, same 300 scenes).

<p align="center">
  <img src="experiment_records/home_data_olp_h_300_20260611/extra/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/labmeeting_report_assets/02_gaussian_cap_curve_psnr_lpips.png" alt="PSNR and LPIPS as the retained Gaussian budget shrinks" width="85%">
</p>

| Gaussians kept | 100% | 75% | 50% | 25% | 10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoPoSplat ΔPSNR | 0.00 | −6.90 | −12.22 | −15.79 | −17.46 |
| OLP-H ΔPSNR | 0.00 | **−4.52** | **−10.53** | **−14.45** | **−16.23** |

### Qualitative — where the errors go

Per-pixel error maps (brighter = larger error). OLP-H alone adds error along roof edges and furniture
boundaries; in these scenes the reliability head removes much of it.

<p align="center">
  <img src="docs/figures/error_maps.jpg" alt="Ground truth and per-pixel error maps for NoPoSplat, OLP-H and OLP-H with reliability" width="100%">
</p>

<details>
<summary><b>More scenes</b> (context · ground truth · NoPoSplat · OLP-H · OLP-H + Reliability, error maps below)</summary>
<br>

<img src="results/reliability_residual/qualitative_examples/scene_4f7ce11656b1bf7b_frame000008_side_by_side.png" alt="Living room" width="100%">
<img src="results/reliability_residual/qualitative_examples/scene_02b618a34bc12ff9_frame000015_side_by_side.png" alt="Aerial rooftops" width="100%">
<img src="results/reliability_residual/qualitative_examples/scene_abf2836f83d04375_frame000074_side_by_side.png" alt="Lake" width="100%">
<img src="results/reliability_residual/qualitative_examples/scene_f22e31c9b9bd3320_frame000148_side_by_side.png" alt="Dining room, low overlap" width="100%">
<img src="results/reliability_residual/qualitative_examples/scene_d59ce927e3434d0e_frame000096_side_by_side.png" alt="Covered patio, low overlap" width="100%">

The last two are low-overlap pairs (0.22–0.23) where all three models struggle.

</details>

### Analysis — what OLP-H does to the latent

Measuring the head feature at the same insertion point, OLP-H output is far more channel-concentrated
than NoPoSplat's identity path: in a typical scene a single channel is the strongest one at 74% of pixels
(vs. 28%), and the OLP score (max channel / RMS) nearly doubles.

<p align="center">
  <img src="experiment_records/home_data_olp_h_300_20260611/extra/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/labmeeting_report_assets/10_channel_baseline_vs_olp_summary.png" alt="Channel concentration statistics, NoPoSplat identity vs OLP-H output" width="80%">
</p>

<details>
<summary><b>Channel outlier maps</b> for the scene with the highest OLP score</summary>
<br>
<img src="experiment_records/home_data_olp_h_300_20260611/extra/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/labmeeting_report_assets/05_channel_outlier_example_maps.png" alt="OLP score, top-1 channel and top-k energy maps" width="100%">
</details>

### Takeaways and limitations

- OLP-H alone is **not** a quality win on the standard benchmark (−1.25 dB vs. the official NoPoSplat
  checkpoint); its value is in how the Gaussians are distributed and how gracefully they can be pruned.
- The floater proxy is a heuristic (opacity and scale tails), not ground-truth floater labels.
- OLP-H outputs show strong channel concentration (a few channels dominate most pixels), but scene-level
  correlation with the floater proxy is weak (r ≈ 0); causality is still open
  ([analysis](experiment_records/home_data_olp_h_300_20260611/extra/home/data/olp_h_re10k_slot0/noposplat_eval/re10k/olp_h_300_20260611/analysis/labmeeting_20260612_olp_h_analysis.md)).
- NoPoSplat numbers come from the official checkpoint trained with a larger batch; our fine-tunes ran on a
  single GPU (batch 4, gradient accumulation).
- Checkpoints are not released yet.

## Repository layout

```text
src/model/encoder/heads/
  latent_preconditioner.py   OLP-H module (randomized Hadamard, warm-up mix, latent stats)
  preconditioner_*.py        FWHT math, config, channel-outlier statistics
  dpt_gs_head.py             Gaussian head + reliability branch + opacity gating
  reliability_cfg.py         reliability head config
src/loss/
  loss_reliability.py        context-residual reliability loss (L_rel)
src/model/gaussian_retention.py   render-time Gaussian budget / top-K retention
src/analysis/                OLP-H statistics recorder used in evaluation
config/                      Hydra configs (OLP-H and reliability are off by default)
scripts/                     reproducible train / eval / analysis launchers
tests/                       unit tests for OLP-H, reliability loss, retention
results/, experiment_records/   curated metrics, configs, figures from the runs above
docs/                        protocol, implementation diff vs. upstream, server runbook
```

Every addition is opt-in: with default configs the code behaves exactly like upstream NoPoSplat.
A file-by-file diff against upstream is in [`docs/OLP_H_DIFF_REPORT_EN.md`](docs/OLP_H_DIFF_REPORT_EN.md).

## Getting started

```bash
# 1. Environment (PyTorch 2.1.2 + CUDA 11.8, builds diff-gaussian-rasterization)
bash scripts/create_cuda118_env.sh
conda activate noposplat-olp-cuda118

# 2. Official NoPoSplat RE10K checkpoint -> pretrained_weights/re10k.ckpt
bash scripts/download_re10k_checkpoint.sh

# 3. Dataset: see DATASETS.md (RealEstate10K)
```

**Fine-tune OLP-H, then the reliability head**

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_re10k_olp_ft.sh

CUDA_VISIBLE_DEVICES=0 \
BASE_CKPT=/path/to/olp_h_300k.ckpt DATA_ROOT=/path/to/re10k \
bash scripts/run_re10k_reliability_residual_ft.sh
```

**Evaluate** (full RE10K test split)

```bash
CUDA_VISIBLE_DEVICES=0 \
BASE_CKPT=/path/to/olp_h_300k.ckpt REL_CKPT=/path/to/olp_h_reliability_75k.ckpt \
DATA_ROOT=/path/to/re10k \
bash scripts/run_re10k_reliability_residual_eval.sh
```

**Unit tests** (no dataset needed)

```bash
python tests/test_olp_preconditioner.py
python tests/test_reliability_loss.py
```

Upstream installation, dataset and checkpoint details: [`docs/NOPOSPLAT_UPSTREAM_README.md`](docs/NOPOSPLAT_UPSTREAM_README.md).

## Further reading

| Document | Content |
| --- | --- |
| [README_OLP_H_EN.md](README_OLP_H_EN.md) | OLP-H settings and run recipes |
| [README_GAUSSIAN_RELIABILITY_KO.md](README_GAUSSIAN_RELIABILITY_KO.md) | Reliability head design notes and loss roadmap (Korean) |
| [docs/OLP_H_EXPERIMENT_PROTOCOL_KO.md](docs/OLP_H_EXPERIMENT_PROTOCOL_KO.md) | Full experiment protocol (Korean) |
| [docs/OLP_H_DIFF_REPORT_EN.md](docs/OLP_H_DIFF_REPORT_EN.md) | What changed relative to NoPoSplat |

## Team

**Junyoung Kwon** ([@junyoung6215](https://github.com/junyoung6215)) · **Kyoosuk Hwang**<br>
Capstone Design, Seoul National University of Science and Technology (SeoulTech), 2026.

## Acknowledgements

This project is built on [NoPoSplat](https://github.com/cvg/NoPoSplat) by Ye et al., which in turn builds on
[pixelSplat](https://github.com/dcharatan/pixelsplat), [DUSt3R](https://github.com/naver/dust3r) /
[MASt3R](https://github.com/naver/mast3r) and [CroCo](https://github.com/naver/croco). We thank the authors
for releasing their code and models.

```bibtex
@article{ye2024noposplat,
  title   = {No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images},
  author  = {Ye, Botao and Liu, Sifei and Xu, Haofei and Li, Xueting and Pollefeys, Marc and Yang, Ming-Hsuan and Peng, Songyou},
  journal = {arXiv preprint arXiv:2410.24207},
  year    = {2024}
}
```

## License

MIT — see [LICENSE](LICENSE). The original NoPoSplat code is © 2024 Botao Ye; the OLP-H and reliability
extensions are © 2026 Junyoung Kwon and Kyoosuk Hwang.
