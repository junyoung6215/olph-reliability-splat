<div align="center">

<h1>Stratified Retention for Compact<br>Pose-free Feed-forward 3D Gaussian Splatting</h1>

**Kyoosuk Hwang** · **Junyoung Kwon**

서울과학기술대학교 (SeoulTech) · 캡스톤디자인 2026

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#설치)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C?logo=pytorch&logoColor=white)](#설치)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-76B900?logo=nvidia&logoColor=white)](#설치)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[English](README.md) · **한국어**

</div>

<p align="center">
  <img src="docs/figures/hero_qualitative.jpg" alt="RealEstate10K novel view: GT, pruning 없는 base(131K), Gaussian 75/50/25%만 남긴 우리 모델" width="100%">
</p>
<p align="center"><sub>
<b>Pose 없는 이미지 2장</b>으로 만든 novel view(RealEstate10K). 픽셀별 Gaussian을 75 / 50 / 25%만 남겨도 pruning 없는
base와 거의 같은 품질을 유지합니다. 각 이미지 구석의 숫자는 PSNR입니다.
</sub></p>

## 소식

- **2026-10-30** — 캡스톤디자인(2) 최종 발표. Stratified retention과 pruning-aware fine-tuning 코드 공개.
- **2026-06-12** — OLP-H와 Gaussian reliability head (캡스톤디자인(1)).

## 요약

Pose 없는 이미지로 동작하는 feed-forward 3D Gaussian Splatting은 픽셀마다 Gaussian을 하나씩 예측하기 때문에,
256×256 이미지 2장만으로도 13만 개가 넘는 Gaussian이 생기고 그중 상당수가 중복됩니다. 우리는 한 번의 forward pass
안에서 품질 손실 없이 이 중 일부만 남기는 방법을 연구했습니다. 사전학습된 pose-free 모델 위에 Gaussian head feature를
고정 randomized Hadamard 회전으로 정돈하는 **OLP-H**와, Gaussian마다 신뢰도 *r*을 예측해 opacity를 부드럽게 줄이는
**reliability head**를 추가했습니다. 그다음 *r* 기준 전역 top-k만 남기는 자연스러운 방법이 실패한다는 것을 보였습니다.
*r*이 낮은 Gaussian은 공간적으로 뭉쳐 있어서 영역이 통째로 사라집니다(50%에서 14.49 dB, 무작위 선택은 19.05 dB).
**Stratified retention**은 *r* < 0.35인 Gaussian을 먼저 걸러낸 뒤 모든 2×2 cell에서 같은 비율만 남겨 빈 영역을 없애고,
**pruning-aware fine-tuning**은 남은 Gaussian이 빈자리를 덮도록 학습합니다. Gaussian을 절반만 써도 RealEstate10K
(Test 500)에서 25.24 dB로 pruning 없는 base보다 0.15 dB 높고, 렌더링 peak memory는 48% 줄었습니다. 또한 *r*은
중복도가 아니라 신뢰도를 재는 점수라서, pruning 순위보다 필터로 쓰는 편이 낫다는 것을 확인했습니다.

## 방법

<p align="center">
  <img src="docs/figures/architecture.png" alt="구조: pose-free backbone, OLP-H, reliability를 포함한 Gaussian head, 픽셀 정렬 Gaussian, stratified retention, splatting, pruning-aware fine-tuning" width="100%">
</p>

| | 구성 요소 | 하는 일 | 코드 |
| :-: | --- | --- | --- |
| ① | **OLP-H** | 256채널 head feature를 고정 직교행렬 `Q = H·D·P / √C`(Walsh–Hadamard, random sign, channel permutation)로 회전합니다. 학습 파라미터는 없습니다. | [`latent_preconditioner.py`](src/model/encoder/heads/latent_preconditioner.py) |
| ② | **Reliability head** | Gaussian마다 `r = r_min + (1 − r_min)·σ(a)`를 예측하고 opacity `α·r`로 렌더링합니다. Context-residual loss로 학습합니다. | [`dpt_gs_head.py`](src/model/encoder/heads/dpt_gs_head.py), [`loss_reliability.py`](src/loss/loss_reliability.py) |
| ③ | **Stratified retention** | 픽셀 격자를 *k*×*k* cell로 나누고 모든 cell에서 같은 비율 ρ만 남깁니다. `r < 0.35`인 Gaussian을 먼저 빼고 나머지는 무작위로 채웁니다. | [`gaussian_retention.py`](src/model/gaussian_retention.py), [`local_cell_retention.py`](src/model/local_cell_retention.py) |
| ④ | **Pruning-aware fine-tuning** | 학습할 때도 같은 선택을 적용합니다. Gradient는 남은 Gaussian으로만 흐르고, 전체 가중치를 10K step 학습합니다. | [`model_wrapper.py`](src/model/model_wrapper.py) |

Retention은 rasterization **이전에** 적용되므로 renderer는 ρ·131K개의 Gaussian만 처리합니다. 모든 구성 요소는 기본값이
OFF이며, 기본 config에서는 사전학습 base 모델과 동일하게 동작합니다.

## 실험 결과

모든 수치는 RealEstate10K 256×256, context view 2장 기준입니다. *Base*는 ① + ②를 적용하고 Gaussian을 모두 쓴 모델,
*Control*은 같은 base를 같은 step 동안 pruning 없이 fine-tuning한 대조군입니다.

### 주요 결과 — RealEstate10K Test 500

<p align="center">
  <img src="docs/figures/quality_vs_gaussians.png" alt="렌더링한 Gaussian 수에 따른 PSNR과 LPIPS" width="100%">
</p>

| 모델 | 남긴 비율 | #Gaussian | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | :-: | :-: | :-: | :-: | :-: |
| Base | 100% | 131K | 25.09 | 0.824 | 0.160 |
| Control (FT 10K, pruning 없음) | 100% | 131K | *25.71* | *0.843* | *0.151* |
| **Ours** | 75% | 98K | **25.49** | **0.837** | **0.157** |
| **Ours** | 50% | 66K | 25.24 | 0.830 | 0.166 |
| **Ours** | 25% | 33K | 24.84 | 0.817 | 0.182 |

50%에서 렌더링 peak memory는 **167 MB → 87 MB(−48%)**입니다. Control은 기울임체로 표시했습니다. 같은 학습량에서
pruning을 하지 않은 대조군은 우리 50% 모델보다 여전히 0.47 dB 높습니다.

### 전체 RealEstate10K test set — 5,601 scene

| 모델 | #Gaussian | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
| --- | :-: | :-: | :-: | :-: |
| 사전학습 pose-free base | 131K | 25.06 | 0.839 | 0.161 |
| + OLP-H (300K step fine-tune) | 131K | 23.80 | 0.794 | 0.181 |
| + Reliability head (+75K step) | 131K | 24.84 | 0.824 | 0.163 |
| **+ Stratified retention + FT (ours, 50%)** | **66K** | 25.01 | 0.829 | 0.169 |
| Control: FT 10K, pruning 없음 | 131K | 25.45 | 0.842 | 0.155 |

첫 단계의 행들은 학습 schedule이 서로 달라서, 단계별 누적 결과로 보면 됩니다.

### Ablation — 왜 stratified retention과 fine-tuning인가

<p align="center">
  <img src="docs/figures/retention_progression.jpg" alt="한 장면에서 Gaussian 50%: 전체, r 기준 전역 top-k, stratified, stratified + pruning-aware FT" width="100%">
</p>

<p align="center">
  <img src="docs/figures/ablation_path.png" alt="구성 요소를 하나씩 더할 때 50%에서의 PSNR" width="100%">
</p>

| 설정 (50% 유지, validation 64 scene) | PSNR ↑ | Δ | LPIPS ↓ |
| --- | :-: | :-: | :-: |
| Base, 100% (pruning 없음) | 25.55 | – | 0.157 |
| *r* 기준 전역 top-k, FT 없음 | 14.49 | −11.06 | 0.389 |
| + Stratified 2×2 | 19.42 | +4.93 | 0.353 |
| + Pruning-aware FT 2K | 25.49 | +6.07 | 0.165 |
| + *r* 필터 (*r* < 0.35) + 무작위 채움 | 25.62 | +0.13 | 0.161 |
| + FT 10K (**최종**) | **25.80** | +0.18 | **0.158** |

FT 없이 대조군에 같은 선택만 적용하면 19.56 dB에 그치므로, pruning-aware fine-tuning은 필수입니다.

### *r*은 좋은 pruning 점수인가?

<p align="center">
  <img src="docs/figures/cell_size_sweep.png" alt="cell 크기 2~16에서 r 순위와 무작위 선택의 PSNR" width="100%">
</p>

Cell이 클수록 *r* 순위는 무작위 선택보다 더 뒤처집니다(FT 후 −0.14 → −0.86 dB, FT 없이 −0.53 → −5.02 dB). *r*이 낮은
Gaussian은 공간적으로 뭉쳐 있고, *r*은 floater 억제용으로 학습됐지 Gaussian을 뺐을 때의 영향을 예측하도록 학습되지
않았기 때문입니다. Budget loss로 *r*을 pruning용으로 학습하면 같은 모델의 무작위 선택보다 0.12–0.32 dB 좋지만, FT를
거친 무작위 선택(25.51 dB)보다는 0.3 dB 낮습니다. 그래서 최종 모델은 *r*을 필터로만 사용합니다.

<details>
<summary><b>Cell 크기 sweep 표</b></summary>

| Cell | *r* 순위 (FT 없음) | 무작위 (FT 없음) | *r* 순위 (FT 2K) | 무작위 (FT 2K) | *r* 필터 + 무작위 (FT 2K) |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 2×2 | 19.42 | 19.95 | 25.49 | 25.63 | 25.62 |
| 4×4 | 17.88 | 19.50 | 25.15 | 25.51 | 25.49 |
| 8×8 | 15.69 | 19.20 | 24.59 | 25.45 | – |
| 16×16 | 14.09 | 19.11 | – | – | – |

</details>

<details>
<summary><b>첫 단계(OLP-H, reliability) 분석</b></summary>
<br>

OLP-H는 high-opacity Gaussian 비율과 floater proxy(high opacity이면서 scale이 큰 Gaussian 비율)를 낮추고, 재학습 없이
렌더링 시점에 Gaussian을 뺄 때 품질이 더 천천히 떨어집니다.

<img src="docs/figures/gaussian_compactness.png" alt="Active Gaussian 비율과 floater proxy" width="100%">
<img src="docs/figures/budget_curve.png" alt="렌더링 시점 opacity top-K에서의 PSNR 하락과 LPIPS" width="100%">

OLP-H 출력은 채널 집중도가 높지만, 장면 단위 floater proxy와의 상관은 약합니다.

<img src="docs/figures/channel_concentration.png" alt="OLP score와 채널 집중도" width="100%">
<img src="docs/figures/channel_outlier_maps.png" alt="한 장면의 channel outlier map" width="100%">

원본 metric: [`results/reliability_residual/`](results/reliability_residual/README.md) ·
분석 노트: [`experiment_records/`](experiment_records/README.md)
</details>

## 설치

Python 3.10, PyTorch 2.1.2, CUDA 11.8에서 확인했습니다.

```bash
git clone https://github.com/junyoung6215/olph-reliability-splat
cd olph-reliability-splat

# conda 환경 생성 및 CUDA 11.8용 Gaussian rasterizer 빌드
bash scripts/create_cuda118_env.sh
conda activate noposplat-olp-cuda118
```

## 데이터

RealEstate10K를 [DATASETS.md](DATASETS.md)의 chunk 형식으로 사용합니다. 평가 index는 [`assets/`](assets/)에 있으며,
`evaluation_index_re10k.json`이 전체 test split입니다.

## Checkpoint

사전학습 pose-free base checkpoint는 다음 명령으로 받습니다.

```bash
bash scripts/download_re10k_checkpoint.sh      # -> pretrained_weights/re10k.ckpt
```

| 모델 | 설명 | 다운로드 |
| --- | --- | :-: |
| OLP-H + Reliability | 1단계 (① + ②) | 공개 예정 |
| Ours, ρ = 50% | 2단계 (③ + ④) | 공개 예정 |

## 학습

학습은 GPU 1장에서 두 단계로 진행합니다.

**1단계 — OLP-H와 reliability head (① + ②)**

```bash
# 사전학습 base에서 OLP-H fine-tuning
CUDA_VISIBLE_DEVICES=0 bash scripts/run_re10k_olp_ft.sh

# context-residual loss로 reliability head 학습 (75K step)
CUDA_VISIBLE_DEVICES=0 BASE_CKPT=/path/to/olph.ckpt DATA_ROOT=/path/to/re10k \
bash scripts/run_re10k_reliability_residual_ft.sh
```

**2단계 — stratified retention + pruning-aware fine-tuning (③ + ④)**

[`config/experiment_runs/retention_suite.yaml`](config/experiment_runs/retention_suite.yaml)을 복사해 경로를 채우고,
최종 retention 설정을 넣습니다.

```yaml
initial_checkpoint: /path/to/olph_reliability.ckpt   # 1단계 결과
reliability_floor: 0.35                              # r 필터
local:
  cell_size: 2                                       # 2x2 cell
  stratified: true
  demand_source: uniform
```

```bash
# 실행할 명령을 출력만 합니다. 실제 실행은 --execute
python scripts/run_retention.py --config my_retention.yaml train \
    --arm control --prune --local --score reliability_filtered --ratio 0.5 --steps 10000 \
    --run-id ours_r50 --execute
```

`--ratio`는 남길 비율 ρ입니다. `--train-scope heads`를 주면 backbone을 고정해 짧은 pilot을 돌릴 수 있습니다.

## 평가

```bash
# 전체 RealEstate10K test split. --split test는 YAML에 지정한 500-scene held-out index
python scripts/run_retention.py --config my_retention.yaml eval-grid \
    --checkpoint /path/to/ours_r50.ckpt \
    --scores reliability_filtered --ratios 0.25 0.5 0.75 --local on --split full --execute

# 모든 run 결과를 CSV 하나로 정리
python scripts/run_retention.py --config my_retention.yaml summarize \
    --root /path/to/outputs --output results.csv
```

단위 테스트는 데이터 없이 CPU에서 실행됩니다.

```bash
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

## 코드 구조

```text
src/model/encoder/heads/latent_preconditioner.py   ① OLP-H
src/model/encoder/heads/dpt_gs_head.py             ② Gaussian head + reliability branch
src/loss/loss_reliability.py                       ② context-residual reliability loss
src/model/gaussian_retention.py                    ③ retention: budget, 점수, r 필터, budget loss
src/model/local_cell_retention.py                  ③ cell 배치와 stratified 순위
src/model/model_wrapper.py                         ④ 학습·평가에서의 retention 적용
src/experiment_runner/retention.py                 train / eval-grid / summarize 실행기
config/                                            Hydra config (모든 구성 요소 기본 OFF)
docs/                                              설계 노트와 구현 보고서
```

## 한계와 향후 과제

- 50% 모델은 같은 step 동안 pruning 없이 학습한 대조군보다 여전히 0.44–0.47 dB 낮습니다.
- 50% 이하에서는 LPIPS가 pruning 없는 base보다 약간 나쁘며, 얇은 구조와 고주파 텍스처에서 손실이 가장 큰 것으로
  보입니다(분석 중).
- *r*은 신뢰도 점수이지 pruning 점수가 아닙니다. 다음 과제는 pruning 전용 점수, 실제 저장 크기(MB)·FPS 측정, 더 많은
  입력 view, 다른 backbone입니다.

## 인용

```bibtex
@misc{hwang2026stratified,
  title        = {Stratified Retention for Compact Pose-free Feed-forward 3D Gaussian Splatting},
  author       = {Hwang, Kyoosuk and Kwon, Junyoung},
  year         = {2026},
  howpublished = {\url{https://github.com/junyoung6215/olph-reliability-splat}},
  note         = {Capstone Design project, Seoul National University of Science and Technology}
}
```

## 감사의 말

이 코드는 pose-free backbone, 데이터 파이프라인, 사전학습 checkpoint를 제공하는
[NoPoSplat](https://github.com/cvg/NoPoSplat)과, 그 기반인 [pixelSplat](https://github.com/dcharatan/pixelsplat),
[DUSt3R](https://github.com/naver/dust3r), [CroCo](https://github.com/naver/croco)를 바탕으로 합니다. 코드를 공개해 준
저자들께 감사드립니다.

## 라이선스

[MIT License](LICENSE)로 공개합니다. 기반 코드는 © 2024 Botao Ye, 우리 확장 부분은 © 2026 Kyoosuk Hwang, Junyoung Kwon입니다.
