# MS-24: SSL teacher-student runs (B2)

We ran the SSL baseline four times on the short schedule (40k steps, ResNet-50, SAM-Mix, λ_u = 2). Each run changed only how the unsupervised loss is computed in `ssl_loss` (`train_ssl.py`). v4 is the final setting and is now the default in `configs/ssl_s5mars.yaml`.

| Run | Folder | Code | Unsupervised loss | Steps |
|---|---|---|---|---|
| v1 | `v1_original_loss/` | main before PR #20 | only on unlabeled pixels, averaged over unlabeled pixels | 40k |
| v2 | `v2_allpixel_norm/` | `0f4d9f6` | only on unlabeled pixels, averaged over all pixels (AugSeg) | stopped early |
| v3 | `v3_unsup_all_pixels/` | `40edde7` | every pixel gets the teacher target (`unsup_pixels: all`, paper Eq. 3) | stopped at 31,250 |
| v4 | `v4_unsup_all_gt/` | `0feae40` | every pixel, but labeled pixels use the ground truth as the target (`unsup_pixels: all_gt`) | 40k |

v2 is still only on the instance. Its folder gets filled in the next time the instance is on.

## Test results

Supervised short is `results/baseline_r50_short_1xHeadLR` (same schedule and head LR as these runs). "best" means the checkpoint with the best val mIoU, "last" is the step 40k checkpoint.

| Model | val mIoU | test mIoU | test mAcc | sky | ridge | soil | sand | bedrock | rock | rover | trace | hole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Supervised short (B1) | - | 67.75 | 74.80 | 94.0 | 92.8 | 82.1 | 72.8 | 91.4 | 20.3 | 73.8 | 67.3 | 15.2 |
| v1 teacher best (40k) | 54.99 | 57.35 | 68.58 | 93.1 | 91.2 | 71.3 | 73.0 | 84.7 | 7.4 | 52.7 | 33.5 | 9.2 |
| v1 student best (40k) | 56.93 | 59.04 | 69.42 | 93.3 | 91.4 | 75.0 | 74.2 | 86.6 | 9.0 | 57.8 | 34.9 | 9.2 |
| v3 teacher best (27.5k) | 52.02 | 56.26 | 65.86 | 93.8 | 91.9 | 71.0 | 71.9 | 87.1 | 0.0 | 43.9 | 46.7 | 0.1 |
| v3 student best (25k) | 52.74 | 56.89 | 65.09 | 92.5 | 90.7 | 73.4 | 71.3 | 88.3 | 0.0 | 41.1 | 51.1 | 3.7 |
| v4 teacher best (37.5k) | 69.90 | 65.28 | 74.79 | 94.2 | 92.4 | 79.4 | 72.5 | 90.8 | 16.0 | 68.1 | 59.2 | 15.1 |
| v4 student best (40k) | 69.16 | 67.62 | 76.41 | 94.4 | 92.6 | 79.5 | 74.0 | 90.5 | 20.2 | 64.5 | 68.2 | 24.7 |
| v4 teacher last (40k) | 68.95 | 66.99 | 75.79 | 94.3 | 92.5 | 79.3 | 74.1 | 90.5 | 18.4 | 64.3 | 67.8 | 21.7 |

v1 and v2 drifted toward predicting soil, because the unsupervised loss only ever saw the gaps, which are mostly soil and bedrock. v3 fixed the drift but never learned rock, since the teacher's guesses overrode the real labels on labeled pixels. v4 keeps the real label there, and it ties supervised on mIoU and is ahead on mAcc.

## What is in each run folder

- `train/`: `config.yaml` (the exact config logged at launch), `train.log`, `s_metrics.jsonl` and `t_metrics.jsonl` (student and teacher val scores at each eval step), `coverage.jsonl` (pseudo-label class shares in the unlabeled gaps)
- `eval/<checkpoint>/{val,test}/`: `metrics.json`, `per_class.csv`, `confusion.npy` from `evaluate.py`
- `eval/eval.log` (v3, v4): printed output of those evaluations, including easy/hard splits and the rock error breakdown
- `checkpoints/` (v4, local only): the `.pth` files are about 160 MB each, so they are git-ignored and need to go on the team Drive
