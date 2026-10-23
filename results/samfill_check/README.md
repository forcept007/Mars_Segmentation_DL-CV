# SAM-guided gap filling: offline checks (proposed method 3)

These are small checks we ran before committing to the method. Nothing here trains a model. The idea is to fill each unlabeled gap pixel with the majority ground-truth class of the labeled pixels in the same SAM segment, if that segment is at least τ pure. All runs use the precomputed SAM masks (`~/data/sam_masks` on the instance).

| File | What it checks | Data | Run on |
|---|---|---|---|
| `full_patch.json` | hide random square patches of labeled pixels, fill them from SAM segments, compare with the hidden labels | 5,000 train images | CPU |
| `full_boundary.json` | same, but hide labeled pixels within 8 px of a class boundary (closer to where real gaps are) | 5,000 train images | CPU |
| `val_full.json` | SAM fill against the v4 SSL teacher on hidden boundary pixels, β = 0 and 1 | 200 val images | GPU |
| `val_beta.json` | β sweep (0, 0.25, 0.5, 0.75, 1) for mixing the teacher's probabilities into the segment vote | 200 val images | GPU |

Each JSON has the raw confusion matrices (rows are the true class, in the order sky, ridge, soil, sand, bedrock, rock, rover, trace, hole), so every number below can be recomputed from them.

## Boundary test on train (`full_boundary.json`, all SAM masks)

| τ | Hidden pixels filled | Correct | Rock correct | Hole correct | Real gaps filled |
|---|---|---|---|---|---|
| 0.6 | 72% | 93.0% | 94.6% | 27.5% | 47% |
| 0.7 | 71% | 93.7% | 94.9% | 27.5% | 45% |
| 0.8 | 69% | 94.7% | 95.2% | 31.6% | 42% |
| 0.9 | 65% | 96.1% | 95.7% | 42.0% | 37% |
| 1.0 | 53% | 99.1% | 98.7% | 94.0% | 18% |

The patch test is easier (98 to 100% correct), since random patches are mostly inside one region.

## SAM fill vs. the v4 teacher on val (`val_full.json`, β = 0, τ = 0.8)

| | Accuracy | mAcc | Rock | Hole |
|---|---|---|---|---|
| Teacher alone | 85.7% | 61.2% | 12% | 21% |
| SAM vote, only on the 70% of pixels it fills | 93.8% | 85.7% | 95% | 41% |
| SAM where it fills, teacher elsewhere | 89.9% | 74.2% | 79% | 36% |

## β sweep (`val_beta.json`, SAM where it fills, teacher elsewhere, τ = 0.6)

| β | Accuracy | mAcc | Rock |
|---|---|---|---|
| teacher alone | 85.7% | 61.2% | 12% |
| 0 | 89.4% | 75.2% | 79% |
| 0.25 | 89.6% | 75.2% | 79% |
| 0.5 | 87.7% | 66.5% | 52% |
| 0.75 | 85.4% | 62.1% | 19% |
| 1.0 | 83.6% | 59.3% | 14% |

Adding the teacher to the vote makes it worse, so the method we propose uses β = 0. `val_beta.json` uses the reworked mixing formula, so its β = 1 rows don't match the β = 1 rows in `val_full.json`. There are very few hole pixels in val, so the hole numbers are noisy.
