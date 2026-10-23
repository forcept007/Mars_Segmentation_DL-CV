# Results index

Where each experiment's results are. `evaluate.py` writes to `results/<run>/<split>/` by default, which is why the older runs are flat folders here with their training logs in `runs/<run>/`.

| Experiment | Results | Training logs |
|---|---|---|
| B1 supervised, short and full schedule (MS-20) | `baseline_r50_short*/`, `baseline_r50_full*/` | `runs/baseline_r50_*/` |
| B2 SSL teacher-student, runs v1 to v4 (MS-24) | `MS-24_ssl/` (see its README) | inside each run folder |
| Method 1, class-adaptive thresholds (MS-33) | `ms33_method1_short_40k/` | `runs/ms33_method1_short_40k/` |
| Linear probe check (H6) | `h6_probe/` | - |
| SAM gap filling offline checks (proposed method 3) | `samfill_check/` (see its README) | - |
| Error and confidence analysis (MS-28, MS-32) | `analysis/results/` | - |

Model checkpoints (`.pth`) are not in git because of their size.
