# B1 trained on SAM-filled labels (proposed method 3, full schedule)

Same model and settings as `baseline_r50_full_1xHeadLR` (ResNet-50 DeepLabV3+, 150k steps), but the train labels are the SAM-filled ones from `scripts/samfill_labels.py` (ρ = 0.8, at least 20 labeled pixels per segment). Val and test labels are the original ones. Config is `configs/samfill_b1_full.yaml`, and the training logs are in `runs/samfill_b1_full/`.

`val/` and `test/` are from `best.pth` (best val mIoU, step 145k). `last_val/` and `last_test/` are from `last.pth` (step 150k). `eval.log` is the printed output of all four evaluations.

| Model | Split | mIoU | mAcc | sky | ridge | soil | sand | bedrock | rock | rover | trace | hole |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 full | val | 84.60 | 89.47 | 97.2 | 97.3 | 91.8 | 88.7 | 95.7 | 56.6 | 97.1 | 81.4 | 55.7 |
| SAM-fill best | val | 78.24 | 88.60 | 97.1 | 96.8 | 81.7 | 77.1 | 94.1 | 52.4 | 93.4 | 59.2 | 52.3 |
| SAM-fill last | val | 74.29 | 87.66 | 95.2 | 96.3 | 81.1 | 75.2 | 94.2 | 52.0 | 60.4 | 60.0 | 54.2 |
| B1 full | test | 73.60 | 79.61 | 95.2 | 93.6 | 85.2 | 74.8 | 92.5 | 27.4 | 85.0 | 76.1 | 32.6 |
| SAM-fill best | test | 71.26 | 80.51 | 95.0 | 93.4 | 81.9 | 73.9 | 91.3 | 26.4 | 72.5 | 67.9 | 39.1 |
| SAM-fill last | test | 70.64 | 79.75 | 94.3 | 92.8 | 81.9 | 75.8 | 91.6 | 23.8 | 72.2 | 69.2 | 33.9 |

Filling the gaps did not help overall. Test mIoU is 2.3 lower than B1 full. mAcc is about the same (+0.9), and hole goes up 6.5, but soil, trace and rover drop, and rock doesn't improve. Most of the filled pixels are soil and bedrock, so one guess is that the extra labels push the model further toward the big classes and blur the soil/sand and trace boundaries. Val mIoU also jumped around a lot during training (between 55 and 78 after step 60k), so best vs. last differs by about 4 points on val.
