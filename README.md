# Mars_Segmentation_DL-CV
Group project for class: Deep Learning for Computer Vision. Project focuses on Mars environment Segmentation

## Download Dataset
https://jhang2020.github.io/S5Mars.github.io/

## Interactive demo

A Gradio app to explore the models on curated test images, compare them, step through the project's models,
play a "beat the model" quiz and render segmented Mastcam panorama sweeps (a parked rover's camera panning over a mosaic) with a terrain-hazard badge.

```bash
pip install -r requirements.txt -r requirements-demo.txt
python -m demo.app                                # auto device (cuda > mps > cpu), http://localhost:7860
uv run -m demo.app
```

**Checkpoints.** Copy the `.pth` files into `demo/checkpoints/` using the names in `demo/models.yaml`
(e.g. `dinov2_vitb14_short.pth`, `baseline_r50_full_1xHeadLR.pth`, `ssl_v4_student_best.pth`). Anything else in
`demo/checkpoints/` or `runs/<run>/best.pth` is picked up automatically, and the *Checkpoints* tab loads any path.
Missing checkpoints run as clearly marked placeholders (random-weight ResNet-50) so the UI still works.
DINOv2 models need the `facebookresearch/dinov2` hub code once (internet, or set `DINOV2_HUB_DIR` to a local clone).

**On the GPU box (AWS).** Precompute predictions, quiz patches and videos, then serve:

```bash
python -m demo.precompute                         # every model with a checkpoint
python -m demo.app --device cuda --host 0.0.0.0   # laptop: ssh -L 7860:localhost:7860 <box>, or add --share
```

`demo/cache/` is small; copy it to a laptop and the curated images, story slider, quiz and videos work instantly
on CPU (the cache is ignored automatically when a checkpoint changes). "Force live inference" shows real latency.
