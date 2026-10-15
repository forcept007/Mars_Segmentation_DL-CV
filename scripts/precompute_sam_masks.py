import argparse, json, os, sys, time
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--model", default="vit_h")
parser.add_argument("--limit", type=int, default=None)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(args.root, "split", "train.json")) as f:
        ids = json.load(f)[:args.limit]

    sam = sam_model_registry[args.model](checkpoint=args.checkpoint)
    sam.to('cuda')
    mask_generator = SamAutomaticMaskGenerator(sam, output_mode='coco_rle')
    i = 1
    t0 = time.time()
    for img_id in ids:
        start = time.time()
        out_path = os.path.join(args.out, img_id + '.json')
        if os.path.exists(out_path):
            continue
        image = np.array(Image.open(os.path.join(args.root, "images", img_id + '.jpg')).convert('RGB'))
        masks = mask_generator.generate(image)
        record = {"id": img_id, "model": args.model, "masks": masks}
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(json.dumps(record))
        os.replace(tmp_path, out_path)
        stop = time.time()
        print(f'{i}/{len(ids)}, total time: {(stop - t0):.2f}, iter time: {(stop - start):.2f}, ETA: {((stop - t0)/i)*(len(ids) - i)/3600:.2f}')
        i = i + 1