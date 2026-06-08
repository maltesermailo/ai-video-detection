#!/usr/bin/env python3
"""
gend_score.py — run the WACV-2026 GenD deepfake detector over your face-crop
frame directories (the output of crop_faces.py) and write one results CSV.

GenD is FRAME-level (CLIP encoder + real/fake head); we average per-clip.

Setup (works on Mac MPS, CPU, or CUDA — no torch 1.8 stack):
    git clone https://github.com/yermandy/GenD
    conda create --name GenD python=3.12 uv -y && conda activate GenD
    uv pip install torch==2.8.0 torchvision==0.23.0 transformers==4.56.2

Run:
    python gend_score.py --repo /path/to/GenD \
        --frames_root faces --manifest metadata/manifest.csv \
        --out results/gend_scores.csv

CALIBRATE THE LABEL INDEX FIRST. GenD outputs 2-class logits; which column is
"fake" must be confirmed, not assumed. Run with --calibrate once: it scores the
repo's two example images (one known fake, one known real) and tells you which
index is fake. Then pass --fake_index accordingly (default 0).
"""
import argparse, csv, os, sys
from glob import glob

import torch
from PIL import Image


def pick_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(repo, model_id, device):
    sys.path.insert(0, repo)                       # so `import src.hf...` resolves
    from src.hf.modeling_gend import GenD
    model = GenD.from_pretrained(model_id).to(device).eval()
    return model


@torch.no_grad()
def score_images(model, imgs, device, fake_index):
    tensors = torch.stack([model.feature_extractor.preprocess(im) for im in imgs]).to(device)
    probs = model(tensors).softmax(dim=-1)         # (N, 2)
    return probs[:, fake_index].float().cpu().tolist()


def calibrate(model, device):
    import requests
    base = "https://github.com/yermandy/deepfake-detection/blob/main"
    pairs = [("FAKE (Deepfakes)", f"{base}/datasets/FF/DF/000_003/000.png?raw=true"),
             ("REAL",             f"{base}/datasets/FF/real/000/000.png?raw=true")]
    print("Calibration — fake should score HIGH at the correct index:")
    for name, url in pairs:
        img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        tensors = model.feature_extractor.preprocess(img).unsqueeze(0).to(device)
        probs = model(tensors).softmax(dim=-1)[0].float().cpu().tolist()
        print(f"  {name:18} probs={['%.3f' % p for p in probs]}")
    print("Set --fake_index to the column where the FAKE row is high.")


def load_manifest(path):
    out = {}
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                out[os.path.splitext(r["filename"])[0]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="GenD repo root (for src.hf import)")
    ap.add_argument("--model_id", default="yermandy/GenD_CLIP_L_14")
    ap.add_argument("--frames_root", help="dir of per-clip PNG subdirs (from crop_faces.py)")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", default="results/gend_scores.csv")
    ap.add_argument("--fake_index", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device} | model: {args.model_id}")
    model = load_model(args.repo, args.model_id, device)

    if args.calibrate:
        calibrate(model, device)
        return

    if not args.frames_root:
        sys.exit("--frames_root is required (unless --calibrate)")

    clip_dirs = sorted(d for d, _, fs in os.walk(args.frames_root)
                       if any(f.lower().endswith(".png") for f in fs))
    if not clip_dirs:
        sys.exit(f"no PNG frame dirs under {args.frames_root}")

    man = load_manifest(args.manifest)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["clip", "source", "label_4class", "label_binary", "prompt_id",
            "gend_fake_score", "pred_label", "n_frames"]
    rows = []
    for cd in clip_dirs:
        stem = os.path.basename(cd.rstrip("/"))
        pngs = sorted(glob(os.path.join(cd, "*.png")))[: args.max_frames]
        if not pngs:
            continue
        imgs = [Image.open(p).convert("RGB") for p in pngs]
        frame_scores = score_images(model, imgs, device, args.fake_index)
        clip_score = sum(frame_scores) / len(frame_scores)   # mean-pool over frames
        m = man.get(stem, {})
        rows.append({
            "clip": stem,
            "source": m.get("source", ""),
            "label_4class": m.get("label_4class", ""),
            "label_binary": m.get("label_binary", ""),
            "prompt_id": m.get("prompt_id", ""),
            "gend_fake_score": round(clip_score, 4),
            "pred_label": "Fake" if clip_score > 0.5 else "Real",
            "n_frames": len(pngs),
        })
        print(f"{stem:24} fake_score={clip_score:.3f}  ({len(pngs)} frames)")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} clips)")


if __name__ == "__main__":
    main()
