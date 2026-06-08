#!/usr/bin/env python3
"""
crop_faces.py — turn each video into a directory of RetinaFace-cropped face
frames (256x256 PNGs, zero-padded names), which is what FakeSTormer's
`test_vid` mode expects as its `-v` argument.

Mirrors package_utils/images_crop.py (RetinaFace, padding=0.25, 256px).

GPU box only — needs torch + retinaface-pytorch.

    python crop_faces.py --videos processed/veo31 --out faces/veo31 --num_frames 32

Then point test.py at each produced subdir:
    python scripts/test.py --cfg <cfg> -v faces/veo31/veo31_p26_001
"""
import argparse, os
from glob import glob
import cv2
import numpy as np
from retinaface.pre_trained_models import get_model

PAD = 0.25
SIZE = 256
EXTS = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi")


def crop_one(model, video_path, out_dir, num_frames):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"  WARN no frames: {video_path}")
        return 0
    idxs = np.linspace(0, total - 1, num_frames, endpoint=True, dtype=np.int64)
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for cnt in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if cnt not in idxs:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = model.predict_jsons(rgb)
        faces = [f for f in faces if f.get("bbox")]
        if not faces:
            continue
        # largest face
        f = max(faces, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        x0, y0, x1, y1 = f["bbox"]
        w, h = x1 - x0, y1 - y0
        H, W = frame.shape[:2]
        x0 = max(int(x0 - PAD * w), 0); y0 = max(int(y0 - PAD * h), 0)
        x1 = min(int(x1 + PAD * w), W); y1 = min(int(y1 + PAD * h), H)
        crop = rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (SIZE, SIZE))
        out = os.path.join(out_dir, f"{saved:03d}.png")
        cv2.imwrite(out, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        saved += 1
    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="dir of input videos")
    ap.add_argument("--out", required=True, help="output dir (one subdir per clip)")
    ap.add_argument("--num_frames", type=int, default=32,
                    help="frames sampled per clip (test_vid uses the first NUM_FRAMES)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model = get_model("resnet50_2020-07-20", max_size=1024, device=args.device)
    model.eval()

    vids = [p for p in sorted(glob(os.path.join(args.videos, "*"))) if p.lower().endswith(EXTS)]
    if not vids:
        raise SystemExit(f"no videos found in {args.videos}")
    for v in vids:
        stem = os.path.splitext(os.path.basename(v))[0]
        n = crop_one(model, v, os.path.join(args.out, stem), args.num_frames)
        flag = "" if n >= 4 else "  <-- WARNING: <4 frames, model needs >=4"
        print(f"{stem}: {n} face frames{flag}")


if __name__ == "__main__":
    main()
