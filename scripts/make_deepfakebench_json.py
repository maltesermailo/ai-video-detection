#!/usr/bin/env python3
"""
make_deepfakebench_json.py — register your face-crop clips as a custom
DeepfakeBench test dataset by emitting the nested JSON its loader expects.

DeepfakeBench's loader reads  preprocessing/dataset_json/<NAME>.json  with shape:

  { "<NAME>": {
      "<subset>": {                 # we use the source (veo31 / real / ...) as subset
        "test": {                    # mode
          "<clip>": {"label": "<subset>", "frames": ["/abs/000.png", ...]},
          ... } } } }

Each video's "label" must also be a key in the label_dict of test_config.yaml
(0 = real, 1 = fake) — this script prints the exact lines to add.

Frames are taken straight from your faces/<source>/<clip>/*.png crops (256px,
numeric names) — no landmark/mask extraction needed for a naive detector.

    python make_deepfakebench_json.py \
        --faces_root faces --dataset_name Custom2026 \
        --out /path/to/DeepfakeBench/preprocessing/dataset_json/Custom2026.json
"""
import argparse, json, os
from glob import glob


def numeric_stem(p):
    s = os.path.splitext(os.path.basename(p))[0]
    return int(s) if s.isdigit() else 0


def collect(faces_root):
    """faces_root/<source>/<clip>/NNN.png  ->  {source: {clip: [abs frame paths]}}"""
    out = {}
    for source in sorted(os.listdir(faces_root)):
        sdir = os.path.join(faces_root, source)
        if not os.path.isdir(sdir):
            continue
        for clip in sorted(os.listdir(sdir)):
            cdir = os.path.join(sdir, clip)
            if not os.path.isdir(cdir):
                continue
            pngs = sorted(glob(os.path.join(cdir, "*.png")), key=numeric_stem)
            if pngs:
                out.setdefault(source, {})[clip] = [os.path.abspath(p) for p in pngs]
    return out


def build(collected, sources, dataset_name, mode):
    d = {dataset_name: {}}
    for source in sources:
        d[dataset_name][source] = {mode: {}}
        for clip, frames in collected[source].items():
            d[dataset_name][source][mode][clip] = {"label": source, "frames": frames}
    return d


def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces_root", required=True, help="dir of <source>/<clip>/*.png crops")
    ap.add_argument("--dataset_name", default="Custom2026")
    ap.add_argument("--out", required=True, help="output JSON (…/preprocessing/dataset_json/<NAME>.json)")
    ap.add_argument("--mode", default="test", choices=["train", "val", "test"])
    ap.add_argument("--real_source", default="real")
    ap.add_argument("--per_generator", action="store_true",
                    help="also emit one JSON per generator (each = that generator + all real)")
    args = ap.parse_args()

    collected = collect(args.faces_root)
    if not collected:
        raise SystemExit(f"no <source>/<clip>/*.png found under {args.faces_root}")
    sources = sorted(collected)

    # main combined dataset (all sources)
    write_json(build(collected, sources, args.dataset_name, args.mode), args.out)
    n_clips = sum(len(v) for v in collected.values())
    print(f"wrote {args.out}")
    print(f"  dataset '{args.dataset_name}': {n_clips} clips across sources {sources}")

    emitted = [args.dataset_name]
    # optional: one dataset per generator (gen + real) for per-generator AUC
    if args.per_generator and args.real_source in collected:
        outdir = os.path.dirname(args.out)
        for gen in sources:
            if gen == args.real_source:
                continue
            name = f"{args.dataset_name}_{gen}"
            obj = build(collected, [gen, args.real_source], name, args.mode)
            p = os.path.join(outdir, f"{name}.json")
            write_json(obj, p)
            emitted.append(name)
            print(f"wrote {p}  ({gen} + {args.real_source})")

    # ---- the manual edits the loader requires ----
    print("\n--- 1) add these to label_dict in training/config/test_config.yaml ---")
    for s in sources:
        print(f"  {s}: {0 if s == args.real_source else 1}")
    print("\n--- 2) in test_config.yaml also set ---")
    print("  lmdb: False        # read PNG frames directly (not from an LMDB)")
    print("\n--- 3) run a naive (frames-only) detector, e.g. xception ---")
    for name in emitted:
        print(f"  python training/test.py --detector_path training/config/detector/xception.yaml \\")
        print(f"      --test_dataset \"{name}\" --weights_path training/pretrained/xception_best.pth")


if __name__ == "__main__":
    main()
