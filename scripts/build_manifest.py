#!/usr/bin/env python3
"""
build_manifest.py — scan raw/<source>/ clips, parse filenames, probe original
specs, and emit metadata/manifest.csv. One row per clip.

Filename convention (enforced):  <source>_p<NN>_<take>.<ext>
    e.g.  veo31_p01_001.mp4   real_p25_002.mov

What gets filled automatically:
    filename (normalized .mp4 name), source, label_binary, label_4class,
    prompt_id, take, orig_resolution, orig_fps, orig_duration_s
Left blank for you to fill (can't be derived from the file):
    orig_url, license            -> real clips (provenance)
    gen_model_version, gen_seed, gen_date -> AI clips (reproducibility)
    notes

Run from project root, or pass --root.
    ./scripts/build_manifest.py
    ./scripts/build_manifest.py --force      # overwrite existing manifest
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

SOURCES = ["real", "veo31", "omniflash", "ltx23"]
LABEL_BINARY = {"real": 0, "veo31": 1, "omniflash": 1, "ltx23": 1}
LABEL_4CLASS = {"real": "real", "veo31": "veo31", "omniflash": "omniflash", "ltx23": "ltx23"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

# Accept short aliases in filenames/dirs and fold them to the canonical source.
ALIASES = {"veo": "veo31", "ltx": "ltx23"}
# canonical source -> directory names to scan (canonical first, then aliases)
DIR_CANDIDATES = {c: [c] for c in SOURCES}
for _alias, _canon in ALIASES.items():
    DIR_CANDIDATES[_canon].append(_alias)

# longer alternatives first so e.g. "veo31" wins over "veo"
NAME_RE = re.compile(
    r"^(?P<source>real|veo31|veo|omniflash|ltx23|ltx)_p(?P<pid>\d{2})_(?P<take>\d+)$"
)

COLUMNS = [
    "filename", "source", "label_binary", "label_4class", "prompt_id", "take",
    "orig_url", "license", "orig_resolution", "orig_fps", "orig_duration_s",
    "gen_model_version", "gen_seed", "gen_date", "notes",
]


def probe(path: Path):
    """Return (resolution, fps, duration_s) from ffprobe, or blanks on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        data = json.loads(out)
        st = (data.get("streams") or [{}])[0]
        w, h = st.get("width"), st.get("height")
        resolution = f"{w}x{h}" if w and h else ""
        afr = st.get("avg_frame_rate", "0/0")
        try:
            fps = round(float(Fraction(afr)), 3) if afr and afr != "0/0" else ""
        except (ZeroDivisionError, ValueError):
            fps = ""
        dur = data.get("format", {}).get("duration", "")
        dur = round(float(dur), 2) if dur else ""
        return resolution, fps, dur
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return "", "", ""


def main():
    ap = argparse.ArgumentParser(description="Build manifest.csv from raw/ clips.")
    ap.add_argument("--root", default=".", help="project root (default: cwd)")
    ap.add_argument("--force", action="store_true", help="overwrite existing manifest.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    raw_dir = root / "raw"
    manifest = root / "metadata" / "manifest.csv"

    if not raw_dir.is_dir():
        sys.exit(f"ERROR: {raw_dir} not found — run scaffold.sh first or pass --root")
    # Only protect a manifest that already holds data rows; a header-only file
    # (as written by scaffold.sh) is safe to (re)generate.
    if manifest.exists() and not args.force:
        with manifest.open(newline="") as fh:
            data_rows = sum(1 for i, line in enumerate(fh) if i >= 1 and line.strip())
        if data_rows:
            sys.exit(f"ERROR: {manifest} already has {data_rows} data row(s). "
                     f"Use --force to overwrite (this discards hand-entered rows).")

    rows, skipped = [], []
    for src in SOURCES:
        for dname in DIR_CANDIDATES[src]:
            d = raw_dir / dname
            if not d.is_dir():
                continue
            for f in sorted(d.iterdir()):
                if f.suffix.lower() not in VIDEO_EXTS:
                    continue
                m = NAME_RE.match(f.stem)
                if not m:
                    skipped.append(f.name)
                    continue
                canon = ALIASES.get(m.group("source"), m.group("source"))
                if canon != src:
                    skipped.append(f"{f.name} (in raw/{dname}/ but named {m.group('source')})")
                    continue
                resolution, fps, dur = probe(f)
                rows.append({
                    "filename": f.stem + ".mp4",          # normalized output name (basename preserved)
                    "source": src,                         # canonical, even if file said veo/ltx
                    "label_binary": LABEL_BINARY[src],
                    "label_4class": LABEL_4CLASS[src],
                    "prompt_id": "p" + m.group("pid"),
                    "take": m.group("take"),
                    "orig_url": "", "license": "",
                    "orig_resolution": resolution, "orig_fps": fps, "orig_duration_s": dur,
                    "gen_model_version": "", "gen_seed": "", "gen_date": "", "notes": "",
                })

    rows.sort(key=lambda r: (r["prompt_id"], r["source"], r["take"]))

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {manifest}  ({len(rows)} clips)")
    by_src = {s: sum(1 for r in rows if r["source"] == s) for s in SOURCES}
    print("  per source:", "  ".join(f"{s}={by_src[s]}" for s in SOURCES))
    n_prompts = len({r["prompt_id"] for r in rows})
    print(f"  distinct prompts covered: {n_prompts}/25")
    if skipped:
        print(f"  SKIPPED {len(skipped)} misnamed file(s):")
        for s in skipped:
            print(f"    - {s}")


if __name__ == "__main__":
    # Behave like a normal Unix filter when piped into head/less, etc.
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass  # SIGPIPE unavailable (e.g. Windows) — not fatal
    main()
