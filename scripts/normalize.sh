#!/usr/bin/env bash
#
# normalize.sh — re-encode every clip in raw/<source>/ into processed/<source>/
#                with identical resolution, fps, duration, codec, bitrate and
#                stripped audio + metadata, so the ONLY surviving difference
#                between classes is the generation itself.
#
# Run from the project root, or set ROOT=/path/to/project.
#
#   ./scripts/normalize.sh            # normalize everything not yet done
#   ./scripts/normalize.sh --force    # re-encode even if output exists
#
# Portable across macOS (bash 3.2) and Linux. Requires ffmpeg + ffprobe.

set -eu

# ---- target spec (edit to taste; keep identical across the whole set) ----
TARGET_W=1280          # output width
TARGET_H=720           # output height  (720p canvas, letterboxed)
FPS=24                 # output frame rate (<= every source's native fps)
DURATION=8             # output length in seconds (Veo's 8s is the floor)
VBITRATE=4M            # constant-ish video bitrate (kills compression tells)
GOP=48                 # keyframe interval (2s @ 24fps), uniform across clips
PRESET=slow            # x264 preset
# --------------------------------------------------------------------------

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

# Locate project root: explicit $ROOT, else parent of this script's dir.
if [ -z "${ROOT:-}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

command -v ffmpeg  >/dev/null 2>&1 || { echo "ERROR: ffmpeg not found"  >&2; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "ERROR: ffprobe not found" >&2; exit 1; }

SOURCES="real veo31 omniflash ltx23"
# scale to fit inside the canvas (no distortion), then pad to exact WxH.
VF="scale=${TARGET_W}:${TARGET_H}:force_original_aspect_ratio=decrease,pad=${TARGET_W}:${TARGET_H}:(ow-iw)/2:(oh-ih)/2,fps=${FPS},format=yuv420p"

n_done=0; n_skip=0; n_warn=0

shopt -s nullglob
for src in $SOURCES; do
  in_dir="$ROOT/raw/$src"
  out_dir="$ROOT/processed/$src"
  [ -d "$in_dir" ] || continue
  mkdir -p "$out_dir"

  for f in "$in_dir"/*.mp4 "$in_dir"/*.mov "$in_dir"/*.mkv "$in_dir"/*.webm "$in_dir"/*.m4v "$in_dir"/*.avi; do
    base="$(basename "$f")"
    stem="${base%.*}"
    out="$out_dir/$stem.mp4"

    if [ -f "$out" ] && [ "$FORCE" -eq 0 ]; then
      n_skip=$((n_skip + 1))
      continue
    fi

    # duration sanity check: can't trim a clip up to length
    dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null || echo 0)"
    # integer-compare without bc (portable): strip decimals
    dur_int="${dur%.*}"; [ -z "$dur_int" ] && dur_int=0
    if [ "$dur_int" -lt "$DURATION" ]; then
      echo "WARN  $base is ${dur}s (< ${DURATION}s target) — output will be short"
      n_warn=$((n_warn + 1))
    fi

    ffmpeg -nostdin -y -loglevel error \
      -ss 0 -t "$DURATION" -i "$f" \
      -vf "$VF" \
      -c:v libx264 -preset "$PRESET" \
      -b:v "$VBITRATE" -maxrate "$VBITRATE" -bufsize "$VBITRATE" \
      -g "$GOP" -keyint_min "$GOP" -sc_threshold 0 \
      -pix_fmt yuv420p \
      -an \
      -map_metadata -1 -map_chapters -1 \
      -movflags +faststart \
      "$out"

    echo "ok    $src/$stem.mp4"
    n_done=$((n_done + 1))
  done
done

echo
echo "normalized: $n_done   skipped(existing): $n_skip   short-clip warnings: $n_warn"
echo "output in:  $ROOT/processed/<source>/"
