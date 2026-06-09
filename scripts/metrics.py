#!/usr/bin/env python3
"""
detection_metrics.py — compute deepfake-detection statistics from a per-clip
score CSV and write two CSVs: a metrics table (by evaluation cut) and a
per-source score-distribution table.

Works for any detector: point --score_col at its score column. Convention:
label 1 = AI/fake (positive class), 0 = real; higher score = more "fake".

    python detection_metrics.py --input results/gend_scores.csv --score_col gend_fake_score
    python detection_metrics.py --input results/fakestormer_scores.csv --score_col score --threshold 0.5

Outputs (next to --input unless --outdir given):
    <stem>_metrics.csv          AUC, EER, accuracy, TPR, TNR, FPR, precision, F1, mean scores
    <stem>_distribution.csv     per-source n / mean / median / std / min / max / low-frame count
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score  # ready-made, correct AUC (handles ties)

# The 10 face-forward scenarios. These are the only clips the FACE-based
# detectors in the brief can really process, so we report a separate row for
# them. Override on the command line with --face_prompts if your set differs.
DEFAULT_FACE = "p08,p15,p21,p26,p27,p31,p32,p33,p34,p35"


def eer(scores, labels):
    """
    FUNCTION 1 of 4 — computes the Equal Error Rate.

    WHAT IT DOES
      Tries every possible threshold and finds the one where the two kinds of
      mistakes balance out: missing an AI clip (calling it real) vs. false-
      alarming on a real clip (calling it fake). The error level at that
      balancing point is the EER. Lower is better; ~0.5 means useless.

    WHY IT EXISTS
      Accuracy depends on where you put the 0.5 cutoff. EER removes that
      arbitrariness by searching for the fairest cutoff itself, so it's a
      cleaner single-number summary of how separable the two classes are.

    INPUTS
      scores : 1-D numpy array of fake-scores (all clips, AI and real mixed)
      labels : 1-D numpy array, same length, 1 = AI/fake, 0 = real
    RETURNS
      a float in [0,1] (the balanced error rate), or NaN if a class is empty.
    """
    pos, neg = scores[labels == 1], scores[labels == 0]      # AI scores, real scores
    if not len(pos) or not len(neg):                         # need both classes
        return float("nan")
    # Candidate thresholds = every distinct score, plus -inf/+inf as the extremes.
    grid = np.concatenate(([-np.inf], np.sort(np.unique(scores)), [np.inf]))
    best = None
    for t in grid:
        fnr = float(np.mean(pos < t))     # fraction of AI scored BELOW t -> missed
        fpr = float(np.mean(neg >= t))    # fraction of real scored AT/ABOVE t -> false alarm
        gap = abs(fnr - fpr)              # how close the two error rates are at this t
        if best is None or gap < best[0]:
            # keep the threshold with the smallest gap; report the averaged error
            best = (gap, (fnr + fpr) / 2.0)
    return best[1]


def r(x, n=4):
    """
    FUNCTION 2 of 4 — a tiny display helper.

    WHAT IT DOES
      Rounds a number to n decimals for the output table. If the value is NaN
      ("not a number", used when a metric is undefined), it returns an empty
      string instead, so the CSV shows a blank cell rather than the word 'nan'.

    THE TRICK
      `x == x` looks pointless but is the standard NaN test: NaN is the only
      value that is NOT equal to itself, so `x == x` is False exactly when x is
      NaN. (No import needed.)
    """
    return round(float(x), n) if x == x else ""


def metrics(pos_scores, neg_scores, group, thr):
    """
    FUNCTION 3 of 4 — the heart of the script. Computes every statistic for ONE
    comparison (one row of the output table).

    WHAT IT DOES
      Given the fake-scores of the AI clips and the fake-scores of the real
      clips, it (a) counts how many fall on each side of the threshold to build
      a confusion matrix, (b) turns those counts into rates (recall, FPR, etc.),
      and (c) computes the two threshold-free summaries, AUC and EER.

    INPUTS
      pos_scores : scores of the POSITIVE class (the AI/fake clips)
      neg_scores : scores of the NEGATIVE class (the real clips)
      group      : a name for this row, e.g. 'veo31_vs_real'
      thr        : the decision cutoff (0.5) for the count-based metrics

    RETURNS
      a dict (one output row): {group, n_AI, n_real, AUC, EER, accuracy, TPR,
      TNR, FPR, precision, F1, mean scores}. main() collects these dicts into
      the final table.

    KEY IDEAS FOR THE NON-NUMPY READER
      - `sp >= thr` compares EVERY element of the array at once and returns a
        boolean array; `.sum()` then counts the True's. So `(sp >= thr).sum()`
        is "how many AI clips were called fake" with no loop.
      - AUC (from sklearn) = probability a random AI clip scores higher than a
        random real clip: 1.0 perfect, 0.5 chance, <0.5 worse than chance.
    """
    sp = np.asarray(pos_scores, float)   # positive (AI) scores
    sn = np.asarray(neg_scores, float)   # negative (real) scores

    # Combine into one score/label vector for AUC and EER (which need both classes).
    scores = np.concatenate([sp, sn])
    labels = np.concatenate([np.ones(len(sp)), np.zeros(len(sn))])

    # --- Confusion-matrix counts at the fixed threshold `thr` ---------------
    # A clip is predicted "fake" if its score >= thr.
    tp = int((sp >= thr).sum())   # AI correctly called fake   (true positive)
    fn = int((sp <  thr).sum())   # AI wrongly  called real    (false negative / miss)
    tn = int((sn <  thr).sum())   # real correctly called real (true negative)
    fp = int((sn >= thr).sum())   # real wrongly  called fake  (false positive / false alarm)

    # --- Rates derived from those counts ------------------------------------
    tpr  = tp / len(sp) if len(sp) else float("nan")   # recall / AI detection rate
    tnr  = tn / len(sn) if len(sn) else float("nan")   # specificity (real kept as real)
    fpr  = fp / len(sn) if len(sn) else float("nan")   # real flagged as fake (= 1 - tnr)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")  # of all "fake" calls, how many right
    # F1 = harmonic mean of precision and recall (only if both are defined)
    f1 = (2 * prec * tpr / (prec + tpr)
          if (prec == prec and tpr == tpr and (prec + tpr)) else float("nan"))
    acc = (tp + tn) / (len(sp) + len(sn)) if (len(sp) + len(sn)) else float("nan")

    # --- Threshold-independent ranking metric: AUC --------------------------
    # AUROC = probability a random AI clip scores higher than a random real clip.
    # 1.0 = perfect, 0.5 = chance, <0.5 = worse than chance (ranking inverted).
    auc = (roc_auc_score(labels, scores)
           if len(sp) and len(sn) and len(np.unique(labels)) == 2 else float("nan"))

    return {
        "group": group,
        "n_AI": len(sp), "n_real": len(sn),          # sample sizes behind the row
        "AUC": r(auc),                                # ranking quality (threshold-free)
        "EER": r(eer(scores, labels)),                # balanced error (threshold-free)
        "accuracy@thr": r(acc),                       # (tp+tn)/total at thr
        "TPR_detect_AI@thr": r(tpr),                  # how many AI clips caught
        "TNR_specificity@thr": r(tnr),                # how many real clips kept
        "FPR_real_as_fake@thr": r(fpr),               # how many real clips misflagged
        "precision@thr": r(prec),
        "F1@thr": r(f1),
        "mean_score_AI": r(sp.mean()) if len(sp) else "",   # avg fake-score for AI clips
        "mean_score_real": r(sn.mean()) if len(sn) else "", # avg fake-score for real clips
    }


def main():
    """
    FUNCTION 4 of 4 — the orchestrator. Decides WHICH slices of data to measure,
    calls metrics() on each, and writes the results.

    THE WHOLE PIPELINE IN ORDER
      1. Read the command-line options (which file, which columns, threshold).
      2. Load the CSV into a pandas table (df) and check the columns exist.
      3. Split the table with boolean masks:
           real = rows where label == 0      (shared negative class)
           ai   = rows where label == 1       (all generators combined)
      4. Build one output row per comparison by calling metrics():
           - overall: all AI vs real
           - one row per generator (veo31, omniflash, ltx23) vs the same real
           - face subset: only the face prompt_ids, AI vs real
      5. Build a second table: per-source score distribution (groupby).
      6. Write both tables to CSV and print them.

    pandas note: `df[mask]` keeps the rows where the boolean Series `mask` is
    True — it's a vectorized filter, like SQL WHERE. `df[col].values` drops a
    column down to a plain numpy array for the math in metrics().
    """
    # --- Command-line options ----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="per-clip score CSV")
    ap.add_argument("--outdir", default="", help="output dir (default: alongside --input)")
    ap.add_argument("--score_col", default="gend_fake_score", help="column holding the fake-score")
    ap.add_argument("--label_col", default="label_binary", help="1=AI/fake, 0=real")
    ap.add_argument("--source_col", default="source", help="generator name / 'real'")
    ap.add_argument("--prompt_col", default="prompt_id", help="used to build the face subset")
    ap.add_argument("--frames_col", default="n_frames", help="optional; for low-frame count")
    ap.add_argument("--min_frames", type=int, default=4,
                    help="drop clips scored on fewer than this many frames (noisy); "
                         "set 0 (or 1) to keep all. Needs frames_col present.")
    ap.add_argument("--real_source", default="real", help="value in source_col meaning 'real'")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision cutoff for @thr metrics")
    ap.add_argument("--face_prompts", default=DEFAULT_FACE,
                    help="comma-separated prompt_ids forming the face subset")
    args = ap.parse_args()

    # --- Load and sanity-check the CSV -------------------------------------
    df = pd.read_csv(args.input)
    for col in (args.score_col, args.label_col, args.source_col):
        if col not in df.columns:                       # fail loudly if a column is misnamed
            raise SystemExit(f"column '{col}' not in {args.input}. Found: {list(df.columns)}")
    # Force the score column numeric; blank/garbage becomes NaN, then we drop those rows.
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
    df = df.dropna(subset=[args.score_col])

    # --- Drop low-frame clips ---------------------------------------------
    # Clips scored on only 1-3 frames give noisy per-clip scores, so by default
    # we filter out anything with fewer than --min_frames frames before
    # computing any metric. Guarded: only runs if the frames column exists.
    if args.frames_col in df.columns and args.min_frames > 1:
        nf = pd.to_numeric(df[args.frames_col], errors="coerce")
        keep = nf >= args.min_frames
        dropped = df[~keep]
        if len(dropped):
            by_src = dropped.groupby(args.source_col).size()
            print(f"dropped {len(dropped)} clip(s) with < {args.min_frames} frames: "
                  + ", ".join(f"{s}={int(n)}" for s, n in by_src.items()))
        df = df[keep]

    thr = args.threshold
    real = df[df[args.label_col] == 0]                  # all real clips (shared negative class)
    ai = df[df[args.label_col] == 1]                    # all AI clips (combined positive class)
    ss = lambda d: d[args.score_col].values             # tiny helper: pull the score array

    # --- Row 1: overall (all generators pooled) vs real --------------------
    rows = [metrics(ss(ai), ss(real), "overall_all_AI_vs_real", thr)]

    # --- One row per generator vs the SAME real set ------------------------
    # Every source that isn't 'real' is treated as its own positive class.
    for gen in sorted(g for g in df[args.source_col].unique() if g != args.real_source):
        rows.append(metrics(ss(df[df[args.source_col] == gen]), ss(real), f"{gen}_vs_real", thr))

    # --- Face-subset row: same AI-vs-real, restricted to face prompts ------
    if args.prompt_col in df.columns:
        face_ids = {p.strip() for p in args.face_prompts.split(",") if p.strip()}
        face = df[df[args.prompt_col].isin(face_ids)]   # keep only face-forward scenarios
        rows.append(metrics(ss(face[face[args.label_col] == 1]),   # face AI clips
                            ss(face[face[args.label_col] == 0]),    # face real clips
                            "face_subset_AI_vs_real", thr))

    metrics_df = pd.DataFrame(rows)

    # --- Second table: raw score distribution per source ------------------
    # Not a detection metric — just describes how the scores spread out for
    # each generator and the real set (useful for a histogram / sanity check).
    dist = (df.groupby(args.source_col)[args.score_col]
              .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
              .round(4).reset_index())
    # Flag data quality: how many clips per source were scored on <4 frames
    # (those scores are noisier; the models expect several frames).
    if args.frames_col in df.columns:
        low = (df[df[args.frames_col] < 4].groupby(args.source_col)["clip"].count()
               if "clip" in df.columns else
               df[df[args.frames_col] < 4].groupby(args.source_col).size())
        dist["n_clips_lt4_frames"] = (low.reindex(dist[args.source_col])
                                         .fillna(0).astype(int).values)

    # --- Write outputs (names derived from the input filename) -------------
    outdir = args.outdir or os.path.dirname(args.input) or "."
    os.makedirs(outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]   # e.g. 'gend_scores'
    m_path = os.path.join(outdir, f"{stem}_metrics.csv")
    d_path = os.path.join(outdir, f"{stem}_distribution.csv")
    metrics_df.to_csv(m_path, index=False)
    dist.to_csv(d_path, index=False)

    # Also print both tables to the terminal so you see results immediately.
    print(metrics_df.to_string(index=False))
    print()
    print(dist.to_string(index=False))
    print(f"\nwrote {m_path}\nwrote {d_path}")


if __name__ == "__main__":
    main()
