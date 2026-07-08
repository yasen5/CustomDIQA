"""Like run_ddi_model_agreement.py (which correlates raw per-image scores across pretrained IQA
models), but measures a different, more human-relevant notion of "agreement": given a pair of DDI
images, do two models pick the *same one* as higher quality? Pearson/Spearman correlation can look
strong even when models disagree on plenty of individual head-to-head comparisons, so this instead
draws a pool of random image pairs and reports, for every pair of models (e.g. (topiq_nr, musiq),
(topiq_nr, qalign_mini), (musiq, qalign_mini)), the fraction of pairs where both models rank the
same image higher — a boolean pairwise-preference agreement rate.

Can also include "deqa" in --model-types. Unlike the other models, DeQA-Score can't be constructed
in this process (its vendored modeling code needs an older pinned transformers than this venv has
-- see src/model/deqa/constants.py), so its scores are instead read from a CSV cache built
separately under deqa_venv:

    deqa_venv/bin/python3 scripts/run_ddi_deqa_scores.py

Run that first (it scores every DDI image once and is cheap to re-run -- already-cached files are
skipped) before passing --model-types ... deqa here.

Saves:
  - a per-pair CSV with both images' scores from every model and each model's "image_a scores
    higher than image_b" boolean
  - a pairwise agreement-rate matrix across models (CSV + heatmap), where cell (m1, m2) is the
    fraction of sampled pairs on which m1 and m2 pick the same image as higher quality

Usage:
    python scripts/run_ddi_pairwise_agreement.py [--model-types TYPE [TYPE ...]] [--image-dir DIR]
        [--metadata CSV] [--output-dir DIR] [--num-pairs N] [--seed N] [--deqa-scores-cache CSV]
"""
import argparse
import itertools
import os
import random
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.trainer import get_device
import script_utils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEQA_SCORES_CACHE_DEFAULT = os.path.join(REPO_ROOT, "validation", "ddi_deqa_scores.csv")
ALL_MODEL_TYPES = tuple(script_utils.PRETRAINED_MODEL_TYPES) + ("deqa",)


def load_deqa_scores(cache_path, files):
    """Reads the {DDI_file: deqa_score} cache built by run_ddi_deqa_scores.py (run under
    deqa_venv) and returns a {file: score} dict restricted to `files`. Raises with a clear
    pointer to that script if the cache is missing entirely or doesn't cover every needed file --
    DeQA can't be scored in this process (see module docstring), so there's no in-process fallback."""
    if not os.path.exists(cache_path):
        raise SystemExit(
            f"--model-types includes 'deqa' but no score cache found at {cache_path}. Run:\n"
            f"  deqa_venv/bin/python3 scripts/run_ddi_deqa_scores.py\n"
            f"first to precompute DeQA scores (DeQA can't be loaded in this venv -- see "
            f"src/model/deqa/constants.py).")
    cache_df = pd.read_csv(cache_path)
    scores = dict(zip(cache_df["DDI_file"], cache_df["deqa_score"]))
    missing = [f for f in files if f not in scores]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(files)} needed DDI files are missing from the deqa score "
            f"cache at {cache_path} (e.g. {missing[:5]}). Run:\n"
            f"  deqa_venv/bin/python3 scripts/run_ddi_deqa_scores.py\n"
            f"to (re-)score them -- already-cached files are skipped, so this is cheap to re-run.")
    return {f: scores[f] for f in files}


def sample_pairs(files, num_pairs, seed):
    """Draws `num_pairs` distinct unordered pairs of filenames out of `files` without replacement
    (a given pair of images won't be drawn twice, though a single image can appear in many pairs).
    Rejection sampling instead of itertools.combinations, since DDI has ~656 images (~215k possible
    pairs) and we only need ~1000 of them."""
    rng = random.Random(seed)
    if len(files) < 2:
        raise SystemExit("Need at least 2 images to form pairs")
    max_possible = len(files) * (len(files) - 1) // 2
    if num_pairs > max_possible:
        raise SystemExit(f"--num-pairs {num_pairs} exceeds the {max_possible} distinct pairs available")

    pairs = set()
    while len(pairs) < num_pairs:
        a, b = rng.sample(files, 2)
        pairs.add((a, b) if a < b else (b, a))
    return list(pairs)


def score_images(iqa_model, files, image_dir, batch_size):
    """Scores every file in `files` (deduplicated) with `iqa_model`, returning a {file: score} dict."""
    scores = {}
    for kept, batch_scores in script_utils.predict_raw_items(
        iqa_model, list(files), image_dir, lambda f: f, batch_size
    ):
        for f, score in zip(kept, batch_scores):
            scores[f] = score
    return scores


def build_pairs_table(pairs, model_types, scores_by_model):
    rows = []
    for image_a, image_b in pairs:
        row = {"image_a": image_a, "image_b": image_b}
        for model_type in model_types:
            score_a = scores_by_model[model_type].get(image_a)
            score_b = scores_by_model[model_type].get(image_b)
            row[f"{model_type}_score_a"] = score_a
            row[f"{model_type}_score_b"] = score_b
            row[f"{model_type}_a_higher"] = None if score_a is None or score_b is None or score_a == score_b \
                else score_a > score_b
        rows.append(row)
    return pd.DataFrame(rows)


def compute_agreement_matrix(df, model_types):
    """Cell (m1, m2) is the fraction of pairs where m1 and m2's "image_a scores higher" booleans
    match, restricted to pairs where neither model called a tie (a_higher is not None)."""
    matrix = pd.DataFrame(np.nan, index=model_types, columns=model_types)
    n_compared = pd.DataFrame(0, index=model_types, columns=model_types)
    for m1, m2 in itertools.combinations_with_replacement(model_types, 2):
        col1, col2 = df[f"{m1}_a_higher"], df[f"{m2}_a_higher"]
        both_called = col1.notna() & col2.notna()
        n = both_called.sum()
        rate = (col1[both_called] == col2[both_called]).mean() if n else float("nan")
        matrix.loc[m1, m2] = matrix.loc[m2, m1] = rate
        n_compared.loc[m1, m2] = n_compared.loc[m2, m1] = n
    return matrix, n_compared


def print_agreement_table(matrix, n_compared):
    print("\n=== Pairwise-preference agreement rate between models (fraction of sampled DDI image "
          "pairs where both models pick the same image as higher quality) ===")
    header = f"{'':<14}" + "".join(f"{m:>12}" for m in matrix.columns)
    print(header)
    for model_type in matrix.index:
        row = f"{model_type:<14}" + "".join(f"{matrix.loc[model_type, m]:>12.3f}" for m in matrix.columns)
        print(row)
    print("\n(n comparisons per pairing, after dropping ties)")
    header = f"{'':<14}" + "".join(f"{m:>12}" for m in n_compared.columns)
    print(header)
    for model_type in n_compared.index:
        row = f"{model_type:<14}" + "".join(f"{n_compared.loc[model_type, m]:>12d}" for m in n_compared.columns)
        print(row)


def plot_agreement_heatmap(matrix, num_pairs, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    im = ax.imshow(matrix.values.astype(float), cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for i, j in itertools.product(range(len(matrix.index)), range(len(matrix.columns))):
        ax.text(j, i, f"{matrix.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_title(f"Pairwise-preference agreement\n({num_pairs} sampled DDI image pairs)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="agreement rate")
    plot_path = os.path.join(output_dir, "ddi_pairwise_agreement_heatmap.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved agreement heatmap to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample a pool of DDI image pairs and measure, for every pair of this repo's "
                     "pretrained IQA models (topiq_nr, musiq, qalign_mini), how often they agree on "
                     "which image in each pair is higher quality.")
    parser.add_argument("--model-types", nargs="+", choices=ALL_MODEL_TYPES,
                         default=list(script_utils.PRETRAINED_MODEL_TYPES),
                         help="Which pretrained models to compare (default: all of them, excluding "
                              "deqa since it requires a precomputed score cache -- see module docstring)")
    parser.add_argument("--deqa-scores-cache", default=DEQA_SCORES_CACHE_DEFAULT,
                         help="CSV cache of DeQA scores built by "
                              "'deqa_venv/bin/python3 scripts/run_ddi_deqa_scores.py'. Only read "
                              "if 'deqa' is in --model-types.")
    parser.add_argument("--image-dir", default=os.path.join(REPO_ROOT, "ddi_data"))
    parser.add_argument("--metadata", default=os.path.join(REPO_ROOT, "ddi_metadata.csv"))
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "validation", "ddi_pairwise_agreement"))
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Chunk size for image loading; these models score one image per "
                              "forward pass regardless, so this only affects I/O batching.")
    parser.add_argument("--num-pairs", type=int, default=1000,
                         help="How many distinct random DDI image pairs to sample")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for pair sampling")
    args = parser.parse_args()

    if len(args.model_types) < 2:
        raise SystemExit("Need at least 2 --model-types to measure pairwise agreement between them")

    os.makedirs(args.output_dir, exist_ok=True)

    all_files = pd.read_csv(args.metadata)["DDI_file"].tolist()
    print(f"Loaded {len(all_files)} DDI images from {args.metadata}")

    pairs = sample_pairs(all_files, args.num_pairs, args.seed)
    pool_files = sorted({f for pair in pairs for f in pair})
    print(f"Sampled {len(pairs)} distinct image pairs, spanning {len(pool_files)} unique images")

    device = get_device()
    print(f"Device: {device}")

    scores_by_model = {}
    for model_type in args.model_types:
        print(f"\n{f' Scoring with pretrained model: {model_type} ':=^88}")
        if model_type == "deqa":
            # Can't be constructed in this process -- read from the cache built separately under
            # deqa_venv (see load_deqa_scores/module docstring).
            scores_by_model[model_type] = load_deqa_scores(args.deqa_scores_cache, pool_files)
            continue
        if model_type == "musiq":
            # Same rationale as run_ddi_model_agreement.py: use pyiqa's own pretrained MUSIQ head
            # (real predictions) rather than script_utils.MusiqModel's untuned-head reimplementation.
            iqa_model = script_utils.MusiqNativeModel(device)
        else:
            model_args = argparse.Namespace(model_type=model_type, model_path=None)
            iqa_model = script_utils.load_model(model_args, device)

        scores_by_model[model_type] = score_images(iqa_model, pool_files, args.image_dir, args.batch_size)

        del iqa_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = build_pairs_table(pairs, args.model_types, scores_by_model)
    score_cols = [f"{m}_score_a" for m in args.model_types] + [f"{m}_score_b" for m in args.model_types]
    n_dropped = df[score_cols].isna().any(axis=1).sum()
    if n_dropped:
        print(f"Warning: {n_dropped} pairs missing a score from at least one model (image failed to "
              f"load), dropping them.")
        df = df.dropna(subset=score_cols).reset_index(drop=True)
    if len(df) < 1:
        raise SystemExit("No pairs have scores from every model — nothing to compare")

    pairs_path = os.path.join(args.output_dir, "ddi_pairwise_agreement_pairs.csv")
    df.to_csv(pairs_path, index=False)
    print(f"\nSaved per-pair scores and preferences to {pairs_path}")

    matrix, n_compared = compute_agreement_matrix(df, args.model_types)
    matrix_path = os.path.join(args.output_dir, "ddi_pairwise_agreement_matrix.csv")
    matrix.to_csv(matrix_path)
    print(f"Saved agreement matrix to {matrix_path}")

    print_agreement_table(matrix, n_compared)

    try:
        plot_agreement_heatmap(matrix, len(df), args.output_dir)
    except Exception as e:
        print(f"Warning: could not generate plot ({e})")


if __name__ == "__main__":
    main()
