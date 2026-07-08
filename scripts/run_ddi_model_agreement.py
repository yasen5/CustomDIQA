"""Run every pretrained (no-checkpoint-required) IQA model in this repo — topiq_nr, musiq, and
qalign_mini — over the DDI images and measure how much their quality estimates agree with each
other, as opposed to run_ddi_ita_correlation.py/run_ddi_fitzpatrick_correlation.py, which each
correlate a single model against a skin-tone signal. The point here is purely inter-model
agreement: where do these IQA models rank the same dermatology image very differently?

Saves:
  - a per-image CSV with every model's raw score, its percentile rank within that model (to make
    scores comparable despite differing scales), and a disagreement score (std of percentile
    ranks across models), sorted by disagreement descending
  - Pearson and Spearman pairwise correlation matrices across models (CSV + heatmap)
  - pairwise scatter plots between every pair of models, colored by disagreement

Usage:
    python scripts/run_ddi_model_agreement.py [--model-types TYPE [TYPE ...]] [--image-dir DIR]
        [--metadata CSV] [--output-dir DIR] [--max-samples N] [--top-n N]
"""
import argparse
import itertools
import os
import sys

import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, ".")
from src.trainer import get_device
import script_utils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def score_model(iqa_model, df, image_dir, batch_size):
    """Scores every row of `df` with `iqa_model`, returning a {DDI_file: score} dict."""
    items = list(df.itertuples(index=False))
    scores = {}
    for kept, batch_scores in script_utils.predict_raw_items(
        iqa_model, items, image_dir, lambda item: item.DDI_file, batch_size
    ):
        for item, score in zip(kept, batch_scores):
            scores[item.DDI_file] = score
    return scores


def build_agreement_table(df, model_types):
    """Adds a `{model}_pct` percentile-rank column per model (0-1 within this dataset, so models
    with different native scales are comparable) plus a `disagreement` column (std of percentile
    ranks across models for that image — high means the models disagree about this image's
    relative quality). Returns df sorted by disagreement descending."""
    pct_cols = []
    for model_type in model_types:
        pct_col = f"{model_type}_pct"
        df[pct_col] = df[f"{model_type}_score"].rank(pct=True)
        pct_cols.append(pct_col)
    df["disagreement"] = df[pct_cols].std(axis=1)
    return df.sort_values("disagreement", ascending=False).reset_index(drop=True)


def compute_correlations(df, model_types):
    score_cols = [f"{m}_score" for m in model_types]
    pearson_df = df[score_cols].corr(method="pearson")
    spearman_df = df[score_cols].corr(method="spearman")
    pearson_df.index = pearson_df.columns = model_types
    spearman_df.index = spearman_df.columns = model_types
    return pearson_df, spearman_df


def print_correlation_table(title, corr_df):
    print(f"\n=== {title} ===")
    header = f"{'':<14}" + "".join(f"{m:>12}" for m in corr_df.columns)
    print(header)
    for model_type in corr_df.index:
        row = f"{model_type:<14}" + "".join(f"{corr_df.loc[model_type, m]:>12.3f}" for m in corr_df.columns)
        print(row)


def plot_correlation_heatmaps(pearson_df, spearman_df, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, corr_df, title in zip(axes, [pearson_df, spearman_df], ["Pearson r", "Spearman rho"]):
        im = ax.imshow(corr_df.values, cmap="RdYlGn", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_yticklabels(corr_df.index)
        for i, j in itertools.product(range(len(corr_df.index)), range(len(corr_df.columns))):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", fontsize=10)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Pairwise agreement between pretrained IQA models (DDI quality scores)")
    plot_path = os.path.join(output_dir, "ddi_model_agreement_heatmap.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation heatmaps to {plot_path}")


def plot_pairwise_scatter(df, model_types, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = list(itertools.combinations(model_types, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.5), constrained_layout=True,
                              squeeze=False)
    axes = axes[0]
    for ax, (m1, m2) in zip(axes, pairs):
        pearson_r, _ = stats.pearsonr(df[f"{m1}_score"], df[f"{m2}_score"])
        spearman_rho, _ = stats.spearmanr(df[f"{m1}_score"], df[f"{m2}_score"])
        scatter = ax.scatter(df[f"{m1}_score"], df[f"{m2}_score"], c=df["disagreement"],
                              cmap="viridis", alpha=0.7, s=18)
        fig.colorbar(scatter, ax=ax, label="disagreement")
        ax.set_xlabel(f"{m1} quality score")
        ax.set_ylabel(f"{m2} quality score")
        ax.set_title(f"{m1} vs {m2}\nPearson r={pearson_r:.2f}, Spearman rho={spearman_rho:.2f}")
    plot_path = os.path.join(output_dir, "ddi_model_agreement_scatter.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved pairwise scatter plots to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure how much this repo's pretrained IQA models (topiq_nr, musiq, "
                     "qalign_mini) agree with each other on the DDI images, to find where they "
                     "disagree.")
    parser.add_argument("--model-types", nargs="+", choices=script_utils.PRETRAINED_MODEL_TYPES,
                         default=list(script_utils.PRETRAINED_MODEL_TYPES),
                         help="Which pretrained models to compare (default: all of them)")
    parser.add_argument("--image-dir", default=os.path.join(REPO_ROOT, "ddi_data"))
    parser.add_argument("--metadata", default=os.path.join(REPO_ROOT, "ddi_metadata.csv"))
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "validation", "ddi_model_agreement"))
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Chunk size for image loading; these models score one image per "
                              "forward pass regardless, so this only affects I/O batching.")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Cap the number of DDI images scored (for a quick smoke test)")
    parser.add_argument("--top-n", type=int, default=20,
                         help="How many highest-disagreement images to print/highlight")
    args = parser.parse_args()

    if len(args.model_types) < 2:
        raise SystemExit("Need at least 2 --model-types to measure agreement between them")

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.metadata)[["DDI_file", "skin_tone", "malignant", "disease"]]
    if args.max_samples is not None and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=0).reset_index(drop=True)
    print(f"Loaded {len(df)} DDI images from {args.metadata}")

    device = get_device()
    print(f"Device: {device}")

    for model_type in args.model_types:
        print(f"\n{f' Scoring with pretrained model: {model_type} ':=^88}")
        if model_type == "musiq":
            # script_utils.load_model's "musiq" path (script_utils.MusiqModel) reuses this repo's
            # manual reimplementation with a randomly-initialized, never-fine-tuned head — fine as
            # an "untuned baseline" elsewhere, but meaningless to compare against topiq_nr/
            # qalign_mini's real predictions here. Use pyiqa's own pretrained MUSIQ head instead,
            # rescaled from its native ~0-100 koniq10k scale to this repo's 1-5 MOS scale.
            iqa_model = script_utils.MusiqNativeModel(device)
        else:
            model_args = argparse.Namespace(model_type=model_type, model_path=None)
            iqa_model = script_utils.load_model(model_args, device)

        scores = score_model(iqa_model, df, args.image_dir, args.batch_size)
        df[f"{model_type}_score"] = df["DDI_file"].map(scores)

        del iqa_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score_cols = [f"{m}_score" for m in args.model_types]
    n_missing = df[score_cols].isna().any(axis=1).sum()
    if n_missing:
        print(f"Warning: {n_missing} images missing a score from at least one model, dropping them.")
        df = df.dropna(subset=score_cols).reset_index(drop=True)
    if len(df) < 2:
        raise SystemExit("Fewer than 2 images have scores from every model — nothing to correlate")

    df = build_agreement_table(df, args.model_types)

    predictions_path = os.path.join(args.output_dir, "ddi_model_agreement_predictions.csv")
    df.to_csv(predictions_path, index=False)
    print(f"\nSaved per-image predictions (sorted by disagreement) to {predictions_path}")

    pearson_df, spearman_df = compute_correlations(df, args.model_types)
    pearson_df.to_csv(os.path.join(args.output_dir, "ddi_model_agreement_pearson.csv"))
    spearman_df.to_csv(os.path.join(args.output_dir, "ddi_model_agreement_spearman.csv"))
    print_correlation_table("Pearson r between models (DDI quality scores)", pearson_df)
    print_correlation_table("Spearman rho between models (DDI quality scores)", spearman_df)

    print(f"\n=== Top {args.top_n} images by model disagreement ===")
    cols = ["DDI_file", "disagreement"] + score_cols + ["skin_tone", "malignant", "disease"]
    print(df[cols].head(args.top_n).to_string(index=False))

    try:
        plot_correlation_heatmaps(pearson_df, spearman_df, args.output_dir)
        plot_pairwise_scatter(df, args.model_types, args.output_dir)
    except Exception as e:
        print(f"Warning: could not generate plots ({e})")


if __name__ == "__main__":
    main()
