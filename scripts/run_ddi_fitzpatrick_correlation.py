"""Run any of this repo's pretrained IQA models on the DDI images and correlate their quality
scores against the directly-predicted Fitzpatrick skin type (I-VI) from
third_party/nn_colorimetry_dermatoscopy (see ddi_fitzpatrick_inference.py there) — as opposed to
run_ddi_ita_correlation.py, which correlates against the continuous ITA-formula-derived score.

Requires validation/ddi_fitzpatrick/ddi_fitzpatrick_predictions.csv to already exist (generate it
with `uv run third_party/nn_colorimetry_dermatoscopy/ddi_fitzpatrick_inference.py` first).

Usage:
    python scripts/run_ddi_fitzpatrick_correlation.py [--model-types TYPE [TYPE ...]] [--image-dir DIR]
        [--fitzpatrick-predictions CSV] [--output-dir DIR] [--max-samples N]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, ".")
from src.trainer import get_device
import script_utils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def correlate_model(iqa_model, model_type, fp_df, image_dir, output_dir, batch_size):
    """Scores every row of `fp_df` with `iqa_model`, then correlates the resulting quality
    scores against the `fitzpatrick` column. Saves per-image predictions and correlation stats
    under `output_dir`, named by `model_type` so a --model-types sweep doesn't clobber files
    across models. Returns a metrics dict, or None if no images could be scored."""
    items = list(fp_df.itertuples(index=False))
    scores = {}
    for kept, batch_scores in script_utils.predict_raw_items(
        iqa_model, items, image_dir, lambda item: item.DDI_file, batch_size
    ):
        for item, score in zip(kept, batch_scores):
            scores[item.DDI_file] = score

    df = fp_df.copy()
    df["quality_score"] = df["DDI_file"].map(scores)
    n_missing = df["quality_score"].isna().sum()
    if n_missing:
        print(f"Warning: {n_missing} images had no {model_type} score (missing/unreadable), dropping them.")
        df = df.dropna(subset=["quality_score"]).reset_index(drop=True)
    if len(df) < 2:
        print(f"WARNING: fewer than 2 usable samples for {model_type}, skipping")
        return None

    predictions_path = os.path.join(output_dir, f"{model_type}_fitzpatrick_predictions.csv")
    df.to_csv(predictions_path, index=False)
    print(f"Saved per-image {model_type}/Fitzpatrick predictions to {predictions_path}")

    pearson_r, pearson_p = stats.pearsonr(df["quality_score"], df["fitzpatrick"])
    spearman_rho, spearman_p = stats.spearmanr(df["quality_score"], df["fitzpatrick"])
    kw_h, kw_p = stats.kruskal(*[g["quality_score"].values for _, g in df.groupby("fitzpatrick")])

    stats_df = pd.DataFrame([
        {"statistic": "pearson_r", "value": pearson_r, "p_value": pearson_p},
        {"statistic": "spearman_rho", "value": spearman_rho, "p_value": spearman_p},
        {"statistic": "kruskal_wallis_H", "value": kw_h, "p_value": kw_p},
        {"statistic": "n", "value": len(df), "p_value": np.nan},
    ])
    stats_path = os.path.join(output_dir, f"{model_type}_fitzpatrick_correlation_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved correlation stats to {stats_path}")
    print(f"\n=== Correlation between {model_type} quality and predicted Fitzpatrick type (DDI) ===")
    print(f"N:            {len(df)}")
    print(f"Pearson r:    {pearson_r:.3f} (p = {pearson_p:.2e})")
    print(f"Spearman rho: {spearman_rho:.3f} (p = {spearman_p:.2e})")
    print(f"Kruskal-Wallis H: {kw_h:.1f} (p = {kw_p:.2e})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        fp_levels = sorted(df["fitzpatrick"].unique())
        ax.boxplot([df.loc[df["fitzpatrick"] == fp, "quality_score"] for fp in fp_levels],
                   tick_labels=fp_levels)
        ax.set_xlabel("Predicted Fitzpatrick type (I-VI, nn_colorimetry_dermatoscopy)")
        ax.set_ylabel(f"{model_type} quality score")
        ax.set_title(f"{model_type} quality vs. predicted Fitzpatrick type (DDI)\n"
                      f"Pearson r={pearson_r:.2f}, Spearman rho={spearman_rho:.2f}")
        plot_path = os.path.join(output_dir, f"{model_type}_vs_fitzpatrick_boxplot.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved box plot to {plot_path}")
    except Exception as e:
        print(f"Warning: could not generate plot ({e})")

    return {"n": len(df), "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_rho": spearman_rho, "spearman_p": spearman_p}


def print_summary_table(all_metrics):
    header = f"{'Model':<14}{'N':>6}{'Pearson r':>12}{'Spearman rho':>14}"
    print("\n=== Summary: quality vs. predicted Fitzpatrick type (DDI), by model ===")
    print(header)
    print("-" * len(header))
    for model_type, m in all_metrics.items():
        print(f"{model_type:<14}{m['n']:>6}{m['pearson_r']:>12.3f}{m['spearman_rho']:>14.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Correlate one or more pretrained IQA models' quality scores with directly "
                     "predicted Fitzpatrick skin type on the DDI dataset.")
    parser.add_argument("--model-types", nargs="+", choices=script_utils.PRETRAINED_MODEL_TYPES,
                         default=list(script_utils.PRETRAINED_MODEL_TYPES),
                         help="Which pretrained models to evaluate (default: all of them)")
    parser.add_argument("--image-dir", default=os.path.join(REPO_ROOT, "ddi_data"))
    parser.add_argument("--fitzpatrick-predictions",
                         default=os.path.join(REPO_ROOT, "validation", "ddi_fitzpatrick", "ddi_fitzpatrick_predictions.csv"),
                         help="Per-image Fitzpatrick predictions from "
                              "third_party/nn_colorimetry_dermatoscopy/ddi_fitzpatrick_inference.py")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "validation", "ddi_fitzpatrick"))
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Chunk size for image loading; these models score one image per "
                              "forward pass regardless, so this only affects I/O batching.")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Cap the number of DDI images scored (for a quick smoke test)")
    args = parser.parse_args()

    if not os.path.exists(args.fitzpatrick_predictions):
        raise SystemExit(
            f"{args.fitzpatrick_predictions} not found. Generate it first with "
            f"`uv run third_party/nn_colorimetry_dermatoscopy/ddi_fitzpatrick_inference.py`."
        )
    os.makedirs(args.output_dir, exist_ok=True)

    fp_df = pd.read_csv(args.fitzpatrick_predictions)
    fp_df = fp_df[["DDI_file", "skin_tone", "malignant", "disease", "fitzpatrick"]]
    if args.max_samples is not None and len(fp_df) > args.max_samples:
        fp_df = fp_df.sample(n=args.max_samples, random_state=0).reset_index(drop=True)
    print(f"Loaded {len(fp_df)} Fitzpatrick predictions from {args.fitzpatrick_predictions}")

    device = get_device()
    print(f"Device: {device}")

    all_metrics = {}
    for model_type in args.model_types:
        print(f"\n{f' Evaluating pretrained model: {model_type} ':=^88}")
        model_args = argparse.Namespace(model_type=model_type, model_path=None)
        iqa_model = script_utils.load_model(model_args, device)

        metrics = correlate_model(iqa_model, model_type, fp_df, args.image_dir, args.output_dir,
                                   args.batch_size)
        if metrics is not None:
            all_metrics[model_type] = metrics

        del iqa_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_metrics:
        raise SystemExit("No model produced usable correlation results")

    if len(all_metrics) > 1:
        print_summary_table(all_metrics)


if __name__ == "__main__":
    main()
