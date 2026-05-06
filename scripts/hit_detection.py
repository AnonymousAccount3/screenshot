"""Hit detection metrics using min(RPE, BJ) as control.

For each treatment, the control viability is min(RPE, BJ).
If only one is available, that one is used. Only treatments
where at least one control sample was measured are kept.

Computes differential viability (sample - control) and evaluates
how well predictions recover the true top hits (differentially
sensitive cell lines).

Metrics:
  - AUROC: discrimination between hits and non-hits
  - Precision@k: fraction of top-k predicted hits that are true hits
  - Recall@k: fraction of true hits recovered in top-k predictions
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

CONTROL_SAMPLES = ["RPE", "BJ"]


def build_treatment_key(df):
    """Add a treatment column from drug+dose columns."""
    drug_cols = [c for c in df.columns if c.startswith("drug")]
    dose_cols = [c for c in df.columns if c.startswith("dose")]
    sub = df.copy()
    for c in drug_cols:
        sub[c] = sub[c].fillna("__NONE__")
    for c in dose_cols:
        sub[c] = sub[c].fillna(0.0)
    sub["treatment"] = sub[drug_cols + dose_cols].apply(tuple, axis=1)
    return sub


def compute_differential_viability(df, controls=None, dose_agg="min"):
    """Compute differential viability vs named control(s).

    Control value = min across controls per treatment. If only one control is
    available for a treatment, that one is used (+inf for the missing one).
    Only treatments where at least one control was measured are kept.

    Args:
        df: DataFrame with sample_id, treatment, float_value, prediction columns.
        controls: list of control sample IDs. Default: CONTROL_SAMPLES.
        dose_agg: how to aggregate delta across doses per drug combo.
            "min" — most sensitive dose point (default, for combo screens).
            "mean" — average across doses (AUC-like, for single-agent screens).

    Returns a DataFrame with columns:
        sample_id, drug_combo, delta_true, delta_pred
    """
    if controls is None:
        controls = CONTROL_SAMPLES

    drug_cols = [c for c in df.columns if c.startswith("drug")]
    dose_cols = [c for c in df.columns if c.startswith("dose")]

    # Average replicates per (sample, treatment)
    group_cols = ["sample_id", "treatment"]
    avg = df.groupby(group_cols + drug_cols + dose_cols).agg(
        true=("float_value", "mean"),
        pred=("prediction", "mean"),
    ).reset_index()

    # Build control table: min(RPE, BJ) per treatment
    ctrl_dfs = []
    for ctrl_name in controls:
        c = avg[avg["sample_id"] == ctrl_name][["treatment", "true", "pred"]].copy()
        c = c.rename(columns={"true": f"true_{ctrl_name}", "pred": f"pred_{ctrl_name}"})
        c = c.set_index("treatment")
        ctrl_dfs.append(c)

    # Outer join so we keep treatments where at least one control exists
    ctrl_merged = ctrl_dfs[0]
    for c in ctrl_dfs[1:]:
        ctrl_merged = ctrl_merged.join(c, how="outer")

    # Min across controls (NaN treated as +inf → use the available one)
    # Always use true values for control baseline (ground truth)
    true_cols = [f"true_{c}" for c in controls]
    ctrl_merged["true_ctrl"] = ctrl_merged[true_cols].min(axis=1)
    ctrl_merged["pred_ctrl"] = ctrl_merged["true_ctrl"]

    # Non-control samples
    others = avg[~avg["sample_id"].isin(controls)]

    # Merge to get control values for each treatment
    merged = others.merge(
        ctrl_merged[["true_ctrl", "pred_ctrl"]],
        left_on="treatment",
        right_index=True,
    )

    # Differential viability per (sample, treatment)
    merged["delta_true"] = merged["true"] - merged["true_ctrl"]
    merged["delta_pred"] = merged["pred"] - merged["pred_ctrl"]

    # Create drug combo key (drugs only, no doses)
    merged["drug_combo"] = merged[drug_cols].apply(tuple, axis=1)

    # Aggregate per (sample, drug_combo) across doses
    combo_agg = merged.groupby(["sample_id", "drug_combo"]).agg(
        delta_true=("delta_true", dose_agg),
        delta_pred=("delta_pred", dose_agg),
    ).reset_index()

    return combo_agg


def compute_differential_viability_median(df):
    """Compute differential viability vs median-across-samples control.

    For datasets without designated control cell lines.
    Control value = median over all samples per treatment.
    Aggregation across doses = mean (AUC-like for single-agent).

    Returns a DataFrame with columns:
        sample_id, drug_combo, delta_true, delta_pred
    where delta = sample_value - median_value (mean across doses).
    """
    drug_cols = [c for c in df.columns if c.startswith("drug")]
    dose_cols = [c for c in df.columns if c.startswith("dose")]

    # Average replicates per (sample, treatment)
    group_cols = ["sample_id", "treatment"]
    avg = df.groupby(group_cols + drug_cols + dose_cols).agg(
        true=("float_value", "mean"),
        pred=("prediction", "mean"),
    ).reset_index()

    # Median across samples per treatment (always use true values as baseline)
    ctrl = avg.groupby("treatment").agg(
        true_ctrl=("true", "median"),
    )
    ctrl["pred_ctrl"] = ctrl["true_ctrl"]

    # Merge control back
    merged = avg.merge(ctrl, left_on="treatment", right_index=True)

    # Differential viability per (sample, treatment)
    merged["delta_true"] = merged["true"] - merged["true_ctrl"]
    merged["delta_pred"] = merged["pred"] - merged["pred_ctrl"]

    # Create drug combo key (drugs only, no doses)
    merged["drug_combo"] = merged[drug_cols].apply(tuple, axis=1)

    # Aggregate per (sample, drug_combo): mean delta across doses (AUC-like)
    combo_agg = merged.groupby(["sample_id", "drug_combo"]).agg(
        delta_true=("delta_true", "mean"),
        delta_pred=("delta_pred", "mean"),
    ).reset_index()

    return combo_agg


def compute_hit_metrics(delta_df, hit_threshold=-0.2, k_values=None):
    """Compute hit detection metrics.

    A "hit" is a (sample, treatment) pair where delta_true < hit_threshold
    (cell line is more sensitive than control).

    Parameters
    ----------
    delta_df : DataFrame with delta_true, delta_pred columns
    hit_threshold : float
        Threshold for defining a hit (default: -0.2)
    k_values : list of int or None
        k values for precision@k and recall@k. If None, uses [50, 100, 200, 500].

    Returns
    -------
    dict with AUROC, precision@k, recall@k, and hit statistics
    """
    if k_values is None:
        k_values = [50, 100, 200, 500]

    # Define hits
    is_hit = (delta_df["delta_true"] < hit_threshold).values
    n_hits = is_hit.sum()
    n_total = len(delta_df)

    if n_hits == 0 or n_hits == n_total:
        return {"n_hits": n_hits, "n_total": n_total, "hit_rate": n_hits / n_total}

    # Score: more negative delta_pred = more likely hit
    scores = -delta_df["delta_pred"].values

    # AUROC
    auroc = roc_auc_score(is_hit, scores)

    # Rank by predicted differential (most negative first = most likely hit)
    ranked_indices = np.argsort(scores)[::-1]
    ranked_hits = is_hit[ranked_indices]

    metrics = {
        "n_hits": int(n_hits),
        "n_total": n_total,
        "hit_rate": n_hits / n_total,
        "AUROC": auroc,
    }

    for k in k_values:
        if k > n_total:
            continue
        top_k_hits = ranked_hits[:k].sum()
        metrics[f"Precision@{k}"] = top_k_hits / k
        metrics[f"Recall@{k}"] = top_k_hits / n_hits

    return metrics


def load_al_wide(dataset_dir, budget, seed):
    """Load a single wide-format AL results file.

    Parameters
    ----------
    dataset_dir : Path or str
        Directory containing b{budget}_s{seed}.feather files.
    budget : int
    seed : int

    Returns
    -------
    pd.DataFrame with raw columns + pred_{method} + sel_{method} columns.
    """
    path = Path(dataset_dir) / f"b{budget}_s{seed}.feather"
    return pd.read_feather(path)


def get_al_methods(df):
    """Return list of method names from a wide-format AL DataFrame."""
    return [c[5:] for c in df.columns if c.startswith("pred_")]


def al_wide_to_long(df, method, pred_col=None, sel_col=None):
    """Extract a single method from wide format, returning a DataFrame with
    'prediction' and 'in_few_shot' columns (compatible with old long format).

    Parameters
    ----------
    df : wide-format AL DataFrame
    method : str (e.g. "random", "cs20_r5")
    pred_col : str or None (default: f"pred_{method}")
    sel_col : str or None (default: f"sel_{method}")
    """
    if pred_col is None:
        pred_col = f"pred_{method}"
    if sel_col is None:
        sel_col = f"sel_{method}"

    # Get non-method columns (raw data columns)
    meta_prefixes = ("pred_", "sel_")
    raw_cols = [c for c in df.columns if not c.startswith(meta_prefixes)]

    result = df[raw_cols].copy()
    result["prediction"] = df[pred_col].values
    if sel_col in df.columns:
        result["in_few_shot"] = df[sel_col].values
    return result


def compute_top_hit_recall_wide(dataset_dir, methods=None, budgets=None,
                                seeds=None, top_pct=0.10, k_pct=0.20,
                                controls=None, control_mode="named"):
    """Compute top-hit recall from wide-format AL results directory.

    Parameters
    ----------
    dataset_dir : Path or str
        Directory containing b{budget}_s{seed}.feather files.
    methods : list of str or None
        Method names to include. None = all found in files.
    budgets : list of int or None
        Budgets to include. None = all found in directory.
    seeds : list of int or None
        Seeds to include. None = all found in directory.
    top_pct, k_pct, controls, control_mode : same as compute_top_hit_recall.

    Returns
    -------
    pd.DataFrame with columns: method, n_few_shots, random_state,
        recall_pct, random_baseline
    """
    if controls is None:
        controls = CONTROL_SAMPLES

    dataset_dir = Path(dataset_dir)
    records = []

    # Discover files
    for fpath in sorted(dataset_dir.glob("b*_s*.feather")):
        parts = fpath.stem.split("_")
        budget = int(parts[0][1:])
        seed = int(parts[1][1:])
        if budgets and budget not in budgets:
            continue
        if seeds and seed not in seeds:
            continue

        df = pd.read_feather(fpath)
        pred_cols = [c for c in df.columns if c.startswith("pred_")]

        for pred_col in pred_cols:
            method = pred_col[5:]
            if methods and method not in methods:
                continue

            # Build a temporary df with 'prediction' column for hit detection
            sub = df.copy()
            sub["prediction"] = sub[pred_col]
            sub = build_treatment_key(sub)

            if control_mode == "named":
                ctrl_treats = set()
                for ctrl_name in controls:
                    ctrl_treats.update(
                        sub[sub["sample_id"] == ctrl_name]["treatment"].unique()
                    )
                sub = sub[sub["treatment"].isin(ctrl_treats)]
                if len(sub) == 0:
                    continue
                delta_df = compute_differential_viability(sub, controls=controls)
            elif control_mode == "median":
                delta_df = compute_differential_viability_median(sub)
            else:
                raise ValueError(f"Unknown control_mode: {control_mode}")

            sample_recalls = []
            sample_baselines = []
            for sample_id in delta_df["sample_id"].unique():
                s = delta_df[delta_df["sample_id"] == sample_id]
                n_total = len(s)
                if n_total == 0:
                    continue
                n_hits = max(1, int(np.ceil(top_pct * n_total)))
                hit_threshold = s["delta_true"].nsmallest(n_hits).iloc[-1]
                is_hit = s["delta_true"] <= hit_threshold
                actual_n_hits = is_hit.sum()
                if actual_n_hits == 0:
                    continue
                k_eff = max(1, int(np.ceil(k_pct * n_total)))
                top_k_idx = s["delta_pred"].nsmallest(k_eff).index
                hits_in_top_k = is_hit.loc[top_k_idx].sum()
                sample_recalls.append(hits_in_top_k / actual_n_hits)
                sample_baselines.append(k_eff / n_total)

            if sample_recalls:
                records.append({
                    "method": method,
                    "n_few_shots": budget,
                    "random_state": seed,
                    "recall_pct": np.mean(sample_recalls) * 100,
                    "random_baseline": np.mean(sample_baselines) * 100,
                })

    return pd.DataFrame(records)


def compute_top_hit_recall(results_dict, top_pct=0.10, k_pct=0.20,
                           controls=None, control_mode="named", dose_agg="min"):
    """Quantile-based top-hit recall across methods and n_few_shots.

    For each (method, n_few_shots, random_state):
      1. Compute differential viability per (sample, drug_combo)
      2. Per sample: top_pct most negative delta_true = true hits
      3. Sort by delta_pred (most negative first), take top k_pct fraction
      4. recall = (# true hits in top k) / (# true hits)
      5. Average recall across samples

    Parameters
    ----------
    results_dict : dict[str, pd.DataFrame]
        Method name -> results DataFrame (with float_value, prediction,
        n_few_shots, random_state, sample_id, drug/dose columns).
    top_pct : float
        Quantile for defining true hits (default 0.10 = top 10%).
    k_pct : float
        Fraction of top predicted entries to check (default 0.20 = top 20%).
    controls : list of str or None
        Control sample names (default: CONTROL_SAMPLES). Only used when
        control_mode="named".
    control_mode : str
        "named" — use named control cell lines (min of RPE, BJ), aggregation
            across doses by min (batchie).
        "median" — use median across all samples as control, aggregation
            across doses by mean (monotherapy).

    Returns
    -------
    pd.DataFrame with columns:
        method, n_few_shots, random_state, recall_pct, random_baseline
    """
    if controls is None:
        controls = CONTROL_SAMPLES

    records = []

    for method, df in results_dict.items():
        df = build_treatment_key(df)

        for n_few in sorted(df["n_few_shots"].unique()):
            for rs in df["random_state"].unique():
                sub = df[
                    (df["n_few_shots"] == n_few)
                    & (df["random_state"] == rs)
                ]
                if len(sub) == 0:
                    continue

                if control_mode == "named":
                    # Filter to treatments where at least one control is present
                    ctrl_treats = set()
                    for ctrl_name in controls:
                        ctrl_treats.update(
                            sub[sub["sample_id"] == ctrl_name]["treatment"].unique()
                        )
                    sub = sub[sub["treatment"].isin(ctrl_treats)]
                    if len(sub) == 0:
                        continue
                    delta_df = compute_differential_viability(sub, controls=controls, dose_agg=dose_agg)
                elif control_mode == "median":
                    delta_df = compute_differential_viability_median(sub)
                else:
                    raise ValueError(f"Unknown control_mode: {control_mode}")

                # Per-sample recall
                sample_recalls = []
                sample_baselines = []

                for sample_id in delta_df["sample_id"].unique():
                    s = delta_df[delta_df["sample_id"] == sample_id]
                    n_total = len(s)
                    if n_total == 0:
                        continue

                    # Top hits: top_pct most negative delta_true
                    n_hits = max(1, int(np.ceil(top_pct * n_total)))
                    hit_threshold = s["delta_true"].nsmallest(n_hits).iloc[-1]
                    is_hit = s["delta_true"] <= hit_threshold
                    actual_n_hits = is_hit.sum()

                    if actual_n_hits == 0:
                        continue

                    # Sort by delta_pred (most negative first), take top k_pct
                    k_eff = max(1, int(np.ceil(k_pct * n_total)))
                    top_k_idx = s["delta_pred"].nsmallest(k_eff).index
                    hits_in_top_k = is_hit.loc[top_k_idx].sum()

                    recall = hits_in_top_k / actual_n_hits
                    sample_recalls.append(recall)

                    # Random baseline: expected recall ≈ k_pct
                    random_recall = k_eff / n_total
                    sample_baselines.append(random_recall)

                if sample_recalls:
                    records.append({
                        "method": method,
                        "n_few_shots": n_few,
                        "random_state": rs,
                        "recall_pct": np.mean(sample_recalls) * 100,
                        "random_baseline": np.mean(sample_baselines) * 100,
                    })

    return pd.DataFrame(records)


def analyze_results(results_dir, methods, dataset_name, n_few_shots_list,
                    random_states, hit_threshold=-0.2, k_values=None):
    """Run hit detection analysis across methods, n_few_shots, and random states.

    Parameters
    ----------
    results_dir : Path
        Directory containing result feather files
    methods : list of str
        Method names (e.g. ["screenshot", "xgb"])
    dataset_name : str
        Dataset name for file pattern (e.g. "batchie")
    n_few_shots_list : list of int
    random_states : list of int
    hit_threshold : float
    k_values : list of int or None

    Returns
    -------
    pd.DataFrame with all metrics
    """
    all_records = []

    for method in methods:
        fpath = Path(results_dir) / f"{method}_{dataset_name}.feather"
        if not fpath.exists():
            print(f"  Warning: {fpath} not found, skipping")
            continue

        df = pd.read_feather(fpath)
        df = build_treatment_key(df)

        for n_few in n_few_shots_list:
            for rs in random_states:
                sub = df[
                    (df["n_few_shots"] == n_few)
                    & (df["random_state"] == rs)
                ]
                if len(sub) == 0:
                    continue

                # Filter to treatments where at least one control is present
                ctrl_treats = set()
                for ctrl_name in CONTROL_SAMPLES:
                    ctrl_treats.update(
                        sub[sub["sample_id"] == ctrl_name]["treatment"].unique()
                    )
                sub = sub[sub["treatment"].isin(ctrl_treats)]

                # Compute differential viability
                delta_df = compute_differential_viability(sub)

                # Compute hit metrics
                metrics = compute_hit_metrics(
                    delta_df, hit_threshold=hit_threshold, k_values=k_values
                )
                metrics["method"] = method
                metrics["n_few_shots"] = n_few
                metrics["random_state"] = rs
                all_records.append(metrics)

    return pd.DataFrame(all_records)


def print_summary(metrics_df, methods):
    """Print a summary table of hit detection metrics."""
    print(f"\nHit Detection Metrics (control=min({', '.join(CONTROL_SAMPLES)}))")
    print("=" * 90)

    # Get k columns
    k_cols = [c for c in metrics_df.columns if c.startswith("Precision@") or c.startswith("Recall@")]
    metric_cols = ["AUROC"] + sorted(k_cols)

    for n_few in sorted(metrics_df["n_few_shots"].unique()):
        print(f"\nn_few_shots = {n_few}")
        n_hits = metrics_df[metrics_df["n_few_shots"] == n_few]["n_hits"].iloc[0]
        n_total = metrics_df[metrics_df["n_few_shots"] == n_few]["n_total"].iloc[0]
        print(f"  Hits: {n_hits} / {n_total} ({n_hits/n_total*100:.1f}%)")

        header = f"{'Method':<15}"
        for col in metric_cols:
            header += f"  {col:>12}"
        print(header)
        print("-" * len(header))

        for method in methods:
            sub = metrics_df[
                (metrics_df["method"] == method)
                & (metrics_df["n_few_shots"] == n_few)
            ]
            if len(sub) == 0:
                continue

            line = f"{method:<15}"
            for col in metric_cols:
                if col in sub.columns:
                    mean = sub[col].mean()
                    std = sub[col].std()
                    line += f"  {mean:>5.3f}±{std:.3f}"
                else:
                    line += f"  {'N/A':>12}"
            print(line)


def main():
    parser = argparse.ArgumentParser(description="Hit detection metrics")
    parser.add_argument("--results-dir", type=str, default="results/few_shot_batchie")
    parser.add_argument("--dataset", type=str, default="batchie")
    parser.add_argument("--methods", nargs="+", default=["screenshot", "xgb"])
    parser.add_argument("--hit-threshold", type=float, default=-0.5)
    parser.add_argument("--k-values", nargs="+", type=int, default=[50, 100, 200, 500])
    args = parser.parse_args()

    # Use config values
    n_few_shots_list = [10, 50, 100]
    random_states = [42, 123, 456, 789, 1024]

    metrics_df = analyze_results(
        results_dir=args.results_dir,
        methods=args.methods,
        dataset_name=args.dataset,
        n_few_shots_list=n_few_shots_list,
        random_states=random_states,
        hit_threshold=args.hit_threshold,
        k_values=args.k_values,
    )

    print_summary(metrics_df, args.methods)


if __name__ == "__main__":
    main()
