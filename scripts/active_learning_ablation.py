"""
Usage:
    python active_learning_ablation.py --config config_active_learning.yaml --data-config config_batchie.yaml --input data.feather
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from onebatch import OneBatchPAM
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from experiment import load_data, load_model, compute_metrics
from hit_detection import build_treatment_key, compute_differential_viability


def kmeans_plusplus_frozen(X, n_clusters, *, sample_weight=None,
                          frozen_centers=None, random_state=None,
                          n_local_trials=None):
    """K-means++ with optional frozen centers."""
    if isinstance(random_state, (int, np.integer)):
        rng = np.random.RandomState(random_state)
    elif random_state is None:
        rng = np.random.RandomState()
    else:
        rng = random_state

    n_samples, n_features = X.shape

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(max(n_clusters, 2)))

    if n_clusters >= n_samples:
        return np.arange(n_samples)

    X = np.asarray(X, dtype=np.float64)
    x_sq = np.sum(X ** 2, axis=1)

    indices = np.empty(n_clusters, dtype=int)

    if frozen_centers is not None and len(frozen_centers) > 0:
        frozen_centers = np.asarray(frozen_centers, dtype=np.float64)
        f_sq = np.sum(frozen_centers ** 2, axis=1)
        dists = x_sq[:, None] - 2.0 * (X @ frozen_centers.T) + f_sq[None, :]
        np.maximum(dists, 0.0, out=dists)
        closest_dist_sq = dists.min(axis=1)
        current_pot = float(closest_dist_sq @ sample_weight)
        start = 0
    else:
        prob = sample_weight / sample_weight.sum()
        center_id = rng.choice(n_samples, p=prob)
        indices[0] = center_id
        closest_dist_sq = x_sq - 2.0 * (X @ X[center_id]) + x_sq[center_id]
        np.maximum(closest_dist_sq, 0.0, out=closest_dist_sq)
        current_pot = float(closest_dist_sq @ sample_weight)
        start = 1

    for c in range(start, n_clusters):
        rand_vals = rng.uniform(size=n_local_trials) * current_pot
        weighted_dist = sample_weight * closest_dist_sq
        cumsum = np.cumsum(weighted_dist)
        candidate_ids = np.searchsorted(cumsum, rand_vals)
        np.clip(candidate_ids, 0, n_samples - 1, out=candidate_ids)

        cand_vecs = X[candidate_ids]
        cand_sq = x_sq[candidate_ids]
        dist_to_cands = x_sq[None, :] - 2.0 * (cand_vecs @ X.T) + cand_sq[:, None]
        np.maximum(dist_to_cands, 0.0, out=dist_to_cands)

        np.minimum(closest_dist_sq[None, :], dist_to_cands, out=dist_to_cands)

        candidates_pot = dist_to_cands @ sample_weight
        best = np.argmin(candidates_pot)

        closest_dist_sq = dist_to_cands[best]
        current_pot = float(candidates_pot[best])
        indices[c] = candidate_ids[best]

    return indices


def get_treatment_cols(n_drugs):
    return ([f"drug{i}" for i in range(1, n_drugs + 1)]
            + [f"dose{i}" for i in range(1, n_drugs + 1)])


def extract_unique_treatments(input_df, n_drugs):
    tcols = get_treatment_cols(n_drugs)
    unique_treatments_df = (
        input_df[tcols]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_treatments_df["treatment_id"] = np.arange(len(unique_treatments_df))
    merged = input_df[tcols].reset_index(drop=True).merge(
        unique_treatments_df, on=tcols, how="left",
    )
    row_treatment_ids = merged["treatment_id"].values
    assert len(row_treatment_ids) == len(input_df)
    return unique_treatments_df, row_treatment_ids


def build_sample_treatment_map(input_df, row_treatment_ids):
    sample_info = {}
    positions = np.arange(len(input_df))
    df_indices = input_df.index.values

    for sample_id in input_df["sample_id"].unique():
        mask = (input_df["sample_id"].values == sample_id)
        sample_positions = positions[mask]
        sample_df_indices = df_indices[mask]
        sample_tids = row_treatment_ids[sample_positions]

        tid_to_rows = {}
        tid_to_positions = {}
        for pos, df_idx, tid in zip(sample_positions, sample_df_indices, sample_tids):
            tid_to_rows.setdefault(tid, []).append(df_idx)
            tid_to_positions.setdefault(tid, []).append(pos)

        sample_info[sample_id] = {
            "treatment_ids": np.array(sorted(tid_to_rows.keys())),
            "tid_to_rows": tid_to_rows,
            "tid_to_positions": tid_to_positions,
        }

    return sample_info


def treatments_to_row_indices(sample_info, per_sample_tids, random_state):
    rng = np.random.RandomState(random_state)
    selected_indices = []
    for sample_id, tids in per_sample_tids.items():
        tid_to_rows = sample_info[sample_id]["tid_to_rows"]
        for tid in tids:
            rows = tid_to_rows[tid]
            if len(rows) == 1:
                selected_indices.append(rows[0])
            else:
                selected_indices.append(rng.choice(rows))
    return selected_indices


def get_candidates(sample_info, sample_id, already_selected, step_size):
    all_tids = sample_info[sample_id]["treatment_ids"]
    already = already_selected.get(sample_id, set())
    candidate_tids = np.array([t for t in all_tids if t not in already])

    if len(candidate_tids) == 0:
        return candidate_tids, True
    if len(candidate_tids) <= step_size:
        return candidate_tids, True
    return candidate_tids, False


def compute_adaptive_steps(n_remaining, n_rounds):
    cumulative = [round(i * n_remaining / n_rounds)
                  for i in range(1, n_rounds + 1)]
    steps = [cumulative[0]]
    for i in range(1, len(cumulative)):
        steps.append(cumulative[i] - cumulative[i - 1])
    return steps


def build_final_treatment_emb(final_emb, sample_info):
    result = {}
    for sample_id, info in sample_info.items():
        tid_embs = {}
        for tid, positions in info["tid_to_positions"].items():
            tid_embs[tid] = final_emb[positions].mean(axis=0)
        result[sample_id] = tid_embs
    return result



def compute_predicted_control_baseline(sample_info, predictions, control_samples):
    control_preds = {}
    for ctrl in control_samples:
        if ctrl not in sample_info:
            continue
        info = sample_info[ctrl]
        for tid in info["treatment_ids"]:
            positions = info["tid_to_positions"][tid]
            pred_val = float(predictions[positions].mean())
            control_preds.setdefault(tid, []).append(pred_val)

    return {tid: min(vals) for tid, vals in control_preds.items()}


def compute_treatment_delta(sample_info, predictions, control_baseline, control_samples):
    result = {}
    for sample_id, info in sample_info.items():
        if sample_id in control_samples:
            continue
        tid_delta = {}
        for tid in info["treatment_ids"]:
            if tid not in control_baseline:
                tid_delta[tid] = 0.0
                continue
            positions = info["tid_to_positions"][tid]
            pred_val = float(predictions[positions].mean())
            tid_delta[tid] = pred_val - control_baseline[tid]
        result[sample_id] = tid_delta

    return result


def compute_hit_recall_named(raw_df, predictions, control_samples, dose_agg="min", top_pct=0.10, k_pct=0.20):
    result = raw_df.copy()
    result["prediction"] = predictions
    result = build_treatment_key(result)

    ctrl_treats = set()
    for ctrl_name in control_samples:
        ctrl_treats.update(
            result[result["sample_id"] == ctrl_name]["treatment"].unique()
        )
    result = result[result["treatment"].isin(ctrl_treats)]
    if len(result) == 0:
        return None

    delta_df = compute_differential_viability(result, controls=control_samples, dose_agg=dose_agg)

    sample_recalls = []
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

    if sample_recalls:
        return np.mean(sample_recalls) * 100
    return None



def select_batch(strategy, sample_info, dd_embeddings, final_treatment_emb,
                 treatment_delta, treatment_uncertainty,
                 already_selected, step_size, random_state, control_samples):
    embedding = strategy["embedding"]
    weight = strategy["weight"]
    selector = strategy["selector"]

    rng = np.random.RandomState(random_state)
    per_sample_new = {}

    for sample_id in sorted(sample_info.keys()):
        candidate_tids, trivial = get_candidates(
            sample_info, sample_id, already_selected, step_size)
        if trivial:
            per_sample_new[sample_id] = candidate_tids.tolist()
            continue

        if sample_id in control_samples:
            sub_emb = dd_embeddings[candidate_tids].astype(np.float32)
            km = OneBatchPAM(n_medoids=step_size, random_state=random_state)
            km.fit(sub_emb)
            per_sample_new[sample_id] = candidate_tids[km.medoid_indices_].tolist()
            continue

        if selector == "random":
            chosen = rng.choice(len(candidate_tids), size=step_size, replace=False)
            per_sample_new[sample_id] = candidate_tids[chosen].tolist()
            continue

        if selector == "top_delta":
            deltas = np.array([treatment_delta[sample_id][t]
                               for t in candidate_tids])
            top_idx = np.argsort(deltas)[:step_size]
            per_sample_new[sample_id] = candidate_tids[top_idx].tolist()
            continue

        if selector == "top_uncertainty":
            unc = np.array([treatment_uncertainty[sample_id][t]
                            for t in candidate_tids])
            top_idx = np.argsort(-unc)[:step_size]
            per_sample_new[sample_id] = candidate_tids[top_idx].tolist()
            continue


        if embedding == "dd":
            sub_emb = dd_embeddings[candidate_tids].astype(np.float64)
        elif embedding == "final":
            se = final_treatment_emb[sample_id]
            sub_emb = np.array([se[int(t)] for t in candidate_tids])
        else:
            raise ValueError(f"Unknown embedding: {embedding}")


        if weight == "uniform":
            w = np.ones(len(candidate_tids))
        elif weight == "delta":
            deltas = np.array([treatment_delta[sample_id][t]
                               for t in candidate_tids])
            w = np.maximum(-deltas, 1e-6)
        elif weight == "smooth_delta":
            deltas = np.array([treatment_delta[sample_id][t]
                               for t in candidate_tids])
            d_min, d_max = deltas.min(), deltas.max()
            if d_max - d_min > 1e-12:
                w = ((d_max - deltas) / (d_max - d_min)) ** 2
            else:
                w = np.ones(len(candidate_tids))
            w = np.maximum(w, 1e-6)
        elif weight == "uncertainty":
            unc = np.array([treatment_uncertainty[sample_id][t]
                            for t in candidate_tids])
            w = np.maximum(unc, 1e-6)
        else:
            raise ValueError(f"Unknown weight: {weight}")


        if selector == "kmedoids":
            sub_emb_f32 = sub_emb.astype(np.float32)

            already_tids = list(already_selected[sample_id])
            frozen = None
            if already_tids:
                if embedding == "dd":
                    frozen = dd_embeddings[np.array(already_tids)].astype(np.float32)
                else:
                    se = final_treatment_emb[sample_id]
                    frozen = np.array([se[int(t)] for t in already_tids], dtype=np.float32)
            km = OneBatchPAM(n_medoids=step_size, random_state=random_state)
            km.fit(sub_emb_f32, sample_weight=w, frozen_medoids=frozen)
            per_sample_new[sample_id] = candidate_tids[km.medoid_indices_].tolist()

        elif selector == "kmeanspp":

            already_tids = list(already_selected[sample_id])
            frozen = None
            if already_tids:
                if embedding == "dd":
                    frozen = dd_embeddings[np.array(already_tids)].astype(np.float64)
                else:
                    se = final_treatment_emb[sample_id]
                    frozen = np.array([se[int(t)] for t in already_tids])

            sel_idx = kmeans_plusplus_frozen(
                sub_emb, step_size, sample_weight=w,
                frozen_centers=frozen, random_state=random_state,
            )
            per_sample_new[sample_id] = candidate_tids[sel_idx].tolist()

        else:
            raise ValueError(f"Unknown selector: {selector}")

    return per_sample_new


def compute_treatment_uncertainty(sample_info, instance_uncertainty, control_samples):
    result = {}
    for sample_id, info in sample_info.items():
        if sample_id in control_samples:
            continue
        tid_unc = {}
        for tid, positions in info["tid_to_positions"].items():
            tid_unc[tid] = float(instance_uncertainty[positions].mean())
        result[sample_id] = tid_unc
    return result


def enable_dropout_on_sample_encoder(model, dropout_p):
    sample_encoder = model.drugscreen_encoder.sample_encoder
    for module in sample_encoder.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p


def compute_mc_dropout_uncertainty(screenshot, input_df, labeled_df, n_drugs,
                                   sample_info, control_samples,
                                   n_passes=10, dropout_p=0.1):
    model = screenshot.model

    enable_dropout_on_sample_encoder(model, dropout_p)

    all_preds = []
    for _ in range(n_passes):
        model.eval()
        model.drugscreen_encoder.sample_encoder.train()

        with torch.no_grad():
            df_pred = screenshot.predict(
                df_input=labeled_df, df_query=input_df,
                n_drugs=n_drugs, sample_id_col="sample_id",
                col_viability="float_value", show_progress=False,
            )
        all_preds.append(df_pred["predicted_viability"].values)


    enable_dropout_on_sample_encoder(model, 0.0)
    model.eval()

    instance_uncertainty = np.std(all_preds, axis=0)
    return compute_treatment_uncertainty(
        sample_info, instance_uncertainty, control_samples)



def main():
    parser = argparse.ArgumentParser(
        description="Active learning ablation with configurable batch strategies"
    )
    parser.add_argument("--config", required=True, help="Path to AL config (config_active_learning.yaml)")
    parser.add_argument("--data-config", required=True, help="Path to dataset config (e.g. config_batchie.yaml)")
    parser.add_argument("--input", required=True, help="Path to input .feather file")
    parser.add_argument("--device", default=None, help="Override device (e.g. cpu, cuda, cuda:0)")
    args = parser.parse_args()


    with open(args.config) as f:
        abl_config = yaml.safe_load(f)

    all_methods = abl_config["methods"]
    budgets = abl_config["budgets"]
    random_states = abl_config["random_states"]
    output_dir = Path(abl_config["output_dir"])


    data = load_data(args.data_config, args.input)
    config = data["config"]
    raw_df = data["raw_df"]
    input_df = data["input_df"]
    drug_library = data["drug_library"]
    n_drugs = data["n_drugs"]


    if args.device:
        if "screenshot" in config:
            config["screenshot"]["device"] = args.device


    control_samples = config["data"]["control_samples"]
    dose_agg = config["data"].get("dose_agg", "min")
    hit_k_pct = abl_config.get("hit_recall_k_pct", 0.20)

    print(f"\n  Control samples: {control_samples}")
    print(f"  Dose aggregation: {dose_agg}")

    model_assets = load_model("screenshot", config, drug_library, input_df)
    screenshot = model_assets["screenshot"]


    print("\n" + "=" * 80)
    print("Extracting unique treatments and computing embeddings...")
    print("=" * 80)

    unique_treatments_df, row_treatment_ids = extract_unique_treatments(input_df, n_drugs)
    print(f"  Unique treatments: {len(unique_treatments_df)}")

    dd_embeddings = screenshot.get_drug_dose_embeddings(
        unique_treatments_df, n_drugs=n_drugs, show_progress=True,
    )
    print(f"  Drug-dose embeddings: {dd_embeddings.shape}")

    sample_info = build_sample_treatment_map(input_df, row_treatment_ids)
    print(f"  Samples: {len(sample_info)}")



    input_filename = Path(args.input).stem
    dataset_dir = output_dir / input_filename
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Methods: {[m['name'] for m in all_methods]}")
    print(f"  Budgets: {budgets}")
    print(f"  Seeds: {random_states}")
    print(f"  Output: {dataset_dir}")


    for random_state in random_states:
        print(f"\n{'#' * 80}")
        print(f"# Seed: {random_state}")
        print(f"{'#' * 80}")

        for budget in budgets:
            print(f"\n{'=' * 80}")
            print(f"Budget: {budget}, seed: {random_state}")
            print(f"{'=' * 80}")

            result_df = raw_df.copy()

            for method_def in all_methods:
                method_name = method_def["name"]
                batch1_strategy = method_def["batch1"]
                adaptive_strategy = method_def.get("adaptive")
                proportion = method_def.get("proportion_cold_start", 1.0)
                n_rounds = method_def.get("n_adaptive_rounds", 1)

                n_cold = round(proportion * budget)

                if adaptive_strategy is None or proportion == 1.0:

                    col_name = method_name
                    print(f"\n--- {col_name} (n={n_cold}) ---")


                    already_selected = {sid: set() for sid in sample_info}
                    batch1_tids = select_batch(
                        strategy=batch1_strategy,
                        sample_info=sample_info,
                        dd_embeddings=dd_embeddings,
                        final_treatment_emb=None,
                        treatment_delta={},
                        treatment_uncertainty={},
                        already_selected=already_selected,
                        step_size=n_cold,
                        random_state=random_state,
                        control_samples=control_samples,
                    )
                    row_indices = treatments_to_row_indices(
                        sample_info, batch1_tids, random_state=random_state)
                    row_indices = sorted(set(row_indices))


                    labeled_df = input_df.loc[row_indices].copy()
                    df_pred = screenshot.predict(
                        df_input=labeled_df, df_query=input_df,
                        n_drugs=n_drugs, sample_id_col="sample_id",
                        col_viability="float_value", show_progress=False,
                    )
                    predictions = df_pred["predicted_viability"].values

                    result_df[f"pred_{col_name}"] = predictions
                    result_df[f"sel_{col_name}"] = input_df.index.isin(row_indices)

                    metrics = compute_metrics(raw_df["float_value"].values, predictions)
                    hit_recall = compute_hit_recall_named(
                        raw_df, predictions, control_samples, dose_agg=dose_agg, k_pct=hit_k_pct)
                    hr_str = f"{hit_recall:.1f}%" if hit_recall is not None else "N/A"
                    print(f"  MAE: {metrics['MAE']:.4f}  Pearson r: {metrics['Correlation']:.4f}  Hit Recall@{int(hit_k_pct*100)}%: {hr_str}")
                    continue


                col_name = method_name
                steps = compute_adaptive_steps(budget - n_cold, n_rounds)
                print(f"\n--- {col_name} (cold={n_cold}, steps={steps}) ---")


                already_selected = {sid: set() for sid in sample_info}
                batch1_tids = select_batch(
                    strategy=batch1_strategy,
                    sample_info=sample_info,
                    dd_embeddings=dd_embeddings,
                    final_treatment_emb=None,
                    treatment_delta={},
                    treatment_uncertainty={},
                    already_selected=already_selected,
                    step_size=n_cold,
                    random_state=random_state,
                    control_samples=control_samples,
                )
                for sid, tids in batch1_tids.items():
                    already_selected[sid].update(tids)

                cold_row_indices = treatments_to_row_indices(
                    sample_info, batch1_tids, random_state=random_state)
                all_row_indices = sorted(set(cold_row_indices))


                labeled_df = input_df.loc[all_row_indices].copy()
                cold_result = screenshot.predict(
                    df_input=labeled_df, df_query=input_df,
                    n_drugs=n_drugs, sample_id_col="sample_id",
                    col_viability="float_value",
                    return_embeddings=True, show_progress=False,
                )
                current_predictions = cold_result["predictions"]["predicted_viability"].values
                current_final_emb = cold_result["embeddings"]


                for round_idx, step in enumerate(steps, 1):
                    pred_ctrl_baseline = compute_predicted_control_baseline(
                        sample_info, current_predictions, control_samples)

                    treatment_delta = compute_treatment_delta(
                        sample_info, current_predictions,
                        control_baseline=pred_ctrl_baseline,
                        control_samples=control_samples,
                    )

                    final_treatment_emb = build_final_treatment_emb(
                        current_final_emb, sample_info,
                    )


                    treatment_uncertainty = {}
                    needs_unc = (
                        adaptive_strategy.get("weight") == "uncertainty"
                        or adaptive_strategy.get("selector") in ("top_uncertainty",)
                    )
                    if needs_unc:
                        treatment_uncertainty = compute_mc_dropout_uncertainty(
                            screenshot, input_df, labeled_df, n_drugs,
                            sample_info, control_samples,
                            n_passes=10, dropout_p=0.1,
                        )


                    per_sample_new_tids = select_batch(
                        strategy=adaptive_strategy,
                        sample_info=sample_info,
                        dd_embeddings=dd_embeddings,
                        final_treatment_emb=final_treatment_emb,
                        treatment_delta=treatment_delta,
                        treatment_uncertainty=treatment_uncertainty,
                        already_selected=already_selected,
                        step_size=step,
                        random_state=random_state,
                        control_samples=control_samples,
                    )


                    for sid, new_tids in per_sample_new_tids.items():
                        already_selected[sid].update(new_tids)
                    new_row_indices = treatments_to_row_indices(
                        sample_info, per_sample_new_tids,
                        random_state=random_state)
                    all_row_indices = sorted(set(all_row_indices + new_row_indices))


                    labeled_df = input_df.loc[all_row_indices].copy()
                    needs_emb = (round_idx < n_rounds)
                    if needs_emb:
                        pred_result = screenshot.predict(
                            df_input=labeled_df, df_query=input_df,
                            n_drugs=n_drugs, sample_id_col="sample_id",
                            col_viability="float_value",
                            return_embeddings=True, show_progress=False,
                        )
                        current_predictions = pred_result["predictions"]["predicted_viability"].values
                        current_final_emb = pred_result["embeddings"]
                    else:
                        df_pred = screenshot.predict(
                            df_input=labeled_df, df_query=input_df,
                            n_drugs=n_drugs, sample_id_col="sample_id",
                            col_viability="float_value", show_progress=False,
                        )
                        current_predictions = df_pred["predicted_viability"].values


                result_df[f"pred_{col_name}"] = current_predictions
                result_df[f"sel_{col_name}"] = input_df.index.isin(all_row_indices)

                metrics = compute_metrics(
                    raw_df["float_value"].values, current_predictions)
                hit_recall = compute_hit_recall_named(
                    raw_df, current_predictions, control_samples, dose_agg=dose_agg, k_pct=hit_k_pct)
                hr_str = f"{hit_recall:.1f}%" if hit_recall is not None else "N/A"
                print(f"  MAE: {metrics['MAE']:.4f}  Pearson r: {metrics['Correlation']:.4f}  Hit Recall@{int(hit_k_pct*100)}%: {hr_str}")


            out_path = dataset_dir / f"b{budget}_s{random_state}.feather"
            result_df.reset_index(drop=True).to_feather(out_path)
            pred_cols = [c for c in result_df.columns if c.startswith("pred_")]
            print(f"\n  Saved {out_path} ({len(result_df)} rows, "
                  f"{len(pred_cols)} method-configs)")

    print(f"\nAll results saved to {dataset_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
