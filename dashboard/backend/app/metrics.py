from __future__ import annotations

import numpy as np
import pandas as pd

def compute_auc_mean(viabilities: np.ndarray) -> float:
    if len(viabilities) == 0:
        return 0.0
    return float(np.mean(viabilities))

def compute_ic50(doses: np.ndarray, viabilities: np.ndarray) -> float | None:
    if len(doses) < 2:
        return None
    log_doses = np.log10(np.maximum(doses, 1e-10))
    sort_idx = np.argsort(log_doses)
    sorted_log_doses = log_doses[sort_idx]
    sorted_viabilities = viabilities[sort_idx]
    for i in range(len(sorted_viabilities) - 1):
        v1, v2 = sorted_viabilities[i], sorted_viabilities[i + 1]
        d1, d2 = sorted_log_doses[i], sorted_log_doses[i + 1]
        if (v1 - 0.5) * (v2 - 0.5) < 0:
            # v = v1 + (v2 - v1) * (d - d1) / (d2 - d1)
            if abs(v2 - v1) < 1e-10:
                ic50_log = (d1 + d2) / 2
            else:
                ic50_log = d1 + (0.5 - v1) * (d2 - d1) / (v2 - v1)
            return float(ic50_log)
    return None

def compute_metrics_per_pair(predictions: list[dict]) -> dict[str, dict]:
    df = pd.DataFrame(predictions)

    if len(df) == 0:
        return {}
    metrics = {}

    for (sample_id, drug), group in df.groupby(["sample_id", "drug1"]):
        viabilities = group["predicted_viability"].values
        doses = group["dose1"].values

        auc = compute_auc_mean(viabilities)
        ic50 = compute_ic50(doses, viabilities)
        key = f"{sample_id}|{drug}"
        metrics[key] = {
            "auc": round(float(auc), 6),
            "ic50": round(float(ic50), 6) if ic50 is not None else None,
        }

    return metrics
