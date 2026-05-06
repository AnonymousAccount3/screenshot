from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class ControlState:
    """Manages control sample selection for delta computation."""

    method: str = "median"  # "user_selected" | "median"
    control_sample_ids: list[str] = field(default_factory=list)

    def get_config(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "control_sample_ids": list(self.control_sample_ids),
        }

    def set_controls(self, sample_ids: list[str]) -> dict[str, Any]:
        if sample_ids:
            self.method = "user_selected"
            self.control_sample_ids = list(sample_ids)
        else:
            self.method = "median"
            self.control_sample_ids = []
        return self.get_config()

def compute_deltas(
    predictions: list[dict],
    control_state: ControlState,
) -> dict[str, Any]:
    import pandas as pd

    if not predictions:
        return {
            "deltas": [],
            "method": control_state.method,
            "control_sample_ids": list(control_state.control_sample_ids),
        }

    df = pd.DataFrame(predictions)

    if control_state.method == "user_selected" and control_state.control_sample_ids:
        control_mask = df["sample_id"].isin(control_state.control_sample_ids)
        if control_mask.any():
            drug_cols = [c for c in df.columns if c.startswith("drug")]
            dose_cols = [c for c in df.columns if c.startswith("dose")]
            group_cols = drug_cols + dose_cols

            control_baseline = (
                df[control_mask]
                .groupby(group_cols, as_index=False)["predicted_viability"]
                .mean()
                .rename(columns={"predicted_viability": "control_viability"})
            )
            non_control = df[~control_mask].copy()
            result = non_control.merge(control_baseline, on=group_cols, how="left")
        else:
            result = _median_fallback(df)
    else:
        result = _median_fallback(df)

    result["delta"] = result["control_viability"] - result["predicted_viability"]

    return {
        "deltas": result.to_dict(orient="records"),
        "method": control_state.method,
        "control_sample_ids": list(control_state.control_sample_ids),
    }

def _median_fallback(df: "pd.DataFrame") -> "pd.DataFrame":
    """Compute control as median viability across all samples per drug-dose."""
    drug_cols = [c for c in df.columns if c.startswith("drug")]
    dose_cols = [c for c in df.columns if c.startswith("dose")]
    group_cols = drug_cols + dose_cols

    median_baseline = (
        df.groupby(group_cols, as_index=False)["predicted_viability"]
        .median()
        .rename(columns={"predicted_viability": "control_viability"})
    )

    result = df.merge(median_baseline, on=group_cols, how="left")
    return result
