"""Sample selection strategies for few-shot experiments."""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from onebatch import OneBatchPAM
from tqdm import tqdm


def select_random(df: pd.DataFrame, n_shots: int, random_state: int):
    """
    Randomly select n_shots instances per sample_id.

    Args:
        df: DataFrame with sample_id column
        n_shots: number of shots to select per sample
        random_state: random seed

    Returns:
        Dictionary mapping sample_id to list of selected DataFrame indices.
    """
    np.random.seed(random_state)

    few_shot_dict = {}

    for sample_id in df['sample_id'].unique():
        sample_df = df[df['sample_id'] == sample_id]
        n_available = len(sample_df)

        if n_available <= n_shots:
            selected_indices = sample_df.index.tolist()
        else:
            selected_indices = np.random.choice(
                sample_df.index,
                size=n_shots,
                replace=False
            ).tolist()

        few_shot_dict[sample_id] = selected_indices

    return few_shot_dict


def select_kmedoids(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    n_shots: int,
    random_state: int,
    desc: str = "Clustering per sample"
):
    """
    Use K-medoids clustering on embeddings to select n_shots instances per sample_id.

    Args:
        df: DataFrame with sample_id column
        embeddings: numpy array of embeddings (n_instances, hidden_size) aligned with df
        n_shots: number of shots to select per sample
        random_state: random seed
        desc: description for the progress bar

    Returns:
        Dictionary mapping sample_id to list of selected DataFrame indices.
    """
    np.random.seed(random_state)

    assert len(embeddings) == len(df), \
        f"Embeddings length {len(embeddings)} != DataFrame length {len(df)}"

    few_shot_dict = {}

    for sample_id in tqdm(df['sample_id'].unique(), desc=desc):
        sample_df = df[df['sample_id'] == sample_id]
        sample_indices = sample_df.index.tolist()
        n_available = len(sample_df)

        if n_available <= n_shots:
            selected_indices = sample_indices
        else:
            sample_positions = [df.index.get_loc(idx) for idx in sample_indices]
            sample_embeddings = embeddings[sample_positions].astype(np.float32)

            km = OneBatchPAM(
                n_medoids=n_shots,
                random_state=random_state,
            )
            km.fit(sample_embeddings)
            selected_indices = [sample_indices[i] for i in km.medoid_indices_]

        few_shot_dict[sample_id] = selected_indices

    return few_shot_dict


def select_kmedoids_bootstrap(
    df: pd.DataFrame,
    bootstrap_predictions: np.ndarray,
    n_shots: int,
    random_state: int,
    already_labeled_indices: list
):
    """
    Select samples using K-medoids on centered bootstrap predictions with active learning adjustment.

    Filters out already-labeled indices and adjusts the distance matrix to favor
    samples that are far from the existing labeled set.

    Args:
        df: DataFrame with sample_id column
        bootstrap_predictions: numpy array of predictions (n_instances, n_bootstraps) aligned with df
        n_shots: number of shots to select per sample
        random_state: random seed
        already_labeled_indices: list of DataFrame indices already labeled

    Returns:
        Dictionary mapping sample_id to list of selected DataFrame indices.
    """
    np.random.seed(random_state)

    assert len(bootstrap_predictions) == len(df), \
        f"Bootstrap predictions length {len(bootstrap_predictions)} != DataFrame length {len(df)}"

    centered_predictions = bootstrap_predictions - bootstrap_predictions.mean(axis=1, keepdims=True)

    few_shot_dict = {}

    for sample_id in tqdm(df['sample_id'].unique(), desc="Clustering per sample (bootstrap)"):
        sample_df = df[df['sample_id'] == sample_id]
        sample_indices = sample_df.index.tolist()

        sample_already_labeled = [idx for idx in already_labeled_indices if idx in sample_indices]
        available_indices = [idx for idx in sample_indices if idx not in sample_already_labeled]

        n_available = len(available_indices)

        if n_available == 0:
            selected_indices = []
        elif n_available <= n_shots:
            selected_indices = available_indices
        else:
            available_predictions = centered_predictions[available_indices].astype(np.float32)

            # Weight by distance to already-labeled set: favor far-away points
            sample_weight = None
            if len(sample_already_labeled) > 0:
                labeled_predictions = centered_predictions[sample_already_labeled]
                dist_to_labeled = pairwise_distances(
                    available_predictions,
                    labeled_predictions,
                    metric='euclidean'
                )
                sample_weight = dist_to_labeled.min(axis=1).astype(np.float32)

            km = OneBatchPAM(
                n_medoids=n_shots,
                random_state=random_state,
            )
            km.fit(available_predictions, sample_weight=sample_weight)
            selected_indices = [available_indices[i] for i in km.medoid_indices_]

        few_shot_dict[sample_id] = selected_indices

    return few_shot_dict
