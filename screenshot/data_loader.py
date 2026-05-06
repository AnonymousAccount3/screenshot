import inspect
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Callable, Union
from functools import partial
from tqdm import tqdm


def set_random_state(random_state):
    # Handle random state
    if random_state is None or random_state is np.random:
        rng = np.random
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        raise ValueError(f"random_state must be None, int, or RandomState, got {type(random_state)}")
    return rng


def random_int_between(min_val: int, max_val: int) -> Callable[[np.random.RandomState], int]:
    """Return a callable that generates random integers between min_val and max_val (inclusive)."""
    return partial(lambda random_state, low, high: random_state.randint(low, high + 1),
                   low=min_val, high=max_val)



def process_df_with_chunks_optimized(df,
                                     sample_id_col,
                                     drug_columns,
                                     dose_columns,
                                     col_perturb_ids='drug_triplet',
                                     col_drug_dose_ids='combo_key',
                                     col_viability='float_value'):
    """
    Create chunks, combo_matrix, and processed dataframe.
    """
    # Sort by sample_id
    df = df.sort_values(sample_id_col).reset_index(drop=True)

    # Create chunks dictionary
    sample_ids = df[sample_id_col].unique()
    chunks = {}

    grouped_indices = df.groupby(sample_id_col, sort=False).indices
    for sample_id in sample_ids:
        indices = grouped_indices[sample_id]
        chunks[int(sample_id)] = (int(indices[0]), int(indices[-1] + 1))

    # Get unique drug-dose combinations
    combo_columns = drug_columns + dose_columns
    combo_df = df[combo_columns].drop_duplicates().reset_index(drop=True)

    nan_row = pd.DataFrame([[0] * len(combo_df.columns)], columns=combo_df.columns)
    combo_df = pd.concat([nan_row, combo_df], ignore_index=True).reset_index(drop=True)

    # Create combo_key using merge
    combo_df[col_drug_dose_ids] = np.arange(1, len(combo_df) + 1)

    df = df.merge(
        combo_df,
        on=combo_columns,
        how='left'
    )

    padding_row = np.zeros((1, len(combo_columns)))
    combo_matrix = np.vstack([padding_row, combo_df[combo_columns].values])

    df[col_perturb_ids] = list(zip(*(df[c] for c in drug_columns)))
    df[col_perturb_ids] = df[col_perturb_ids].astype(str)
    df = df[[sample_id_col, col_perturb_ids, col_drug_dose_ids, col_viability]]
    return df, chunks, combo_matrix


def fast_randomized_factorize(group, rng):
    codes, uniques = pd.factorize(group, sort=False)
    shuffled_labels = np.arange(len(uniques))
    rng.shuffle(shuffled_labels)
    return pd.Series(shuffled_labels[codes], index=group.index)


def sample_ordered_assignment(df,
                              sample_chunks_info,
                              col_perturb_ids='drug_triplet',
                              col_drug_dose_ids='combo_key',
                              col_viability='float_value'):
    i_list = []
    j_list = []
    k_list = []
    combo_keys_list = []
    viability_list = []

    sample_ids = np.array(list(sample_chunks_info.keys()))

    for i, sample_id in enumerate(sample_ids):
        start, end = sample_chunks_info[sample_id]
        df_sample = df.iloc[start:end]

        j_idx = pd.factorize(df_sample[col_perturb_ids], sort=False)[0]
        k_idx = df_sample.groupby(j_idx, sort=False).cumcount().values

        combo_keys = df_sample[col_drug_dose_ids].values
        viability = df_sample[col_viability].values

        i_list.append(np.full((len(j_idx),), i))
        j_list.append(j_idx)
        k_list.append(k_idx)
        combo_keys_list.append(combo_keys)
        viability_list.append(viability)

    i = np.concatenate(i_list)
    j = np.concatenate(j_list)
    k = np.concatenate(k_list)
    combo_keys = np.concatenate(combo_keys_list)
    viability = np.concatenate(viability_list)

    return (i, j, k), combo_keys, viability, sample_ids


def sample_random_assignment(df,
                             n_samples,
                             n_combos,
                             n_obs,
                             n_drugs,
                             sample_chunks_info,
                             col_perturb_ids='drug_triplet',
                             col_drug_dose_ids='combo_key',
                             col_viability='float_value',
                             random_state=None):
    rng = set_random_state(random_state)

    i_list = []
    j_list = []
    k_list = []
    combo_keys_list = []
    viability_list = []

    all_sample_ids = np.array(list(sample_chunks_info.keys()))
    sample_ids = rng.choice(all_sample_ids,
                            size=min(n_samples, len(all_sample_ids)),
                            replace=False)

    # Process each sample
    for i, sample_id in enumerate(sample_ids):
        start, end = sample_chunks_info[sample_id]
        df_sample = df.iloc[start:end]

        random_values = pd.Series(rng.rand(len(df_sample)), index=df_sample.index)
        j_idx = fast_randomized_factorize(df_sample[col_perturb_ids], rng)
        k_idx = random_values.groupby(j_idx).rank(method='first').astype('int') - 1

        j_idx = j_idx.values
        k_idx = k_idx.values
        mask = (j_idx < n_combos) & (k_idx < n_obs)

        combo_keys = df_sample[col_drug_dose_ids].values[mask]
        viability = df_sample[col_viability].values[mask]

        j_idx = j_idx[mask]
        k_idx = k_idx[mask]

        i_list.append(np.full((len(j_idx),), i))
        j_list.append(j_idx)
        k_list.append(k_idx)
        combo_keys_list.append(combo_keys)
        viability_list.append(viability)

    i = np.concatenate(i_list)
    j = np.concatenate(j_list)
    k = np.concatenate(k_list)
    combo_keys = np.concatenate(combo_keys_list)
    viability = np.concatenate(viability_list)

    return (i, j, k), combo_keys, viability, sample_ids


def batch_test_query(df,
                   input_sample_ids,
                   drug_columns,
                   dose_columns,
                   sample_id_col,
                   col_viability):
    """
    Create test query batch.
    """
    n_samples = len(input_sample_ids)
    n_drugs = len(drug_columns)

    i_list = []
    j_list = []
    ind_list = []

    query_size = 0
    grouped_indices = df.groupby(sample_id_col, sort=False).indices
    for i, sample_id in enumerate(input_sample_ids):
        if sample_id in grouped_indices:
            indices = grouped_indices[sample_id]

            if len(indices) > query_size:
                query_size = len(indices)

            i_list.append(np.full((len(indices),), i))
            j_list.append(np.arange(len(indices)))
            ind_list.append(indices)

    indices = np.concatenate(ind_list)
    i = np.concatenate(i_list)
    j = np.concatenate(j_list)

    df = df.iloc[indices]

    drug_ids = np.zeros((n_samples, query_size, n_drugs), dtype=np.int32)
    dose_levels = np.zeros((n_samples, query_size, n_drugs), dtype=np.float32)
    origin_index = np.full((n_samples, query_size), np.nan)

    drug_ids[i, j, :] = df[drug_columns].values
    dose_levels[i, j, :] = df[dose_columns].values
    origin_index[i, j] = indices

    if col_viability in df.columns:
        label_viability = np.full((n_samples, query_size), np.nan)
        label_viability[i, j] = df[col_viability].values
        label_viability = torch.from_numpy(label_viability).float().unsqueeze(0)
    else:
        label_viability = torch.full((1, n_samples, query_size), np.nan, dtype=torch.float32)

    # Create perturbation mask (which drug positions are non-zero)
    perturb_mask = (drug_ids != 0).astype(np.int32)

    # Reshape to (bs, query_size, n_drugs)
    output_dict = {
        "perturb_drug_ids": torch.from_numpy(drug_ids).long().unsqueeze(0),
        "perturb_mask": torch.from_numpy(perturb_mask).long().unsqueeze(0),
        "label_viability": label_viability,
        "label_dose": torch.from_numpy(dose_levels).float().unsqueeze(0),
        "sample_ids": torch.from_numpy(np.array(input_sample_ids)).long().unsqueeze(0),
        "origin_index": torch.from_numpy(origin_index).float().unsqueeze(0)
    }
    return output_dict


def batch_test_input(df,
                     n_drugs,
                     sample_chunks_info,
                     global_drug_dose_matrix,
                     col_viability="float_value",
                     col_perturb_ids='drug_triplet',
                     col_drug_dose_ids='combo_key'):

    return batch_train_input(df,
                    n_samples=None,
                    n_combos=None,
                    n_obs=None,
                    n_drugs=n_drugs,
                    sampling_func=sample_ordered_assignment,
                    sample_chunks_info=sample_chunks_info,
                    global_drug_dose_matrix=global_drug_dose_matrix,
                    col_perturb_ids=col_perturb_ids,
                    col_drug_dose_ids=col_drug_dose_ids,
                    col_viability=col_viability,
                    adaptive_size=True,
                    random_state=None,
                    sampling_func_params=dict())


def batch_train_input(df,
                    n_samples,
                    n_combos,
                    n_obs,
                    n_drugs,
                    sampling_func,
                    sample_chunks_info,
                    global_drug_dose_matrix,
                    col_perturb_ids='drug_triplet',
                    col_drug_dose_ids='combo_key',
                    col_viability='float_value',
                    adaptive_size=True,
                    random_state=None,
                    sampling_func_params=dict()):
    """
    Create batch with standard factorize + rank assignment.
    """
    func_params = dict(
        n_samples=n_samples,
        n_combos=n_combos,
        n_obs=n_obs,
        n_drugs=n_drugs,
        sample_chunks_info=sample_chunks_info,
        global_drug_dose_matrix=global_drug_dose_matrix,
        col_perturb_ids=col_perturb_ids,
        col_drug_dose_ids=col_drug_dose_ids,
        col_viability=col_viability,
        random_state=random_state,
        **sampling_func_params
    )
    valid_params = inspect.signature(sampling_func).parameters.keys()
    func_params = {k: v for k, v in func_params.items() if k in valid_params}

    (i, j, k), combo_keys, viability, sample_ids = sampling_func(
        df, **func_params
    )
    if adaptive_size:
        n_samples = int(max(i)) + 1 if len(i) > 0 else 0
        n_combos = int(max(j)) + 1 if len(j) > 0 else 0
        n_obs = int(max(k)) + 1 if len(k) > 0 else 0

    # Build matrices with +1 for CLS tokens
    sample_perturb_ids = np.zeros((n_samples, n_combos + 1, n_obs + 1), dtype=np.int32)
    input_viability = np.ones((n_samples, n_combos + 1, n_obs + 1), dtype=np.float32)

    # Directly fill with +1 offset (leaves position 0 for CLS)
    sample_perturb_ids[i, j + 1, k + 1] = combo_keys
    input_viability[i, j + 1, k + 1] = viability

    # Get unique combos and remap efficiently
    unique_batch_combos = np.unique(combo_keys)  # Already sorted

    # Fast remapping using searchsorted
    batch_ids = np.arange(len(unique_batch_combos), dtype=np.int32) + 3

    # Remap all combo_keys in sample_perturb_ids
    # Create a lookup array (max_combo_id + 1 size for direct indexing)
    max_combo_id = unique_batch_combos[-1]
    lookup = np.zeros(max_combo_id + 1, dtype=np.int32)
    lookup[unique_batch_combos] = batch_ids

    # Apply remapping (only to non-zero entries)
    mask = (sample_perturb_ids > 0)
    sample_perturb_ids[mask] = lookup[sample_perturb_ids[mask]]

    # Set CLS tokens at position 0
    sample_perturb_ids[:, :, 0] = 1  # CLS for observations
    sample_perturb_ids[:, 0, :] = 0  # CLS perturbation (padding)
    sample_perturb_ids[:, 0, 0] = 2  # First position is CLS

    # Build vocab
    vocab_size = len(unique_batch_combos) + 3
    drug_dose_ids = np.zeros((vocab_size, n_drugs * 2), dtype=np.float32)
    drug_dose_ids[1, 0] = 1  # CLS obs
    drug_dose_ids[2, 0] = 2  # CLS perturb
    drug_dose_ids[3:, :] = global_drug_dose_matrix[unique_batch_combos]

    # Masks
    mask_obs = (sample_perturb_ids != 0).astype(np.int32)
    mask_obs[:, :, 0] = 1  # CLS always attended
    mask_perturbs = (mask_obs.sum(axis=-1) > 1).astype(np.int32)
    mask_perturbs[:, 0] = 1  # CLS perturbation always attended
    mask_sample = (mask_perturbs.sum(axis=-1) > 1).astype(np.int32)

    # Split drug/dose (first n_drugs columns are drug IDs, last n_drugs are doses)
    drug_ids = drug_dose_ids[:, :n_drugs].astype(np.int32)
    dose_levels = drug_dose_ids[:, n_drugs:].astype(np.float32)

    mask_drugs = (drug_ids != 0).astype(np.int32)
    mask_drugs[0, 0] = 1

    output_dict = {
        "sample_perturb_ids": torch.from_numpy(sample_perturb_ids).long().unsqueeze(0),
        "drug_ids": torch.from_numpy(drug_ids).long(),
        "dose_levels": torch.from_numpy(dose_levels).float(),
        "viability_levels": torch.from_numpy(input_viability).float().unsqueeze(0),
        "attention_mask_drug": torch.from_numpy(mask_drugs).to(torch.long),
        "attention_mask_obs": torch.from_numpy(mask_obs).long().unsqueeze(0),
        "attention_mask_perturbs": torch.from_numpy(mask_perturbs).long().unsqueeze(0),
        "attention_mask_sample": torch.from_numpy(mask_sample).long().unsqueeze(0),
        "sample_ids": torch.from_numpy(sample_ids).long().unsqueeze(0),
    }

    return output_dict


def batch_train_query(df,
                   sample_ids,
                   query_size,
                   n_drugs,
                   sample_chunks_info,
                   global_drug_dose_matrix,
                   col_drug_dose_ids='combo_key',
                   col_viability='float_value',
                   random_state=None):
    """
    Sample test data for evaluation.
    """
    rng = set_random_state(random_state)

    n_samples = len(sample_ids)

    # Sample indices for each sample
    ind_list = []

    for s in sample_ids:
        if s not in sample_chunks_info:
            raise ValueError(f"Sample ID {s} not found in chunks info")

        start, end = sample_chunks_info[s]

        sampled_indices = rng.choice(
            np.arange(start, end),
            size=query_size,
            replace=True
        )

        ind_list.extend(sampled_indices)

    # Get combo keys and viability values
    ind_array = np.array(ind_list)
    obs_combo_keys = df[col_drug_dose_ids].iloc[ind_array].values
    obs_viability = df[col_viability].iloc[ind_array].values

    # Query drug and dose information from global matrix
    vals = global_drug_dose_matrix[obs_combo_keys]
    drug_ids = vals[:, :n_drugs].astype(np.int32)
    dose_levels = vals[:, n_drugs:].astype(np.float32)

    # Create perturbation mask (which drug positions are non-zero)
    perturb_mask = (drug_ids != 0).astype(np.int32)

    # Reshape to (bs, query_size, n_drugs)
    output_dict = {
        "perturb_drug_ids": torch.from_numpy(drug_ids).reshape(1, n_samples, query_size, n_drugs).long(),
        "perturb_mask": torch.from_numpy(perturb_mask).reshape(1, n_samples, query_size, n_drugs).long(),
        "label_viability": torch.from_numpy(obs_viability).reshape(1, n_samples, query_size).float(),
        "label_dose": torch.from_numpy(dose_levels).reshape(1, n_samples, query_size, n_drugs).float(),
        "sample_ids": torch.from_numpy(np.array(sample_ids)).long().unsqueeze(0),
        "origin_index": torch.from_numpy(ind_array).reshape(1, n_samples, query_size)
    }

    return output_dict


class TestDataset(Dataset):
    """
    Test dataset for drug screening model.
    Takes preprocessed dataframes (drugs already tokenized, doses normalized).
    """

    def __init__(self,
                 input_df: pd.DataFrame,
                 query_df: pd.DataFrame,
                 n_drugs: int = 3,
                 sample_id_col: str = "sample_id",
                 col_viability: str = 'float_value',
                 batch_size: int = 128):
        """
        Args:
            input_df: Preprocessed input dataframe (tokenized drugs, normalized doses)
            query_df: Preprocessed query dataframe (tokenized drugs, normalized doses)
            n_drugs: Number of drugs per combination
            sample_id_col: Column name for sample IDs
            col_viability: Column name for viability values
            batch_size: Batch size for queries
        """
        self.n_drugs = n_drugs
        self.sample_id_col = sample_id_col
        self.col_viability = col_viability
        self.batch_size = batch_size

        self.col_perturb_ids = 'perturb_id'
        self.col_drug_dose_ids = 'drug_dose_id'

        self.df_input = input_df.copy()
        self.df_query = query_df.copy()

        all_ids = pd.concat([self.df_input[sample_id_col], self.df_query[sample_id_col]]).unique()
        id_map = {sid: i for i, sid in enumerate(all_ids, start=1)}
        self.df_input[sample_id_col] = self.df_input[sample_id_col].map(id_map)
        self.df_query[sample_id_col] = self.df_query[sample_id_col].map(id_map)

        self.drug_columns = [f"drug{i+1}" for i in range(n_drugs)]
        self.dose_columns = [f"dose{i+1}" for i in range(n_drugs)]

        # Process input data
        (self.df_input_, self.chunks_input_,
         self.global_drug_dose_matrix_input_) = process_df_with_chunks_optimized(
            self.df_input,
            sample_id_col=self.sample_id_col,
            drug_columns=self.drug_columns,
            dose_columns=self.dose_columns,
            col_perturb_ids=self.col_perturb_ids,
            col_drug_dose_ids=self.col_drug_dose_ids,
            col_viability=self.col_viability
        )

        # Select relevant columns from query
        columns = [self.sample_id_col] + list(self.drug_columns) + list(self.dose_columns)
        if self.col_viability in self.df_query.columns:
            columns.append(self.col_viability)
        self.df_query_ = self.df_query[columns]

        full_batch_input = batch_test_input(
            df=self.df_input_,
            n_drugs=self.n_drugs,
            sample_chunks_info=self.chunks_input_,
            global_drug_dose_matrix=self.global_drug_dose_matrix_input_,
            col_perturb_ids=self.col_perturb_ids,
            col_drug_dose_ids=self.col_drug_dose_ids,
            col_viability=self.col_viability,
        )

        self.sample_ids = full_batch_input["sample_ids"].squeeze(0).detach().numpy()

        # Store global drug_ids, dose_levels, attention_mask_drug (unchanged)
        self.drug_ids = full_batch_input["drug_ids"]
        self.dose_levels = full_batch_input["dose_levels"]
        self.attention_mask_drug = full_batch_input["attention_mask_drug"]

        # Split and trim batch_input by sample
        self.batch_inputs = []
        n_samples = full_batch_input["sample_perturb_ids"].shape[1]

        for sample_idx in range(n_samples):
            sample_batch = {}

            # Get attention masks for this sample
            attn_obs = full_batch_input["attention_mask_obs"][:, sample_idx, :, :]
            attn_perturbs = full_batch_input["attention_mask_perturbs"][:, sample_idx, :]

            # Find max valid n_perturbs
            nonzero_perturbs = torch.nonzero(attn_perturbs[0], as_tuple=False)
            max_perturbs = nonzero_perturbs.max().item() + 1 if len(nonzero_perturbs) > 0 else 1

            # Find max valid n_obs
            max_obs = 0
            for perturb_idx in range(attn_obs.shape[1]):
                nonzero_obs = torch.nonzero(attn_obs[0, perturb_idx], as_tuple=False)
                if len(nonzero_obs) > 0:
                    max_obs = max(max_obs, nonzero_obs.max().item() + 1)

            # Trim tensors for this sample
            sample_batch["sample_perturb_ids"] = full_batch_input["sample_perturb_ids"][:, sample_idx:sample_idx+1, :max_perturbs, :max_obs]
            sample_batch["viability_levels"] = full_batch_input["viability_levels"][:, sample_idx:sample_idx+1, :max_perturbs, :max_obs]
            sample_batch["attention_mask_obs"] = full_batch_input["attention_mask_obs"][:, sample_idx:sample_idx+1, :max_perturbs, :max_obs]
            sample_batch["attention_mask_perturbs"] = full_batch_input["attention_mask_perturbs"][:, sample_idx:sample_idx+1, :max_perturbs]
            sample_batch["attention_mask_sample"] = full_batch_input["attention_mask_sample"][:, sample_idx:sample_idx+1]
            sample_batch["sample_ids"] = full_batch_input["sample_ids"][:, sample_idx:sample_idx+1]

            self.batch_inputs.append(sample_batch)

        # Split batch_query by sample and trim NaNs
        full_batch_query = batch_test_query(
            df=self.df_query_,
            input_sample_ids=self.sample_ids,
            drug_columns=self.drug_columns,
            dose_columns=self.dose_columns,
            sample_id_col=self.sample_id_col,
            col_viability=self.col_viability,
        )

        self.batch_queries = []
        self.query_lengths = []
        self.batches_per_sample = []
        self.cumulative_batches = [0]

        n_samples = full_batch_query["origin_index"].shape[1]

        for sample_idx in range(n_samples):
            sample_batch = {}

            # Get origin_index for this sample
            origin_idx = full_batch_query["origin_index"][:, sample_idx, :]

            # Find where NaNs start
            is_valid = ~torch.isnan(origin_idx[0])
            valid_length = is_valid.sum().item()

            # Trim all keys except sample_ids
            for k in full_batch_query:
                if k != "sample_ids":
                    sample_batch[k] = full_batch_query[k][:, sample_idx:sample_idx+1, :valid_length]

            self.batch_queries.append(sample_batch)
            self.query_lengths.append(valid_length)

            # Calculate number of batches for this sample
            n_batches_this_sample = (valid_length + self.batch_size - 1) // self.batch_size
            self.batches_per_sample.append(n_batches_this_sample)
            self.cumulative_batches.append(self.cumulative_batches[-1] + n_batches_this_sample)

    def __len__(self) -> int:
        return self.cumulative_batches[-1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        batch = {}
        bs = self.batch_size

        # Find which sample this batch belongs to
        sample_idx = None
        for i in range(len(self.cumulative_batches) - 1):
            if self.cumulative_batches[i] <= idx < self.cumulative_batches[i + 1]:
                sample_idx = i
                break

        if sample_idx is None:
            raise IndexError(f"Batch index {idx} out of range")

        # Find which batch within this sample
        local_batch_idx = idx - self.cumulative_batches[sample_idx]

        # Calculate slice within this sample
        local_start = local_batch_idx * bs
        local_end = min(local_start + bs, self.query_lengths[sample_idx])

        # Get batch_input for this sample (already trimmed per sample)
        for k in self.batch_inputs[sample_idx]:
            batch[k] = self.batch_inputs[sample_idx][k]

        # Add global drug/dose info (unchanged)
        batch["drug_ids"] = self.drug_ids
        batch["dose_levels"] = self.dose_levels
        batch["attention_mask_drug"] = self.attention_mask_drug

        # Get batch_query for this sample (already trimmed, now slice for batch)
        for k in self.batch_queries[sample_idx]:
            batch[k] = self.batch_queries[sample_idx][k][:, :, local_start:local_end]

        return batch


def mask_missing_drugs(batch, vocab_size, n_missing_tokens, alpha_max, rng):
    """
    Randomly replace a fraction of drug IDs with missing drug token IDs.

    During training, this teaches the model to handle unseen drugs by
    replacing alpha fraction of unique drugs in the batch with trainable
    missing drug tokens (IDs vocab_size+1 through vocab_size+n_missing_tokens).

    Args:
        batch: dict with 'drug_ids' (mini_vocab, n_drugs) and
               'perturb_drug_ids' (1, n_samples, query_size, n_drugs)
        vocab_size: base vocabulary size (without missing tokens)
        n_missing_tokens: number of reserved missing token IDs
        alpha_max: max fraction of unique drugs to mask
        rng: numpy RandomState
    """
    if n_missing_tokens <= 0 or alpha_max <= 0:
        return batch

    alpha = rng.uniform(0, alpha_max)

    # Collect unique drug IDs from both input and query (exclude 0=pad, 1,2=CLS)
    drug_ids = batch["drug_ids"].numpy()
    perturb_drug_ids = batch["perturb_drug_ids"].numpy()

    all_ids = np.concatenate([drug_ids.ravel(), perturb_drug_ids.ravel()])
    unique_ids = np.unique(all_ids)
    unique_ids = unique_ids[unique_ids > 2]  # skip padding and CLS tokens

    n_to_mask = max(1, int(len(unique_ids) * alpha))
    if n_to_mask == 0 or len(unique_ids) == 0:
        return batch

    masked_ids = rng.choice(unique_ids, size=min(n_to_mask, len(unique_ids)), replace=False)

    # Assign each masked drug a different missing token ID
    missing_token_ids = vocab_size + np.arange(len(masked_ids)) % n_missing_tokens

    # Build replacement map and apply
    for orig, repl in zip(masked_ids, missing_token_ids):
        drug_ids[drug_ids == orig] = repl
        perturb_drug_ids[perturb_drug_ids == orig] = repl

    batch["drug_ids"] = torch.from_numpy(drug_ids).long()
    batch["perturb_drug_ids"] = torch.from_numpy(perturb_drug_ids).long()

    return batch


class TrainingDataset(Dataset):
    """
    PyTorch Dataset for loading and batching drug combination data.
    Takes cleaned dataframes and drug_library.
    """
    def __init__(self,
                 dataframes: Union[List[pd.DataFrame], Dict[str, pd.DataFrame]],
                 n_samples: Union[int, Callable[[np.random.RandomState], int]],
                 n_combos: Union[int, Callable[[np.random.RandomState], int]],
                 n_obs: Union[int, Callable[[np.random.RandomState], int]],
                 n_drugs: int,
                 query_size: int,
                 col_viability: str = 'float_value',
                 sample_id_col: str = 'sample_id',
                 adaptive_size: bool = True,
                 max_batch_size: Optional[int] = 32000,
                 n_missing_tokens: int = 0,
                 missing_drug_alpha_max: float = 0.2,
                 vocab_size: Optional[int] = None,
                 random_state: Optional[int] = None):
        """
        Args:
            dataframes:
                Either
                - List[pd.DataFrame]: preprocessed dataframes (tokenized drugs,
                  normalized doses), processed in order
                - Dict[str, pd.DataFrame]: mapping from name → preprocessed dataframe.
                  Keys are used for progress display and bookkeeping.
            n_samples: Number of samples per batch (int or callable)
            n_combos: Max perturbations per sample (int or callable)
            n_obs: Max observations per perturbation (int or callable)
            n_drugs: Number of drugs per combination
            query_size: Size of query set
            col_viability: Column name for viability values
            sample_id_col: Column name for sample IDs
            adaptive_size: Whether to use adaptive sizing
            max_batch_size: Maximum allowed batch size
            random_state: Random seed for reproducibility
        """
        super().__init__()

        self.n_samples = n_samples
        self.n_combos = n_combos
        self.n_obs = n_obs
        self.n_drugs = n_drugs
        self.query_size = query_size
        self.col_viability = col_viability
        self.sample_id_col = sample_id_col
        self.adaptive_size = adaptive_size
        self.max_batch_size = max_batch_size
        self.n_missing_tokens = n_missing_tokens
        self.missing_drug_alpha_max = missing_drug_alpha_max
        self.vocab_size = vocab_size
        self.random_state = set_random_state(random_state)

        self.col_perturb_ids = 'drug_triplet'
        self.col_drug_dose_ids = 'combo_key'

        # Storage for processed data
        self.dataframes = []
        self.combo_matrices = []
        self.chunks = []
        self.dataframe_names = []

        # Process all dataframes
        print(f"Processing {len(dataframes)} dataframes...")

        if isinstance(dataframes, dict):
            iterator = dataframes.items()
            total = len(dataframes)
        else:
            iterator = enumerate(dataframes)
            total = len(dataframes)

        with tqdm(iterator, total=total) as pbar:
            for i, df in pbar:
                if isinstance(dataframes, dict):
                    desc = f"Processing {i}"
                else:
                    desc = f"Processing dataframe {i + 1}/{total}"

                pbar.set_description(desc)

                # Factorize sample IDs
                df = df.copy()
                df[sample_id_col] = pd.factorize(df[sample_id_col])[0]

                # Create chunks and combo matrix
                drug_columns = [f"drug{i+1}" for i in range(n_drugs)]
                dose_columns = [f"dose{i+1}" for i in range(n_drugs)]

                df_final, chunks, combo_matrix = process_df_with_chunks_optimized(
                    df,
                    sample_id_col=sample_id_col,
                    drug_columns=drug_columns,
                    dose_columns=dose_columns,
                    col_perturb_ids=self.col_perturb_ids,
                    col_drug_dose_ids=self.col_drug_dose_ids,
                    col_viability=col_viability
                )

                self.dataframes.append(df_final)
                self.combo_matrices.append(combo_matrix)
                self.chunks.append(chunks)
                self.dataframe_names.append(f"df_{i}")

        print(f"\nDataset initialized with {len(self.dataframes)} dataframes")
        self._print_stats()

    def _get_value(self, param: Union[int, Callable[[np.random.RandomState], int]]) -> int:
        """Helper to get value from int or callable."""
        if callable(param):
            return param(random_state=self.random_state)
        return param

    def _adjust_for_max_batch_size(self, n_samples: int, n_combos: int, query_size: int) -> tuple:
        """Adjust n_combos and query_size to respect max_batch_size constraint."""
        if self.max_batch_size is None:
            return n_combos, query_size

        adjusted_n_combos = n_combos
        adjusted_query_size = query_size

        # Check and adjust n_combos
        if n_samples * n_combos > self.max_batch_size:
            adjusted_n_combos = self.max_batch_size // n_samples
            adjusted_n_combos = max(1, adjusted_n_combos)

        # Check and adjust query_size
        if n_samples * query_size > self.max_batch_size:
            adjusted_query_size = self.max_batch_size // n_samples
            adjusted_query_size = max(1, adjusted_query_size)

        return adjusted_n_combos, adjusted_query_size

    def _print_stats(self):
        """Print dataset statistics."""
        total_rows = sum(len(df) for df in self.dataframes)
        total_samples = sum(len(chunks) for chunks in self.chunks)
        total_combos = sum(len(cm) for cm in self.combo_matrices)

        n_samples_display = self.n_samples if isinstance(self.n_samples, int) else "callable"
        n_combos_display = self.n_combos if isinstance(self.n_combos, int) else "callable"
        n_obs_display = self.n_obs if isinstance(self.n_obs, int) else "callable"

        print(f"  Total rows: {total_rows:,}")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total combos: {total_combos:,}")
        print(f"  Batch config: {n_samples_display} samples × {n_combos_display} combos × {n_obs_display} obs")
        if self.max_batch_size is not None:
            print(f"  Max batch size: {self.max_batch_size:,}")

    def __len__(self) -> int:
        """Returns the number of dataframes."""
        return len(self.dataframes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a batch from the specified dataframe."""
        if idx >= len(self.dataframes):
            raise IndexError(f"Index {idx} out of range for {len(self.dataframes)} dataframes")

        # Get data for this index
        df = self.dataframes[idx]
        combo_matrix = self.combo_matrices[idx]
        chunks = self.chunks[idx]

        # Get actual values (resolve callables if needed)
        n_samples = self._get_value(self.n_samples)
        n_combos = self._get_value(self.n_combos)
        n_obs = self._get_value(self.n_obs)

        # Adjust n_combos and query_size to respect max_batch_size
        n_combos, query_size = self._adjust_for_max_batch_size(n_samples,
                                                               n_combos,
                                                               self.query_size)

        batch = batch_train_input(
            df=df,
            n_samples=n_samples,
            n_combos=n_combos,
            n_obs=n_obs,
            n_drugs=self.n_drugs,
            sampling_func=sample_random_assignment,
            sample_chunks_info=chunks,
            global_drug_dose_matrix=combo_matrix,
            col_perturb_ids=self.col_perturb_ids,
            col_drug_dose_ids=self.col_drug_dose_ids,
            col_viability=self.col_viability,
            random_state=self.random_state,
            adaptive_size=self.adaptive_size,
            sampling_func_params={}
        )

        sample_ids = batch["sample_ids"].squeeze(0).detach().numpy()

        batch_query = batch_train_query(
            df=df,
            sample_ids=sample_ids,
            query_size=query_size,
            n_drugs=self.n_drugs,
            sample_chunks_info=chunks,
            global_drug_dose_matrix=combo_matrix,
            col_drug_dose_ids=self.col_drug_dose_ids,
            col_viability=self.col_viability,
            random_state=self.random_state,
        )

        batch.update(batch_query)

        # Apply missing drug masking during training
        if self.n_missing_tokens > 0 and self.vocab_size is not None:
            batch = mask_missing_drugs(
                batch,
                vocab_size=self.vocab_size,
                n_missing_tokens=self.n_missing_tokens,
                alpha_max=self.missing_drug_alpha_max,
                rng=self.random_state,
            )

        return batch


class MultiEpochTrainingDataset(Dataset):
    """
    Extended version that allows multiple batches per dataframe per epoch.
    """

    def __init__(self,
                 dataframes: List[pd.DataFrame],
                 n_samples: int,
                 n_combos: int,
                 n_obs: int,
                 n_drugs: int,
                 query_size: int,
                 batches_per_source: int = 10,
                 col_viability: str = 'float_value',
                 sample_id_col: str = 'sample_id',
                 adaptive_size: bool = True,
                 max_batch_size: Optional[int] = 32000,
                 n_missing_tokens: int = 0,
                 missing_drug_alpha_max: float = 0.2,
                 vocab_size: Optional[int] = None,
                 random_state = None):
        """
        Args:
            batches_per_source: Number of batches to generate per dataframe per epoch
            (other args same as TrainingDataset)
        """
        self.random_state = set_random_state(random_state)

        self.base_dataset = TrainingDataset(
            dataframes=dataframes,
            n_samples=n_samples,
            n_combos=n_combos,
            n_obs=n_obs,
            n_drugs=n_drugs,
            query_size=query_size,
            col_viability=col_viability,
            sample_id_col=sample_id_col,
            adaptive_size=adaptive_size,
            max_batch_size=max_batch_size,
            n_missing_tokens=n_missing_tokens,
            missing_drug_alpha_max=missing_drug_alpha_max,
            vocab_size=vocab_size,
            random_state=self.random_state,
        )
        self.batches_per_source = batches_per_source
        self.total_batches = len(self.base_dataset) * batches_per_source

        print(f"MultiEpochDataset: {self.total_batches} batches per epoch "
              f"({batches_per_source} per source)")

    def __len__(self) -> int:
        return self.total_batches

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item."""
        source_idx = self.random_state.choice(len(self.base_dataset))
        return self.base_dataset[source_idx]