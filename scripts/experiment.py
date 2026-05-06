import yaml
import io
import contextlib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

from datascreen import DrugLibrary, extract_drug_names
from screenshot import Preprocessor
from utils import create_morgan_fingerprint_cache


def deep_merge(base, override):
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_data(config_path, input_path: str):
    if isinstance(config_path, (list, tuple)):
        config_paths = config_path
    else:
        config_paths = [config_path]

    config = {}
    for cp in config_paths:
        with open(cp, 'r') as f:
            config = deep_merge(config, yaml.safe_load(f))
        print(f"Loaded configuration from {cp}")
    print(f"Input file: {input_path}")

    data_config = config['data']
    n_drugs = data_config.get('n_drugs', 3)

    drug_library_path = data_config['drug_library']
    if not Path(drug_library_path).exists():
        raise ValueError(f"Drug library not found: {drug_library_path}")
    with contextlib.redirect_stdout(io.StringIO()):
        drug_library = DrugLibrary.from_pretrained(drug_library_path)

    raw_df = pd.read_feather(Path(input_path))
    raw_df = raw_df.reset_index(drop=True)
    if 'float_value' in raw_df.columns:
        raw_df['float_value'] = raw_df['float_value'].clip(lower=0.0, upper=1.0)
    print(f"  {len(raw_df)} rows, {raw_df['sample_id'].nunique()} unique samples")

    max_samples = data_config.get('max_samples')
    if max_samples is not None and raw_df['sample_id'].nunique() > max_samples:
        drug_cols = [c for c in raw_df.columns if c.startswith('drug')]
        combos_per_sample = raw_df.groupby('sample_id')[drug_cols].apply(
            lambda x: len(x.drop_duplicates())
        ).sort_values(ascending=False)
        top_samples = combos_per_sample.head(max_samples).index.tolist()
        raw_df = raw_df[raw_df['sample_id'].isin(top_samples)].reset_index(drop=True)
        print(f"  Filtered to top {max_samples} samples: {len(raw_df)} rows")

    with contextlib.redirect_stdout(io.StringIO()):
        drug_library.update(extract_drug_names(raw_df))

    morgan_config = config.get('morgan', {})
    n_bits = morgan_config.get('n_bits', 512)
    name_cache = create_morgan_fingerprint_cache(
        raw_df, drug_library, n_drugs=n_drugs,
        radius=morgan_config.get('radius', 2),
        n_bits=n_bits,
    )

    preprocessor = Preprocessor(
        drug_library=drug_library,
        n_drugs=n_drugs,
        min_dose_val=data_config["min_dose_val"],
        max_dose_val=data_config["max_dose_val"]
    )
    input_df = preprocessor.transform(raw_df)

    unique_samples = input_df['sample_id'].unique()
    sample_id_map = {sid: i for i, sid in enumerate(unique_samples)}
    input_df['sample_idx'] = input_df['sample_id'].map(sample_id_map)

    fingerprint_cache = {}
    zero_fp = np.zeros(n_bits)
    for drug_name, fp in name_cache.items():
        if not isinstance(drug_name, str):
            continue
        token_id = drug_library.encode(drug_name)
        fingerprint_cache[token_id] = fp
    fingerprint_cache[0] = zero_fp
    fingerprint_cache[None] = zero_fp

    return {
        "config": config,
        "drug_library": drug_library,
        "raw_df": raw_df,
        "input_df": input_df,
        "preprocessor": preprocessor,
        "n_drugs": n_drugs,
        "fingerprint_cache": fingerprint_cache,
    }


def load_model(method, config, drug_library, input_df):
    if method in ('screenshot', 'mlp', 'mlp_finetuned'):
        import torch
        import torch.nn as nn
        from screenshot import ScreenShot
        from mlp import DrugResponseMLP

    if method == 'screenshot':
        sc_config = config['screenshot']
        screenshot = ScreenShot.from_pretrained(
            model_path=sc_config['model_path'],
            device=sc_config.get('device', 'cpu'),
            batch_size=sc_config.get('batch_size', 128)
        )
        return {"screenshot": screenshot}

    elif method == 'mlp':
        mlp_config = config['mlp']
        device = mlp_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        n_samples = input_df['sample_id'].nunique()
        model = DrugResponseMLP(
            n_drugs=drug_library.vocab_size,
            n_samples=n_samples,
            embedding_dim=mlp_config.get('embedding_dim', 64),
            hidden_dim=mlp_config.get('hidden_dim', 128),
            n_layers=mlp_config.get('n_layers', 3),
            dropout=mlp_config.get('dropout', 0.0),
            activation=mlp_config.get('activation', 'relu')
        )
        model = model.to(device)
        return {"model": model, "device": device}

    elif method == 'mlp_finetuned':
        mlp_config = config['mlp_finetuned']
        device = mlp_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = mlp_config.get('checkpoint')
        if not checkpoint_path or not Path(checkpoint_path).exists():
            raise ValueError(f"MLP checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = DrugResponseMLP(
            n_drugs=mlp_config.get('drug_vocab_size', drug_library.vocab_size),
            n_samples=mlp_config.get('sample_vocab_size', 7000),
            embedding_dim=mlp_config.get('embedding_dim', 64),
            hidden_dim=mlp_config.get('hidden_dim', 128),
            n_layers=mlp_config.get('n_layers', 3),
            dropout=mlp_config.get('dropout', 0.0),
            activation=mlp_config.get('activation', 'relu')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        nn.init.xavier_uniform_(model.sample_embedding.weight)
        model = model.to(device)
        return {"model": model, "device": device}

    elif method in ('xgb', 'tabpfn'):
        return {}

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Correlation": pearsonr(y_true, y_pred)[0],
    }


def save_results(results_list, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_results = pd.concat(results_list, axis=0, ignore_index=True)
    output_path = output_dir / filename
    final_results.reset_index(drop=True).to_feather(output_path)
    return final_results
