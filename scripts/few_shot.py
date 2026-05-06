"""
Usage:
    python few_shot.py --method screenshot --config config_few_shot.yaml --data-config config_batchie.yaml --input data.feather
"""

import sys
import os
import argparse
import copy
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from experiment import load_data, load_model, save_results, compute_metrics
from selection import select_random
from hit_detection import build_treatment_key, compute_differential_viability
from utils import prepare_features_with_morgan


def compute_hit_recall(raw_df, predictions, control_samples, dose_agg="min",
                       top_pct=0.10, k_pct=0.20):
    result = raw_df.copy()
    result["prediction"] = predictions
    result = build_treatment_key(result)

    ctrl_treats = set()
    for ctrl_name in control_samples:
        ctrl_treats.update(
            result[result["sample_id"] == ctrl_name]["treatment"].unique())
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


def train_predict_xgb(train_df, test_df, fingerprint_cache, config, n_drugs):
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV

    xgb_config = config['xgboost']
    X_train, y_train = prepare_features_with_morgan(train_df, fingerprint_cache, n_drugs)
    X_test, _ = prepare_features_with_morgan(test_df, fingerprint_cache, n_drugs)

    param_grid = xgb_config.get('param_grid', {})
    fixed_params = {
        'objective': xgb_config.get('objective', 'reg:squarederror'),
        'eval_metric': 'mae',
        'seed': xgb_config.get('seed', 42),
        'colsample_bytree': 1.0,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 1.0,
    }

    base_model = xgb.XGBRegressor(
        n_estimators=xgb_config.get('n_estimators', 100),
        **fixed_params
    )

    cv_folds = xgb_config.get('cv_folds', 5)
    if len(X_train) < cv_folds:
        base_model.fit(X_train, y_train)
        return base_model.predict(X_test)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=xgb_config.get('n_jobs', -1),
        verbose=0,
        refit=True
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_.predict(X_test)


def train_predict_tabpfn(train_df, test_df, fingerprint_cache, config, n_drugs):
    import gc
    import torch
    from tabpfn import TabPFNRegressor

    device = config.get('tabpfn', {}).get('device', 'cpu')
    X_train, y_train = prepare_features_with_morgan(train_df, fingerprint_cache, n_drugs)
    X_test, _ = prepare_features_with_morgan(test_df, fingerprint_cache, n_drugs)

    model = TabPFNRegressor(device=device)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions


def train_predict_mlp(base_model, train_df, val_df, test_df, config, device, config_key='mlp'):
    import torch
    from torch.utils.data import DataLoader
    from mlp import DrugResponseDataset, FixedIterationDataLoader, Trainer, split_train_val

    mlp_config = config[config_key]
    model = copy.deepcopy(base_model).to(device)
    n_drugs = config['data']['n_drugs']
    batch_size = mlp_config.get('batch_size', 32)

    train_dataset = DrugResponseDataset(train_df, n_drugs=n_drugs)
    val_dataset = DrugResponseDataset(val_df, n_drugs=n_drugs)

    base_train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    train_loader = FixedIterationDataLoader(
        base_train_loader, mlp_config.get('iterations_per_epoch', 100))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=mlp_config.get('lr', 0.001),
        weight_decay=mlp_config.get('weight_decay', 0.0)
    )

    temp_dir = tempfile.mkdtemp(prefix='mlp_train_')
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            output_dir=temp_dir,
            gradient_clip=mlp_config.get('gradient_clip', 1.0),
            verbose=False,
        )

        num_epochs = mlp_config.get('num_epochs', 200)
        trainer.train(num_epochs=num_epochs, save_every=num_epochs + 1)

        best_path = Path(temp_dir) / 'best_model.pt'
        if best_path.exists():
            trainer.load_checkpoint(str(best_path))

        test_dataset = DrugResponseDataset(test_df, n_drugs=n_drugs)
        test_loader = DataLoader(
            test_dataset, batch_size=128, shuffle=False, num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                _, preds = model(
                    sample_ids=batch['sample_ids'],
                    drug_ids=batch['drug_ids'],
                    doses=batch['doses']
                )
                all_preds.append(preds.cpu().numpy())
        predictions = np.concatenate(all_preds)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return predictions


def predict_screenshot(screenshot, few_shot_df, query_df, config):
    n_drugs = config['data']['n_drugs']
    df_pred = screenshot.predict(
        df_input=few_shot_df,
        df_query=query_df,
        n_drugs=n_drugs,
        sample_id_col="sample_id",
        col_viability="float_value",
        show_progress=True
    )
    return df_pred['predicted_viability'].values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, choices=['xgb', 'tabpfn', 'mlp', 'mlp_finetuned', 'screenshot'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-config', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    method = args.method
    data = load_data([args.config, args.data_config], args.input)
    config = data['config']
    raw_df = data['raw_df']
    input_df = data['input_df']
    drug_library = data['drug_library']
    n_drugs = data['n_drugs']
    fingerprint_cache = data['fingerprint_cache']

    if args.device:
        for key in ['screenshot', 'mlp', 'mlp_finetuned', 'tabpfn']:
            if key in config:
                config[key]['device'] = args.device

    control_samples = config['data']['control_samples']
    dose_agg = config['data'].get('dose_agg', 'min')
    print(f"\n  Control samples: {control_samples}")
    print(f"  Dose aggregation: {dose_agg}")

    unique_samples = input_df['sample_id'].unique()
    sample_id_map = {sid: i for i, sid in enumerate(unique_samples)}
    input_df['sample_idx'] = input_df['sample_id'].map(sample_id_map)

    model_assets = load_model(method, config, drug_library, input_df)

    n_few_shots_list = config['n_few_shots']
    random_states = config['random_states']
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    hit_k_pct = config.get('hit_recall_k_pct', 0.20)

    input_filename = Path(args.input).stem
    save_path = output_dir / f"{method}_{input_filename}.feather"

    all_results = []
    completed = set()
    if save_path.exists():
        existing = pd.read_feather(save_path)
        for (ns, rs), grp in existing.groupby(['n_few_shots', 'random_state']):
            all_results.append(grp)
            completed.add((int(ns), int(rs)))
        print(f"Resuming: {len(completed)} experiments already completed")

    mlp_config_key = 'mlp_finetuned' if method == 'mlp_finetuned' else 'mlp'
    val_ratio = config.get(mlp_config_key, {}).get('val_ratio', 0.2)

    for n_shots in n_few_shots_list:
        for random_state in random_states:
            if (n_shots, random_state) in completed:
                continue

            print(f"\n  n_shots={n_shots}, seed={random_state}")

            few_shot_dict = select_random(input_df, n_shots=n_shots, random_state=random_state)

            if method == 'screenshot':
                if n_shots == 0:
                    samples = input_df['sample_id'].unique()
                    placeholder_rows = []
                    for s in samples:
                        row = {'sample_id': s, 'float_value': 1.0}
                        for i in range(1, n_drugs + 1):
                            row[f'drug{i}'] = 0
                            row[f'dose{i}'] = 0.0
                        row['drug1'] = 3
                        placeholder_rows.append(row)
                    few_shot_df = pd.DataFrame(placeholder_rows)
                    input_df_marked = input_df.copy()
                    input_df_marked['in_few_shot'] = False
                else:
                    all_indices = np.concatenate([np.array(v) for v in few_shot_dict.values()])
                    few_shot_df = input_df.loc[all_indices].copy()
                    input_df_marked = input_df.copy()
                    input_df_marked['in_few_shot'] = input_df.index.isin(all_indices)

                predictions = predict_screenshot(
                    model_assets['screenshot'], few_shot_df, input_df, config
                )

                result_df = raw_df.copy()
                result_df['in_few_shot'] = input_df_marked['in_few_shot'].values
                result_df['prediction'] = predictions
                result_df['n_few_shots'] = n_shots
                result_df['random_state'] = random_state
                result_df['selection_method'] = 'random'
                all_results.append(result_df)

                metrics = compute_metrics(result_df['float_value'].values, result_df['prediction'].values)
                hit_recall = compute_hit_recall(raw_df, result_df['prediction'].values, control_samples, dose_agg=dose_agg, k_pct=hit_k_pct)
                hr_str = f"{hit_recall:.1f}%" if hit_recall is not None else "N/A"
                print(f"  MAE: {metrics['MAE']:.4f}  Pearson r: {metrics['Correlation']:.4f}  Hit Recall@{int(hit_k_pct*100)}%: {hr_str}")

                save_results(all_results, output_dir, save_path.name)
                continue

            # Per-sample loop for xgb, tabpfn, mlp
            sample_predictions = []
            sample_indices_list = []
            sample_in_few_shot = []

            for idx, (sample_id, fs_indices) in enumerate(few_shot_dict.items(), 1):
                print(f"    [{idx}/{len(few_shot_dict)}] {sample_id} ({len(fs_indices)} shots)")

                sample_df = input_df[input_df['sample_id'] == sample_id].copy()
                train_df = sample_df.loc[fs_indices].copy()

                if method == 'xgb':
                    preds = train_predict_xgb(train_df, sample_df, fingerprint_cache, config, n_drugs)
                elif method == 'tabpfn':
                    preds = train_predict_tabpfn(train_df, sample_df, fingerprint_cache, config, n_drugs)
                elif method in ('mlp', 'mlp_finetuned'):
                    from mlp import split_train_val
                    train_split, val_split = split_train_val(train_df, val_ratio=val_ratio, random_state=random_state)
                    preds = train_predict_mlp(
                        model_assets['model'], train_split, val_split, sample_df,
                        config, model_assets['device'], config_key=mlp_config_key
                    )

                sample_predictions.append(preds)
                sample_indices_list.append(sample_df.index.tolist())
                in_fs = np.zeros(len(sample_df), dtype=bool)
                in_fs[sample_df.index.isin(fs_indices)] = True
                sample_in_few_shot.append(in_fs)

            all_idx = []
            all_preds = []
            all_fs = []
            for indices, preds, fs in zip(sample_indices_list, sample_predictions, sample_in_few_shot):
                all_idx.extend(indices)
                all_preds.extend(preds)
                all_fs.extend(fs)

            result_df = raw_df.loc[all_idx].copy()
            result_df['prediction'] = all_preds
            result_df['n_few_shots'] = n_shots
            result_df['random_state'] = random_state
            result_df['selection_method'] = 'random'
            result_df['in_few_shot'] = all_fs
            all_results.append(result_df)

            metrics = compute_metrics(result_df['float_value'].values, result_df['prediction'].values)
            hit_recall = compute_hit_recall(raw_df, result_df['prediction'].values, control_samples, dose_agg=dose_agg, k_pct=hit_k_pct)
            hr_str = f"{hit_recall:.1f}%" if hit_recall is not None else "N/A"
            print(f"  MAE: {metrics['MAE']:.4f}  Pearson r: {metrics['Correlation']:.4f}  Hit Recall@{int(hit_k_pct*100)}%: {hr_str}")

            save_results(all_results, output_dir, save_path.name)

    save_results(all_results, output_dir, save_path.name)
    print("\nDone.")


if __name__ == '__main__':
    main()
