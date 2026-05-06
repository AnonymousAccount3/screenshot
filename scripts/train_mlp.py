import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from typing import List

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

from mlp import DrugResponseMLP, Trainer, DrugResponseDataset, load_config
from utils import get_dataset_paths
from screenshot import Preprocessor
from datascreen import DrugLibrary


def load_and_preprocess_data(
    dataset_paths: List[str],
    drug_library,
    n_drugs: int = 3,
    val_split: float = 0.1,
    random_state: int = 42,
    min_dose_val: float = -6.0,
    max_dose_val: float = 4.0
):
    """
    Load .feather files from paths, preprocess them, and merge.
    Creates unique sample_ids by prefixing with filename.
    """
    print(f"\nLoading {len(dataset_paths)} data files...")

    dfs = []
    for file_path_str in dataset_paths:
        file_path = Path(file_path_str)
        print(f"  Loading and preprocessing {file_path.name}...")
        df = pd.read_feather(file_path)

        # Preprocess the dataframe
        preprocessor = Preprocessor(drug_library, n_drugs=n_drugs, min_dose_val=min_dose_val, max_dose_val=max_dose_val)
        df = preprocessor.transform(df)

        # Create unique sample_id by prefixing with filename
        filename = file_path.stem
        df['sample_id'] = filename + '_' + df['sample_id'].astype(str)

        dfs.append(df)
        print(f"    {len(df)} rows, {df['sample_id'].nunique()} unique samples")

    # Concatenate all dataframes
    print("\nMerging all dataframes...")
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"Total: {len(merged_df)} rows, {merged_df['sample_id'].nunique()} unique samples")

    # Encode sample_ids as integers
    print("\nEncoding sample_ids...")
    unique_samples = merged_df['sample_id'].unique()
    sample_to_id = {sample: idx + 1 for idx, sample in enumerate(unique_samples)}
    merged_df['sample_id'] = merged_df['sample_id'].map(sample_to_id)
    merged_df['sample_idx'] = merged_df['sample_id']

    # Split into train and validation based on (sample_id, drug1) pairs
    print(f"\nSplitting data (val_split={val_split})...")
    np.random.seed(random_state)

    # Create (sample_id, drug1) pairs
    merged_df['sample_drug1_pair'] = merged_df['sample_id'].astype(str) + '_' + merged_df['drug1'].astype(str)
    unique_pairs = merged_df['sample_drug1_pair'].unique()

    print(f"Total unique (sample_id, drug1) pairs: {len(unique_pairs)}")

    # Shuffle and split pairs
    np.random.shuffle(unique_pairs)
    n_val = int(len(unique_pairs) * val_split)
    val_pairs = set(unique_pairs[:n_val])

    # Split data based on pairs
    train_df = merged_df[~merged_df['sample_drug1_pair'].isin(val_pairs)].copy()
    val_df = merged_df[merged_df['sample_drug1_pair'].isin(val_pairs)].copy()

    # Drop the temporary column
    train_df = train_df.drop(columns=['sample_drug1_pair'])
    val_df = val_df.drop(columns=['sample_drug1_pair'])

    # Check for any samples or drugs in validation that are not in training
    train_samples = set(train_df['sample_id'].unique())
    val_samples = set(val_df['sample_id'].unique())
    train_drugs = set(train_df['drug1'].unique())
    val_drugs = set(val_df['drug1'].unique())

    missing_samples = val_samples - train_samples
    missing_drugs = val_drugs - train_drugs

    if missing_samples or missing_drugs:
        print(f"\n  Warning: Found validation data with unseen samples/drugs:")
        if missing_samples:
            print(f"    - {len(missing_samples)} samples only in validation")
        if missing_drugs:
            print(f"    - {len(missing_drugs)} drug1 values only in validation")
        print(f"  Moving these back to training set...")

        # Move rows with unseen samples or drugs back to training
        move_to_train = (
            val_df['sample_id'].isin(missing_samples) |
            val_df['drug1'].isin(missing_drugs)
        )
        rows_to_move = val_df[move_to_train]
        train_df = pd.concat([train_df, rows_to_move], axis=0, ignore_index=True)
        val_df = val_df[~move_to_train].reset_index(drop=True)

    print(f"Train: {len(train_df)} rows, {train_df['sample_id'].nunique()} samples")
    print(f"Val: {len(val_df)} rows, {val_df['sample_id'].nunique()} samples")

    # Verify no overlap in (sample_id, drug1) pairs
    train_pairs = set(
        train_df['sample_id'].astype(str) + '_' + train_df['drug1'].astype(str)
    )
    val_pairs = set(
        val_df['sample_id'].astype(str) + '_' + val_df['drug1'].astype(str)
    )
    overlap = train_pairs & val_pairs

    if overlap:
        print(f"\n  Warning: Found {len(overlap)} overlapping (sample_id, drug1) pairs!")
    else:
        print(f"  ✓ No overlap in (sample_id, drug1) pairs between train and validation")

    n_samples = merged_df['sample_id'].max() + 1

    return train_df, val_df, n_samples


def create_optimizer(model, config: dict):
    """Create optimizer from config."""
    opt_config = config['optimizer']
    opt_type = opt_config['type'].lower()

    if opt_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.0)
        )
    elif opt_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            weight_decay=opt_config.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")

    return optimizer


def create_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Create learning rate scheduler from config."""
    if 'scheduler' not in config or config['scheduler'] is None:
        return None

    sched_config = config['scheduler']
    sched_type = sched_config['type'].lower()

    if sched_type == 'cosine':
        num_epochs = config['training']['num_epochs']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs * steps_per_epoch,
            eta_min=sched_config.get('eta_min', 0)
        )
    elif sched_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config.get('step_size', 10) * steps_per_epoch,
            gamma=sched_config.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}")

    return scheduler


def main():
    parser = argparse.ArgumentParser(description='Train drug response MLP model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded configuration from {args.config}")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Get dataset paths with filtering
    data_config = config['data']
    dataset_paths = get_dataset_paths(
        data_config['data_dir'],
        include=data_config.get('include'),
        exclude=data_config.get('exclude')
    )

    # Load drug library from FM training
    print("\n" + "="*80)
    print("Loading drug library...")
    print("="*80)
    drug_library_path = data_config['drug_library']
    drug_library = DrugLibrary.from_pretrained(drug_library_path)
    print(f"Drug library loaded (vocab size: {drug_library.vocab_size})")

    # Load and preprocess data
    print("\n" + "="*80)
    print("Loading and preprocessing data...")
    print("="*80)
    train_df, val_df, n_samples = load_and_preprocess_data(
        dataset_paths,  # Changed: pass filtered paths
        drug_library=drug_library,
        n_drugs=data_config['n_drugs'],
        val_split=data_config.get('val_split', 0.1),
        random_state=data_config.get('random_state', 42),
        min_dose_val=data_config.get('min_dose_val', -6.0),
        max_dose_val=data_config.get('max_dose_val', 4.0)
    )

    # Create datasets
    print("\n" + "="*80)
    print("Creating datasets...")
    print("="*80)
    train_dataset = DrugResponseDataset(train_df, n_drugs=data_config['n_drugs'])
    val_dataset = DrugResponseDataset(val_df, n_drugs=data_config['n_drugs'])

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 0),
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create model
    print("\n" + "="*80)
    print("Creating model...")
    print("="*80)
    model_config = config['model']
    model = DrugResponseMLP(
        n_drugs=model_config["drug_vocab_size"],
        n_samples=model_config["sample_vocab_size"],
        embedding_dim=model_config.get('embedding_dim', 64),
        hidden_dim=model_config.get('hidden_dim', 128),
        n_layers=model_config.get('n_layers', 3),
        dropout=model_config.get('dropout', 0.2),
        activation=model_config.get('activation', 'relu')
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Drug vocabulary size: {drug_library.vocab_size}")
    print(f"Sample vocabulary size: {n_samples}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    # Create trainer
    print("\n" + "="*80)
    print("Setting up trainer...")
    print("="*80)
    training_config = config['training']
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=training_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
        output_dir=training_config['output_dir'],
        gradient_clip=training_config.get('gradient_clip', 1.0)
    )

    # Load checkpoint if specified
    if 'checkpoint' in training_config and training_config['checkpoint']:
        print(f"\nLoading checkpoint: {training_config['checkpoint']}")
        trainer.load_checkpoint(training_config['checkpoint'])

    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    trainer.train(
        num_epochs=training_config['num_epochs'],
        save_every=training_config.get('save_every', 5)
    )

    print("\nTraining completed!")


if __name__ == "__main__":
    main()