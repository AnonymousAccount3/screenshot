import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class DrugResponseMLP(nn.Module):
    """MLP for drug dose response prediction."""

    def __init__(
        self,
        n_drugs: int,
        n_samples: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Embeddings
        self.drug_embedding = nn.Embedding(n_drugs, embedding_dim, padding_idx=0)
        self.sample_embedding = nn.Embedding(n_samples, embedding_dim)
        self.dose_encoder = nn.Linear(1, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.drug_embedding.weight)
        nn.init.xavier_uniform_(self.sample_embedding.weight)
        nn.init.xavier_uniform_(self.dose_encoder.weight)
        nn.init.zeros_(self.dose_encoder.bias)

        # MLP layers
        input_dim = 2 * embedding_dim
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, hidden_dim))
        nn.init.xavier_uniform_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(self.layers[-1].weight)
            nn.init.zeros_(self.layers[-1].bias)

        self.output_layer = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        # Activation and dropout
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'silu': nn.SiLU(),
        }
        self.activation = activations.get(activation.lower(), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, sample_ids, drug_ids, doses):
        """
        Forward pass.

        Args:
            sample_ids: (batch_size,) Sample IDs
            drug_ids: (batch_size, n_drugs) Drug IDs
            doses: (batch_size, n_drugs) Dose IDs

        Returns:
            logits: (batch_size,) Raw predictions
            viability: (batch_size,) Sigmoid predictions
        """
        sample_embeds = self.sample_embedding(sample_ids)
        drug_embeds = self.drug_embedding(drug_ids)

        # Normalize doses and encode
        doses_normalized = doses.to(drug_embeds.dtype)
        dose_embeds = self.dose_encoder(doses_normalized.unsqueeze(-1))

        # Combine drug and dose embeddings
        drug_dose_embeds = (drug_embeds * dose_embeds).mean(1)

        # Concatenate sample and drug-dose embeddings
        x = torch.cat([sample_embeds, drug_dose_embeds], dim=-1)

        # Pass through MLP
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)

        logits = self.output_layer(x).squeeze(-1)
        viability = torch.sigmoid(logits)

        return logits, viability


class Trainer:
    """Trainer for drug response model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        optimizer=None,
        scheduler=None,
        device: str = 'cuda',
        output_dir: Optional[str] = './checkpoints',
        gradient_clip: float = 1.0,
        verbose: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_clip = gradient_clip
        self.verbose = verbose

        # Setup output directory if provided
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler
        self.criterion = nn.L1Loss()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epoch = 0
        self.best_epoch = 0

    def compute_metrics(self, predictions, targets):
        """Compute evaluation metrics."""
        with torch.no_grad():
            mse = ((predictions - targets) ** 2).mean()
            mae = (torch.abs(predictions - targets)).mean()

            # R2
            ss_res = ((targets - predictions) ** 2).sum()
            ss_tot = ((targets - targets.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)

            # Pearson correlation
            vx = targets - targets.mean()
            vy = predictions - predictions.mean()
            corr = (vx * vy).sum() / (
                torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8
            )

        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'r2': r2.item(),
            'correlation': corr.item()
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        all_predictions = []
        all_targets = []

        if self.verbose:
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {self.epoch + 1} [Val]",
                ncols=100
            )
        else:
            progress_bar = self.val_loader

        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            _, predictions = self.model(
                sample_ids=batch['sample_ids'],
                drug_ids=batch['drug_ids'],
                doses=batch['doses']
            )

            all_predictions.append(predictions.cpu())
            all_targets.append(batch['viability'].cpu())

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        loss = self.criterion(all_predictions, all_targets)
        metrics = self.compute_metrics(all_predictions, all_targets)
        metrics['loss'] = loss.item()

        return metrics

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        if self.output_dir is None:
            return

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
        }

        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)

        if self.verbose:
            print(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = self.output_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            if self.verbose:
                print(f"Best model saved: {best_path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint["best_epoch"]

        if self.verbose:
            print(f"Checkpoint loaded: {filename}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        if self.verbose:
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {self.epoch + 1} [Train]",
                ncols=100
            )
        else:
            progress_bar = self.train_loader

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            _, predictions = self.model(
                sample_ids=batch['sample_ids'],
                drug_ids=batch['drug_ids'],
                doses=batch['doses']
            )

            loss = self.criterion(predictions, batch['viability'])

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1
            total_loss += loss.item()

            if self.verbose:
                avg_loss = total_loss / (step + 1)
                progress_bar.set_postfix({'loss': f"{avg_loss:.4f}"})

    def train(self, num_epochs: int, save_every: int = 5):
        """Main training loop."""
        if self.verbose:
            print(f"Starting training for {num_epochs} epochs")
            print(f"Device: {self.device}")
            if self.output_dir is not None:
                print(f"Output directory: {self.output_dir}")
            else:
                print("Output directory: None (checkpoints will not be saved)")

        for epoch in range(num_epochs):
            self.epoch = epoch

            self.train_epoch()

            if self.val_loader is not None:
                val_metrics = self.validate()

                if self.verbose:
                    print(f"Epoch {epoch + 1} Val Metrics:")
                    for k, v in val_metrics.items():
                        print(f"  {k}: {v:.4f}")

                val_loss = val_metrics['loss']
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.save_checkpoint('best_model.pt', is_best=True)

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        if self.verbose:
            print("\nTraining completed!")
            if self.val_loader is not None:
                print(f"Best validation loss: {self.best_val_loss:.4f}")


class DrugResponseDataset(Dataset):
    """Simple dataset for drug response data."""

    def __init__(self, df: pd.DataFrame, n_drugs: int):
        self.df = df.reset_index(drop=True)
        self.n_drugs = n_drugs
        self.drug_cols = [f'drug{i+1}' for i in range(n_drugs)]
        self.dose_cols = [f'dose{i+1}' for i in range(n_drugs)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        drug_ids = [row[col] for col in self.drug_cols]
        doses = [row[col] for col in self.dose_cols]

        return {
            'sample_ids': torch.tensor(row['sample_idx'], dtype=torch.long),
            'drug_ids': torch.tensor(drug_ids, dtype=torch.long),
            'doses': torch.tensor(doses, dtype=torch.float32),
            'viability': torch.tensor(row['float_value'], dtype=torch.float32),
        }


class FixedIterationDataLoader:
    """Wrapper around DataLoader that provides a fixed number of iterations per epoch."""

    def __init__(self, dataloader, iterations_per_epoch):
        self.dataloader = dataloader
        self.iterations_per_epoch = iterations_per_epoch
        self.iterator = None

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        self.current_iteration = 0
        return self

    def __next__(self):
        if self.current_iteration >= self.iterations_per_epoch:
            raise StopIteration

        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)

        self.current_iteration += 1
        return batch

    def __len__(self):
        return self.iterations_per_epoch


def split_train_val(df: pd.DataFrame, val_ratio: float = 0.2, random_state: int = 42):
    """
    Split dataframe into train and validation sets.

    Args:
        df: Input dataframe
        val_ratio: Ratio of validation samples (default 0.2 for 20%)
        random_state: Random seed for reproducibility

    Returns:
        train_df, val_df
    """
    np.random.seed(random_state)

    indices = np.arange(len(df))
    np.random.shuffle(indices)

    n_val = max(1, int(len(df) * val_ratio))

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()

    return train_df, val_df


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config