import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import sys
from datetime import datetime


class Logger:
    """Simple logger that writes to both console and file."""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.terminal = sys.stdout

        if self.log_file:
            # Create parent directory if needed
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            # Open file in append mode
            self.file = open(log_file, 'a')
            # Write header with timestamp
            self.file.write(f"\n{'='*80}\n")
            self.file.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file.write(f"{'='*80}\n\n")
            self.file.flush()

    def write(self, message):
        """Write message to both terminal and file."""
        self.terminal.write(message)
        if self.log_file:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        if self.log_file:
            self.file.flush()

    def close(self):
        """Close file handle."""
        if self.log_file:
            self.file.write(f"\n{'='*80}\n")
            self.file.write(f"Training ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.file.write(f"{'='*80}\n\n")
            self.file.close()


class FoundationModelTrainer:
    """Trainer for DoseResponseGenerator model."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        output_dir: str = './model_checkpoints',
        gradient_clip: float = 1.0,
        callbacks = [],
        log_file: Optional[str] = None,
    ):
        """
        Args:
            model: FoundationModel
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (AdamW if None)
            scheduler: Learning rate scheduler
            device: Device to train on
            output_dir: Directory to save checkpoints
            gradient_clip: Max gradient norm for clipping
            callbacks: List of callback functions
            log_file: Path to log file (if None, only prints to console)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_clip = gradient_clip
        self.callbacks = callbacks

        # Setup logger
        if log_file is None:
            # Default log file in output directory
            log_file = str(self.output_dir / 'training.log')
        self.logger = Logger(log_file)

        # Redirect print to logger
        self._original_stdout = sys.stdout
        sys.stdout = self.logger

        # Setup optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                model.parameters(),
                lr=1e-5,
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Loss function - MSE for regression
        self.criterion = nn.L1Loss(reduction='none')

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_epoch = 0
        self.epoch = 0

    def __del__(self):
        """Cleanup: restore stdout and close logger."""
        if hasattr(self, 'logger'):
            sys.stdout = self._original_stdout
            self.logger.close()

    def compute_loss(self, predictions, targets):
        """
        Compute masked MSE loss.

        Args:
            predictions: (bs, n_perturb) - Predicted viability
            targets: (bs, n_perturb) - True viability
        """
        # Compute element-wise loss
        loss = self.criterion(predictions, targets).mean()
        return loss

    def compute_metrics(self, predictions, targets):
        """Compute evaluation metrics."""
        with torch.no_grad():
            # MSE
            mse = ((predictions - targets) ** 2).mean()

            # MAE
            mae = (torch.abs(predictions - targets)).mean()

            # R2
            ss_res = ((targets - predictions) ** 2).sum()
            ss_tot = ((targets - predictions.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)

            # Pearson correlation
            vx = targets - targets.mean()
            vy = predictions - predictions.mean()
            correlation = (vx * vy).sum() / (
                torch.sqrt((vx ** 2).sum()) * torch.sqrt((vy ** 2).sum()) + 1e-8
            )

        return {
            'mse': mse.item(),
            'mae': mae.item(),
            'r2': r2.item(),
            'correlation': correlation.item()
        }

    def _compute_cindex(self, predictions, targets, sample_ids, drug_ids, dose_levels, threshold=0.2):
        """Compute concordance index across samples for shared treatments.

        Groups predictions by treatment (drug+dose combination), forms cross-sample
        pairs, and checks if predicted ranking matches true ranking.
        Only counts pairs where |true_diff| > threshold.
        """
        import pandas as pd

        n_drugs = drug_ids.shape[1]

        # Build dataframe
        data = {
            'pred': predictions,
            'true': targets,
            'sample_id': sample_ids,
        }
        for d in range(n_drugs):
            data[f'drug{d+1}'] = drug_ids[:, d]
            data[f'dose{d+1}'] = np.round(dose_levels[:, d], 6)

        df = pd.DataFrame(data)

        # Average replicates (same sample + same treatment)
        drug_cols = [f'drug{d+1}' for d in range(n_drugs)]
        dose_cols = [f'dose{d+1}' for d in range(n_drugs)]
        group_cols = ['sample_id'] + drug_cols + dose_cols
        df = df.groupby(group_cols).agg(
            pred=('pred', 'mean'),
            true=('true', 'mean'),
        ).reset_index()

        # Create treatment key for grouping
        df['treatment'] = df[drug_cols + dose_cols].apply(tuple, axis=1)

        concordant = 0
        discordant = 0

        for _, group in df.groupby('treatment'):
            if len(group) < 2:
                continue

            true_vals = group['true'].values
            pred_vals = group['pred'].values
            n = len(true_vals)

            # Vectorized pairwise differences
            true_diff = true_vals[:, None] - true_vals[None, :]
            pred_diff = pred_vals[:, None] - pred_vals[None, :]

            # Upper triangle only (avoid double-counting)
            tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            true_diff = true_diff[tri_mask]
            pred_diff = pred_diff[tri_mask]

            # Apply threshold — only count clinically meaningful differences
            significant = np.abs(true_diff) >= threshold
            true_diff = true_diff[significant]
            pred_diff = pred_diff[significant]

            concordant += int(np.sum(true_diff * pred_diff > 0))
            discordant += int(np.sum(true_diff * pred_diff < 0))

        total = concordant + discordant
        if total == 0:
            return float('nan')

        return concordant / total

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        # Collect all predictions, targets, and metadata for C-index
        all_predictions = []
        all_targets = []
        all_sample_ids = []
        all_drug_ids = []
        all_dose_levels = []

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.epoch + 1} [Val]",
            ncols=100,
            file=self._original_stdout  # Progress bar to original stdout
        )

        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if v is not None}

            # Forward pass
            outputs = self.model(**batch)

            predictions = outputs["predictions"].view(-1)
            labels = batch['label_viability'].view(-1)

            mask = ~torch.isnan(labels)

            predictions = predictions[mask]
            labels = labels[mask]

            # Collect metadata for C-index
            if 'perturb_drug_ids' in batch and 'label_dose' in batch:
                perturb_drugs = batch['perturb_drug_ids'].reshape(-1, batch['perturb_drug_ids'].shape[-1])
                query_doses = batch['label_dose'].reshape(-1, batch['label_dose'].shape[-1])
                sample_id_val = batch['sample_ids'].view(-1)[0].item()

                all_drug_ids.append(perturb_drugs[mask].detach().cpu())
                all_dose_levels.append(query_doses[mask].detach().cpu())
                all_sample_ids.append(np.full(predictions.shape[0], sample_id_val))

            # Collect predictions and targets
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(labels.detach().cpu())

        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute loss on all data
        loss = self.compute_loss(all_predictions, all_targets)

        # Compute metrics on all data
        metrics = self.compute_metrics(all_predictions, all_targets)

        # Compute C-index with threshold 0.2 if metadata was collected
        if all_sample_ids:
            try:
                c_index = self._compute_cindex(
                    all_predictions.numpy(),
                    all_targets.numpy(),
                    np.concatenate(all_sample_ids),
                    torch.cat(all_drug_ids, dim=0).numpy(),
                    torch.cat(all_dose_levels, dim=0).numpy(),
                    threshold=0.2,
                )
                metrics['c_index_0.2'] = c_index
            except Exception as e:
                print(f"  Warning: C-index computation failed: {e}")

        return metrics

    def save_model_pretrained(self, save_name: str = "model"):
        """
        Save model using save_pretrained method.

        Args:
            save_name: Name of the subdirectory to save the model
        """
        save_path = self.output_dir / save_name
        self.model.save_pretrained(save_path)
        print(f"Model saved with save_pretrained: {save_path}")

    def run_callbacks(self):
        for callback in self.callbacks:
            callback(self)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1} [Train]",
            ncols=120,
            file=self._original_stdout  # Progress bar to original stdout
        )

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if v is not None}

            # Forward pass
            outputs = self.model(**batch)

            # Compute loss
            loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            # Optimizer step
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

            total_loss += loss.item()

            avg_loss = total_loss / (step + 1)

            progress_bar.set_postfix({
                'loss': f"{avg_loss:.4f}",
            })

        return avg_loss

    def train(self, num_epochs: int):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        print(f"Log file: {self.logger.log_file}")
        print()

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")
            avg_loss = self.train_epoch()

            # Validate
            if self.val_loader is not None:
                val_metrics = self.validate()
                print(f"\nEpoch {epoch + 1} Metrics:")
                print(f"  Train Loss: {avg_loss:.4f}")
                for k, v in val_metrics.items():
                    print(f"  Val {k}: {v:.4f}")

                # Save best model
                val_loss = val_metrics['mae']
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.best_val_epoch = epoch
                    print(f"\n  New best model! Val MAE: {val_loss:.4f}")
                    self.save_model_pretrained("best_model")
                else:
                    print(f"\n  Best Val MAE: {self.best_val_loss:.4f} (Epoch {self.best_val_epoch + 1})")

            self.run_callbacks()

        # Save final model using save_pretrained
        print(f"\n{'='*80}")
        print("Saving final model...")
        self.save_model_pretrained("final_model")

        print(f"\n{'='*80}")
        print("Training completed!")
        if self.val_loader is not None:
            print(f"Best validation MAE: {self.best_val_loss:.4f} at epoch {self.best_val_epoch + 1}")
        print(f"{'='*80}\n")