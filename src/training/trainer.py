"""Training loop with validation, early stopping, LR scheduling, and checkpointing.

All hyperparameters are driven by the YAML config loaded at construction time.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import compute_metrics, format_metrics_table
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Monitor a validation metric and signal when training should stop.

    Args:
        patience: Number of epochs to wait for improvement before stopping.
        min_delta: Minimum decrease in monitored value to count as improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Update state with latest validation loss.

        Args:
            val_loss: Validation loss for the current epoch.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.
        """
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """Full training loop for the CorruptionClassifier.

    Manages training and validation phases, early stopping, LR scheduling,
    gradient clipping, model checkpointing, and training history logging.

    Args:
        model: :class:`~src.models.resnet_classifier.CorruptionClassifier` instance.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        config: Parsed configuration dictionary (from ``default_config.yaml``).
        output_dir: Directory where checkpoints and history JSON are saved.
        device: Compute device. Auto-detected if ``None``.

    Example:
        >>> trainer = Trainer(model, train_loader, val_loader, cfg, output_dir)
        >>> history = trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        output_dir: Path,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        train_cfg = config["training"]
        self._epochs: int = train_cfg["epochs"]
        self._lr: float = train_cfg["learning_rate"]
        self._weight_decay: float = train_cfg["weight_decay"]
        self._grad_clip: float = train_cfg["grad_clip"]
        self._patience: int = train_cfg["patience"]
        self._lr_factor: float = train_cfg["lr_factor"]
        self._lr_patience: int = train_cfg["lr_patience"]
        self._lr_min: float = train_cfg["lr_min"]

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=self._lr_factor,
            patience=self._lr_patience,
            min_lr=self._lr_min,
        )
        self._early_stopping = EarlyStopping(patience=self._patience)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir = checkpoint_dir

        self._history: Dict[str, List] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }
        self._best_val_acc: float = 0.0
        self._best_epoch: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, List]:
        """Run the full training loop.

        Returns:
            Training history dictionary with per-epoch metrics.
        """
        logger.info(
            "Training started | device=%s | epochs=%d | batch_size=%d | lr=%.4e",
            self.device,
            self._epochs,
            self.train_loader.batch_size,
            self._lr,
        )
        self.model.to(self.device)
        start_time = time.time()

        for epoch in range(1, self._epochs + 1):
            train_loss, train_acc = self._run_epoch(epoch, phase="train")
            val_loss, val_acc = self._run_epoch(epoch, phase="val")
            current_lr = self._optimizer.param_groups[0]["lr"]

            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)
            self._history["train_acc"].append(train_acc)
            self._history["val_acc"].append(val_acc)
            self._history["lr"].append(current_lr)

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | train_acc=%.4f | "
                "val_loss=%.4f | val_acc=%.4f | lr=%.2e",
                epoch,
                self._epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                current_lr,
            )

            # Save best checkpoint
            if val_acc > self._best_val_acc:
                self._best_val_acc = val_acc
                self._best_epoch = epoch
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)

            # LR scheduling
            self._scheduler.step(val_loss)

            # Early stopping
            if self._early_stopping.step(val_loss):
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d). "
                    "Best val_acc=%.4f at epoch %d.",
                    epoch,
                    self._patience,
                    self._best_val_acc,
                    self._best_epoch,
                )
                break

        elapsed = time.time() - start_time
        logger.info(
            "Training complete in %.1fs | Best val_acc=%.4f at epoch %d",
            elapsed,
            self._best_val_acc,
            self._best_epoch,
        )
        self._save_history()
        return self._history

    def evaluate(
        self,
        loader: DataLoader,
        class_names: Optional[List[str]] = None,
    ) -> Tuple[float, Dict]:
        """Evaluate the model on a DataLoader and return loss + full metrics.

        Args:
            loader: DataLoader to evaluate on (typically the test set).
            class_names: Class name strings for the metrics report.

        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_true: List[int] = []

        with torch.no_grad():
            for imgs, labels, _ in tqdm(loader, desc="Evaluating", leave=False):
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(imgs)
                loss = self._criterion(logits, labels)
                total_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_true.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(loader.dataset)
        metrics = compute_metrics(all_true, all_preds, class_names)
        logger.info("\n%s", format_metrics_table(metrics, class_names))
        return avg_loss, metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, phase: str) -> Tuple[float, float]:
        """Run a single training or validation epoch.

        Args:
            epoch: Current epoch number (1-based, for logging only).
            phase: ``"train"`` or ``"val"``.

        Returns:
            Tuple of (average_loss, accuracy) for the epoch.
        """
        is_train = phase == "train"
        self.model.train(is_train)
        loader = self.train_loader if is_train else self.val_loader

        total_loss = 0.0
        correct = 0
        total = 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            for imgs, labels, _ in tqdm(
                loader,
                desc=f"Epoch {epoch:3d} [{phase}]",
                leave=False,
            ):
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if is_train:
                    self._optimizer.zero_grad()

                logits = self.model(imgs)
                loss = self._criterion(logits, labels)

                if is_train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self._grad_clip
                    )
                    self._optimizer.step()

                batch_size = imgs.size(0)
                total_loss += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size

        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool = False,
    ) -> None:
        """Persist the current model state to disk.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss at this epoch.
            val_acc: Validation accuracy at this epoch.
            is_best: If ``True``, also save as ``best_model.pt``.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "config": self.config,
        }
        path = self._checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt"
        torch.save(checkpoint, path)
        if is_best:
            best_path = self._checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(
                "New best model saved (epoch=%d, val_acc=%.4f) -> %s",
                epoch,
                val_acc,
                best_path,
            )

    def _save_history(self) -> None:
        """Write the training history dict to a JSON file.

        The file is saved to ``{output_dir}/training_history.json``.
        """
        history_path = self.output_dir / "training_history.json"
        # Convert numpy types to plain Python for JSON serialisation
        serialisable = {
            k: [float(v) for v in vals] for k, vals in self._history.items()
        }
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)
        logger.info("Training history saved to %s", history_path)
