import json
import math
from pathlib import Path

import lightning
import torch
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from torch import nn

from src.metrics import compute_metrics, plot_confusion_matrix
from src.models import build_network, split_parameters


def build_schedule(optimizer, total_steps, warmup_steps, cosine_decay):
    """Linear warmup, then constant lr or cosine decay to 0. Stepped every batch."""
    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if not cosine_decay:
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)


class ArtworkClassifier(lightning.LightningModule):
    def __init__(
        self,
        classes: list[str],
        model_name: str,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        img_size: int = 224,
        lr_backbone: float = 1e-4,
        lr_head: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_epochs: float = 0.0,
        cosine_decay: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.classes = classes
        self.network = build_network(model_name, len(classes), img_size, pretrained)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if freeze_backbone:
            backbone, _ = split_parameters(self.network)
            for p in backbone:
                p.requires_grad = False

        self.outputs = []
        self.train_losses = []
        self.val_metrics = {}

    def forward(self, x):
        if x.dim() == 2:
            # precomputed features (data.cache_features): only the head runs
            if not self.hparams.freeze_backbone:
                raise ValueError("data.cache_features needs model.freeze_backbone: true")
            return self.network.get_classifier()(x)
        return self.network(x)

    def configure_optimizers(self):
        backbone, head = split_parameters(self.network)
        groups = [{"params": head, "lr": self.hparams.lr_head}]
        if not self.hparams.freeze_backbone:
            groups.insert(0, {"params": backbone, "lr": self.hparams.lr_backbone})
        optimizer = torch.optim.AdamW(groups, weight_decay=self.hparams.weight_decay)

        total_steps = self.trainer.estimated_stepping_batches
        steps_per_epoch = total_steps / self.trainer.max_epochs
        warmup_steps = int(self.hparams.warmup_epochs * steps_per_epoch)
        scheduler = build_schedule(optimizer, total_steps, warmup_steps, self.hparams.cosine_decay)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def configure_callbacks(self):
        if self.logger:
            return [LearningRateMonitor(logging_interval="epoch")]
        return []

    def on_train_epoch_start(self):
        self.train_losses = []
        if self.hparams.freeze_backbone:
            # otherwise the batchnorm running stats still get updated
            self.network.eval()

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        if not torch.isfinite(loss):
            raise RuntimeError("loss is nan/inf, try a lower lr or precision 32-true")
        self.train_losses.append(loss.detach())
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        train_loss = torch.stack(self.train_losses).mean().item()
        print(f"epoch {self.current_epoch + 1} | train_loss {train_loss:.4f} | val_loss {self.val_metrics['loss']:.4f} | "
              f"val_macro_f1 {self.val_metrics['macro_f1']:.4f} | val_accuracy {self.val_metrics['accuracy']:.4f}")

    def eval_step(self, batch):
        x, y = batch
        self.outputs.append((self(x).float().cpu(), y.cpu()))

    def eval_epoch_end(self, split):
        logits = torch.cat([o[0] for o in self.outputs])
        labels = torch.cat([o[1] for o in self.outputs])
        self.outputs = []

        metrics = compute_metrics(logits.numpy(), labels.numpy(), self.classes)
        # validate on a checkpoint logs to val_best/*, separate from the fit curves
        key = "val_best" if split == "val" and self.trainer.state.fn != "fit" else split
        if self.trainer.state.fn == "fit":
            metrics["loss"] = self.criterion(logits, labels).item()
            self.log("val/loss", metrics["loss"])
        self.log(f"{key}/macro_f1", metrics["macro_f1"])
        self.log(f"{key}/accuracy", metrics["accuracy"])
        for c, f1 in metrics["f1_per_class"].items():
            self.log(f"{key}_f1/{c}", f1)

        if self.trainer.state.fn == "fit":
            self.val_metrics = metrics
            return

        # validate / test: save the metrics in the run folder
        out = Path(self.trainer.default_root_dir) / f"metrics_{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))

        cm_path = out.parent / f"confusion_matrix_{split}.png"
        plot_confusion_matrix(metrics["confusion_matrix"], self.classes, cm_path,
                              title=f"{split} | macro-F1 {metrics['macro_f1']:.3f}")
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(key=f"{key}/confusion_matrix", images=[str(cm_path)])
        print(f"[{split}] macro-F1 {metrics['macro_f1']:.4f} | acc {metrics['accuracy']:.4f}")
        for c in self.classes:
            print(f"    {c:<20} F1 {metrics['f1_per_class'][c]:.4f}  (n={metrics['support'][c]})")
        print(f"saved {cm_path}")

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch)

    def on_validation_epoch_end(self):
        self.eval_epoch_end("val")

    def test_step(self, batch, batch_idx):
        self.eval_step(batch)

    def on_test_epoch_end(self):
        self.eval_epoch_end("test")
