import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

MAIZE_CATEGORIES = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_HD"]
RICE_CATEGORIES = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN"]

CATEGORIES_MAP = {"maize": MAIZE_CATEGORIES, "rice": RICE_CATEGORIES}

ALL_CLASSES = []

for grain, categories in CATEGORIES_MAP.items():
    for c in categories:
        ALL_CLASSES.append(f"{grain}_{c}")

CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate, num_classes, cosine_t_max):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model
        self.cosine_t_max = cosine_t_max

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_macro_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_f1_per_class = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )

        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_macro_precision = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_macro_recall = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_macro_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_precision_per_class = torchmetrics.Precision(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.test_recall_per_class = torchmetrics.Recall(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.test_f1_per_class = torchmetrics.F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )
        self.num_classes = num_classes

        # Store predictions and targets for epoch-end logging (like confusion matrix)
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        # Apply label smoothing (class weights kept optional as in your draft)
        loss = F.cross_entropy(logits, true_labels, label_smoothing=0.1)

        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_acc(predicted_labels, true_labels)
        self.log(
            "train_acc", self.train_acc, prog_bar=True, on_epoch=True, on_step=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.val_macro_f1(predicted_labels, true_labels)
        self.log(
            "val_macro_f1",
            self.val_macro_f1,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )

        self.val_f1_per_class(predicted_labels, true_labels)

        # Store for epoch-end logging
        self.val_preds.append(predicted_labels.detach().cpu())
        self.val_targets.append(true_labels.detach().cpu())

    def on_validation_epoch_end(self):
        if not self.val_preds or not self.val_targets:
            return

        f1_per_class = self.val_f1_per_class.compute()
        for class_idx in range(self.num_classes):
            class_name = IDX_TO_CLASS[class_idx]
            self.log(f"val_f1_{class_name}", f1_per_class[class_idx])

        # Reset the metric for the next epoch
        self.val_f1_per_class.reset()

        # Concatenate all stored predictions and targets for Comet
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()

        # Access the Comet logger via self.loggers
        if self.loggers:
            for logger in self.loggers:
                if hasattr(logger, "experiment") and hasattr(
                    logger.experiment, "log_confusion_matrix"
                ):
                    labels = [IDX_TO_CLASS[i] for i in range(self.num_classes)]
                    logger.experiment.log_confusion_matrix(
                        y_true=targets,
                        y_predicted=preds,
                        labels=labels,
                        title=f"Validation Confusion Matrix Epoch {self.current_epoch}",
                        file_name=f"val_confusion_matrix_epoch_{self.current_epoch}.json",
                    )

        # Clear paths to prevent memory leaks
        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)

        # Just update the metrics
        self.test_acc(predicted_labels, true_labels)
        self.test_macro_precision(predicted_labels, true_labels)
        self.test_macro_recall(predicted_labels, true_labels)
        self.test_macro_f1(predicted_labels, true_labels)
        self.test_precision_per_class(predicted_labels, true_labels)
        self.test_recall_per_class(predicted_labels, true_labels)
        self.test_f1_per_class(predicted_labels, true_labels)

        # Log scalar metrics directly
        self.log("test_acc", self.test_acc)
        self.log("test_macro_precision", self.test_macro_precision)
        self.log("test_macro_recall", self.test_macro_recall)
        self.log("test_macro_f1", self.test_macro_f1)

        # Store for epoch-end logging
        self.test_preds.append(predicted_labels.detach().cpu())
        self.test_targets.append(true_labels.detach().cpu())

    def on_test_epoch_end(self):
        if not self.test_preds or not self.test_targets:
            return

        precision_per_class = self.test_precision_per_class.compute()
        recall_per_class = self.test_recall_per_class.compute()
        f1_per_class = self.test_f1_per_class.compute()

        for i in range(self.num_classes):
            class_name = IDX_TO_CLASS[i]
            self.log(f"test_precision_{class_name}", precision_per_class[i])
            self.log(f"test_recall_{class_name}", recall_per_class[i])
            self.log(f"test_f1_{class_name}", f1_per_class[i])

        self.test_precision_per_class.reset()
        self.test_recall_per_class.reset()
        self.test_f1_per_class.reset()

        preds = torch.cat(self.test_preds).numpy()
        targets = torch.cat(self.test_targets).numpy()

        if self.loggers:
            for logger in self.loggers:
                if hasattr(logger, "experiment") and hasattr(
                    logger.experiment, "log_confusion_matrix"
                ):
                    labels = [IDX_TO_CLASS[i] for i in range(self.num_classes)]
                    logger.experiment.log_confusion_matrix(
                        y_true=targets,
                        y_predicted=preds,
                        labels=labels,
                        title="Test Confusion Matrix",
                    )

        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cosine_t_max
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",
                "frequency": 1,
            },
        }


class GrainDataset(ImageFolder):
    def __init__(self, root, grain_type, transform=None):
        self.grain_type = grain_type
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        class_name = self.classes[label]
        full_label = f"{self.grain_type}_{class_name}"

        label_idx = CLASS_TO_IDX[full_label]
        return img, label_idx


class GrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        train_transform,
        val_transform,
        test_transform,
        batch_size=32,
        num_workers=4,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage=None):
        maize_train = GrainDataset(
            self.data_dir / "maize/train",
            grain_type="maize",
            transform=self.train_transform,
        )

        rice_train = GrainDataset(
            self.data_dir / "rice/train",
            grain_type="rice",
            transform=self.train_transform,
        )

        maize_val = GrainDataset(
            self.data_dir / "maize/val",
            grain_type="maize",
            transform=self.val_transform,
        )

        rice_val = GrainDataset(
            self.data_dir / "rice/val", grain_type="rice", transform=self.val_transform
        )

        maize_test = GrainDataset(
            self.data_dir / "maize/test",
            grain_type="maize",
            transform=self.test_transform,
        )

        rice_test = GrainDataset(
            self.data_dir / "rice/test",
            grain_type="rice",
            transform=self.test_transform,
        )

        self.train_dataset = ConcatDataset([maize_train, rice_train])
        self.val_dataset = ConcatDataset([maize_val, rice_val])
        self.test_dataset = ConcatDataset([maize_test, rice_test])

    def train_dataloader(self):
        # Weighted sampling with proper global label mapping
        class_counts = _count_class_samples(self.train_dataset, len(CLASS_TO_IDX))
        class_weights = 1.0 / class_counts.float()

        # Extract labels for each sample to map to weights
        all_labels = []
        for dataset in self.train_dataset.datasets:
            for img_path, local_label in dataset.imgs:
                class_name = dataset.classes[local_label]
                full_label = f"{dataset.grain_type}_{class_name}"
                global_label = CLASS_TO_IDX[full_label]
                all_labels.append(global_label)

        labels = torch.tensor(all_labels)
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            sampler=sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def plot_loss_and_acc(
    log_dir, loss_ylim=(0.0, 0.9), acc_ylim=(0.7, 1.0), save_loss=None, save_acc=None
):

    metrics = pd.read_csv(f"{log_dir}/metrics.csv")

    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )

    plt.ylim(loss_ylim)
    if save_loss is not None:
        plt.savefig(save_loss)

    df_metrics[["train_acc", "val_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    plt.ylim(acc_ylim)
    if save_acc is not None:
        plt.savefig(save_acc)


def _count_class_samples(train_dataset, num_classes):
    class_counts = torch.zeros(num_classes)

    for dataset in train_dataset.datasets:
        for img_path, local_label in dataset.imgs:
            class_name = dataset.classes[local_label]
            full_label = f"{dataset.grain_type}_{class_name}"
            global_label = CLASS_TO_IDX[full_label]
            class_counts[global_label] += 1

    return class_counts
