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

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self(features)

        loss = F.cross_entropy(logits, true_labels)
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

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)

        self.test_acc(predicted_labels, true_labels)
        self.test_macro_precision(predicted_labels, true_labels)
        self.test_macro_recall(predicted_labels, true_labels)
        self.test_macro_f1(predicted_labels, true_labels)
        self.test_precision_per_class(predicted_labels, true_labels)
        self.test_recall_per_class(predicted_labels, true_labels)
        self.test_f1_per_class(predicted_labels, true_labels)

        self.log("test_acc", self.test_acc, metric_attribute="test_acc")
        self.log(
            "test_macro_precision",
            self.test_macro_precision,
            metric_attribute="test_macro_precision",
        )
        self.log(
            "test_macro_recall",
            self.test_macro_recall,
            metric_attribute="test_macro_recall",
        )
        self.log("test_macro_f1", self.test_macro_f1, metric_attribute="test_macro_f1")

        # Log per-class metrics with class names
        precision_per_class = self.test_precision_per_class.compute()
        recall_per_class = self.test_recall_per_class.compute()
        f1_per_class = self.test_f1_per_class.compute()

        for i in range(self.num_classes):
            class_name = IDX_TO_CLASS[i]
            self.log(f"test_precision_{class_name}", precision_per_class[i])
            self.log(f"test_recall_{class_name}", recall_per_class[i])
            self.log(f"test_f1_{class_name}", f1_per_class[i])

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cosine_t_max
        )  # New!

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "train_loss",
                "interval": "step",  # step means "batch" here, default: epoch   # New!
                "frequency": 1,  # default
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
        # Weighted sampling
        # Extract labels from all datasets in the ConcatDataset
        all_labels = []
        for dataset in self.train_dataset.datasets:
            all_labels.extend(dataset.imgs)

        labels = torch.tensor([label for _, label in all_labels])

        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()

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
