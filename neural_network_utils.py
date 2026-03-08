import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path

MAIZE_CATEGORIES = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_HD", "7_IM"]
RICE_CATEGORIES = ["0_NOR", "1_F&S", "2_SD", "3_MY", "4_AP", "5_BN", "6_UN", "7_IM"]

CATEGORIES_MAP = {"maize": MAIZE_CATEGORIES, "rice": RICE_CATEGORIES}

ALL_CLASSES = []

for grain, categories in CATEGORIES_MAP.items():
    for c in categories:
        ALL_CLASSES.append(f"{grain}_{c}")

CLASS_TO_IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class LightningModele(L.LightningModule):
    def __init__(self, model, learning_rate, num_classes):
        self.learning_rate = learning_rate
        self.model = model

        self.training_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

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

        self.log("val_loss", loss)
        self.val_acc(predicted_labels, true_labels)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        _, true_labels, predicted_labels = self._shared_step(batch)

        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


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
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Train transformations (with augmentations)
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Val transformations (minimal processing)
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Test transformations (minimal processing)
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
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
