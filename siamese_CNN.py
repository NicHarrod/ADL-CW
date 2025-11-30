import os
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple
from dataloader import ProgressionDataset

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import argparse
import pathlib
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For reproducibility, though it can slow down training
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Call this at the start of main
SEED = 42  # You can choose any integer
set_seed(SEED)


class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


class SiameseCNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count: int):
        super().__init__()
        self.input_shape = ImageShape(height=height, width=width, channels=channels)
        self.class_count = class_count

        print(
            f"Building Siamese CNN with input shape {self.input_shape} and {self.class_count} output classes."
        )

        # convolutional layers:
        # ((conv, batch norm) x 2 , pool) x 4

        # first conv layer pair:
        self.conv_1_0 = nn.Conv2d(
            in_channels=self.input_shape.channels, out_channels=64, kernel_size=(3, 3), padding=(1, 1),
        )
        self.batch_norm_1_0 = nn.BatchNorm2d(64)
        self.initialise_layer(self.conv_1_0)

        self.conv_1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1),)
        self.batch_norm_1_1 = nn.BatchNorm2d(64)
        self.initialise_layer(self.conv_1_1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # second conv layer pair:
        self.conv2_0 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1),)
        self.batch_norm2_0 = nn.BatchNorm2d(128)
        self.initialise_layer(self.conv2_0)

        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1),)
        self.batch_norm2_1 = nn.BatchNorm2d(128)
        self.initialise_layer(self.conv2_1)

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # third conv layer pair:
        self.conv3_0 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1),)
        self.batch_norm3_0 = nn.BatchNorm2d(256)
        self.initialise_layer(self.conv3_0)

        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1),)
        self.batch_norm3_1 = nn.BatchNorm2d(256)
        self.initialise_layer(self.conv3_1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # fourth conv layer pair:
        self.conv4_0 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1),)
        self.batch_norm4_0 = nn.BatchNorm2d(512)
        self.initialise_layer(self.conv4_0)

        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1),)
        self.batch_norm4_1 = nn.BatchNorm2d(512)
        self.initialise_layer(self.conv4_1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.dropout1 = nn.Dropout2d(p=0.5)

        # global avg pooling to reduce spatial dimension to 1x1
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # first FC layer (after concatenation of two 512-d embeddings -> 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(p=0.5)

        # output heads
        self.fc_similarity = nn.Linear(512, 1)  # similarity head (binary)
        self.fc_class = nn.Linear(512, 2)  # class direction head (0 or 1)

    def embed(self, image):
        x = self.convForward(image)
        x = self.gap(x)
        x = x.flatten(start_dim=1)
        # L2 normalization
        x = F.normalize(x, p=2, dim=1)

        return x

    def convForward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.batch_norm_1_0(self.conv_1_0(x)))
        x = F.relu(self.batch_norm_1_1(self.conv_1_1(x)))
        x = self.pool1(x)

        x = F.relu(self.batch_norm2_0(self.conv2_0(x)))
        x = F.relu(self.batch_norm2_1(self.conv2_1(x)))
        x = self.pool2(x)

        x = F.relu(self.batch_norm3_0(self.conv3_0(x)))
        x = F.relu(self.batch_norm3_1(self.conv3_1(x)))
        x = self.pool3(x)

        x = F.relu(self.batch_norm4_0(self.conv4_0(x)))
        x = F.relu(self.batch_norm4_1(self.conv4_1(x)))
        x = self.pool4(x)
        x = self.dropout1(x)

        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: list or tuple [img_a_batch, img_b_batch]
        returns: similarity_logit (B,1), class_logits (B,2), feat1 (B,512), feat2 (B,512)
        """
        feat1 = self.embed(images[0])
        feat2 = self.embed(images[1])

        # concatenate for classification
        concatenated = torch.cat((feat1, feat2), dim=1)  # shape (B, 1024)
        x = self.dropout(concatenated)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Head 1: similarity (binary)
        similarity_logit = self.fc_similarity(x)  # (B,1)
        # Head 2: class direction (0 or 1 only)
        class_logits = self.fc_class(x)  # (B,2)
        return similarity_logit, class_logits, feat1, feat2

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight") and layer.weight is not None:
            nn.init.kaiming_normal_(layer.weight)


def test_CNN():
    model = SiameseCNN(height=512, width=512, channels=3, class_count=3)
    sample_inputs = [torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512)]
    resize = transforms.Resize((224, 224))
    sample_inputs = [resize(img) for img in sample_inputs]
    output = model(sample_inputs)


def test_forward():
    model = SiameseCNN(height=512, width=512, channels=3, class_count=3)
    sample_inputs = [torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512)]
    resize = transforms.Resize((224, 224))
    sample_inputs = [resize(img) for img in sample_inputs]
    output = model(sample_inputs)


def main(args):
    torch.backends.cudnn.benchmark = True
    """
    Initialize the ProgressionDataset.
    """

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])

    train_dataset = ProgressionDataset(
        root_dir=os.path.join(args.dataset_root, "train"),
        transform=transform,
        mode="train",
        epoch_size=2000,
        recipe_ids_list=[os.path.basename(p) for p in (args.dataset_root / "train").glob("*")],
    )
    test_dataset = ProgressionDataset(
        root_dir=os.path.join(args.dataset_root, "test"),
        transform=transform,
        mode="test",
        label_file=str(args.dataset_root / "test_labels.txt"),
    )

    val_dataset = ProgressionDataset(
        root_dir=os.path.join(args.dataset_root, "val"),
        transform=transform,
        mode="val",
        label_file=str(args.dataset_root / "val_labels.txt"),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    model = SiameseCNN(height=224, width=224, channels=3, class_count=3)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)

    trainer = Trainer(model, train_loader, val_loader, test_loader, criterion, optimizer, summary_writer, DEVICE)

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
        per_class_frequency=args.per_class_frequency,
    )

    trainer.test()


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        # Euclidean distance
        dist = F.pairwise_distance(feat1, feat2)

        # label = 1 → same recipe → pull together
        # label = 0 → different recipe → push apart
        loss = label * dist.pow(2) + (1 - label) * F.relu(self.margin - dist).pow(2)

        return loss.mean()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.ce = criterion                   # CE for class 0/1
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

        self.bce = nn.BCEWithLogitsLoss()     # BCE for similarity


    ############################################################
    # ------------------------ TRAIN ---------------------------
    ############################################################
    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0,
        per_class_frequency: int = 3,
    ):
        self.model.train()

        for epoch in range(start_epoch, epochs):
            for img_a, img_b, labels in self.train_loader:
                batch = [img_a.to(self.device), img_b.to(self.device)]
                labels = labels.to(self.device)

                # similarity label (1 = same, 0 = different)
                sim_label = (labels != 2).float()

                # ---------------------- Forward ----------------------
                similarity_logit, class_logits, feat1, feat2 = self.model(batch)

                # BCE for same/different
                sim_loss = self.bce(similarity_logit.squeeze(1), sim_label)

                # CE for class 0/1, only when same
                same_mask = (labels != 2)
                if same_mask.sum() > 0:
                    ce_loss = self.ce(class_logits[same_mask], labels[same_mask])
                else:
                    ce_loss = torch.tensor(0.0, device=self.device)

                # ---------------------- Total Loss ----------------------
                loss = sim_loss + ce_loss

                # ---------------------- Backprop ----------------------
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # ---------------------- Accuracy ----------------------
                preds = self.compute_multitask_predictions(similarity_logit, class_logits)
                accuracy = compute_accuracy(labels.cpu(), preds.cpu())

                # ---------------------- Logging ------------------------
                if (self.step + 1) % log_frequency == 0:
                    self.log_metrics(epoch, accuracy, loss)

                if (self.step + 1) % print_frequency == 0:
                    print(f"epoch {epoch} | step {self.step} | "
                          f"loss {loss:.4f} | acc {accuracy*100:.2f}%")

                self.step += 1

            # after epoch
            self.summary_writer.add_scalar("epoch", epoch, self.step)

            if (epoch + 1) % val_frequency == 0:
                results = self.validate()

                if ((epoch + 1) % per_class_frequency) == 0:
                    labels_tensor = torch.tensor(results["labels"])
                    preds_tensor = torch.tensor(results["preds"])
                    print_per_class_accuracy(labels_tensor, preds_tensor, self.model.class_count)



    ############################################################
    # ---------------------- PREDICTION ------------------------
    ############################################################
    def compute_multitask_predictions(self, sim_logit, class_logits):
        """
        sim < 0.5 → class 2
        sim ≥ 0.5 → choose 0/1 by CE head
        """
        sim_prob = torch.sigmoid(sim_logit).squeeze(1)
        preds = torch.zeros_like(sim_prob, dtype=torch.long)

        # different
        diff_mask = sim_prob < 0.5
        preds[diff_mask] = 2

        # same → class 0/1
        same_mask = sim_prob >= 0.5
        preds[same_mask] = class_logits[same_mask].argmax(dim=1)

        return preds


    ############################################################
    # ----------------------- VALIDATE -------------------------
    ############################################################
    def validate(self):
        self.model.eval()
        results = {"labels": [], "preds": []}

        total_loss = 0
        with torch.no_grad():
            for img_a, img_b, labels in self.val_loader:
                batch = [img_a.to(self.device), img_b.to(self.device)]
                labels = labels.to(self.device)

                sim_label = (labels != 2).float()

                similarity_logit, class_logits, _, _ = self.model(batch)

                sim_loss = self.bce(similarity_logit.squeeze(1), sim_label)

                # CE for class 0/1 only
                same_mask = (labels != 2)
                if same_mask.sum() > 0:
                    ce_loss = self.ce(class_logits[same_mask], labels[same_mask])
                else:
                    ce_loss = torch.tensor(0.0, device=self.device)

                loss = sim_loss + ce_loss
                total_loss += loss.item()

                preds = self.compute_multitask_predictions(similarity_logit, class_logits)

                results["labels"].extend(labels.cpu().numpy())
                results["preds"].extend(preds.cpu().numpy())

        acc = compute_accuracy(np.array(results["labels"]), np.array(results["preds"]))
        avg_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars("accuracy", {"test": acc}, self.step)
        self.summary_writer.add_scalars("loss", {"test": avg_loss}, self.step)

        print(f"[VAL] Loss: {avg_loss:.4f} | Acc: {acc*100:.2f}%")

        self.model.train()
        return results


    ############################################################
    # ------------------------- TEST ---------------------------
    ############################################################
    def test(self):
        self.model.eval()
        results = {"labels": [], "preds": []}
        total_loss = 0

        with torch.no_grad():
            for img_a, img_b, labels in self.test_loader:
                batch = [img_a.to(self.device), img_b.to(self.device)]
                labels = labels.to(self.device)

                sim_label = (labels != 2).float()

                similarity_logit, class_logits, _, _ = self.model(batch)
                sim_loss = self.bce(similarity_logit.squeeze(1), sim_label)

                same_mask = (labels != 2)
                if same_mask.sum() > 0:
                    ce_loss = self.ce(class_logits[same_mask], labels[same_mask])
                else:
                    ce_loss = torch.tensor(0.0, device=self.device)

                loss = sim_loss + ce_loss
                total_loss += loss.item()

                preds = self.compute_multitask_predictions(similarity_logit, class_logits)

                results["labels"].extend(labels.cpu().numpy())
                results["preds"].extend(preds.cpu().numpy())

        acc = compute_accuracy(np.array(results["labels"]), np.array(results["preds"]))
        avg_loss = total_loss / len(self.test_loader)

        print("\n===== FINAL TEST =====")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {acc*100:.2f}%")
        print("======================\n")


    ############################################################
    # -------------------------- LOG ---------------------------
    ############################################################
    def log_metrics(self, epoch, acc, loss):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalar("loss/train", loss.item(), self.step)
        self.summary_writer.add_scalar("accuracy/train", acc, self.step)

def compute_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Args:
        labels: ``(batch_size,)`` tensor or array containing example labels
        preds: ``(batch_size,)`` tensor or array containing model prediction
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


# --- added: per-class accuracy helpers ---
def compute_per_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray], class_count: int) -> dict:
    """
    Computes per-class accuracy for each class in a multi-class classification task.
    Returns a dict mapping class index -> accuracy (0..1).
    """
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)
    if isinstance(preds, torch.Tensor):
        preds_np = preds.cpu().numpy()
    else:
        preds_np = np.array(preds)

    class_accuracies = {}
    for class_idx in range(class_count):
        mask = (labels_np == class_idx)
        total = mask.sum()
        if total == 0:
            class_accuracies[class_idx] = 0.0
        else:
            correct = (preds_np[mask] == class_idx).sum()
            class_accuracies[class_idx] = float(correct) / float(total)
    return class_accuracies


def print_per_class_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray], class_count: int):
    """
    Prints per-class accuracy as percentages.
    """
    accs = compute_per_class_accuracy(labels, preds, class_count)
    print("\nPer-class accuracy:")
    for cls, a in accs.items():
        print(f"  Class {cls}: {a * 100:5.2f}%")
# --- end added helpers ---


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.
    """
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=pathlib.Path,
        default=pathlib.Path("./dataset/"),
        help="Path to root directory of dataset.",
    )
    parser.add_argument(
        "--log_dir",
        type=pathlib.Path,
        default=pathlib.Path("./logs/siamese_CNN/"),
        help="Path to directory where TensorBoard logs should be written.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2.5e-4,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--worker_count",
        type=int,
        default=cpu_count(),
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--val_frequency",
        type=int,
        default=1,
        help="Frequency (in epochs) of validation during training.",
    )
    parser.add_argument(
        "--print_frequency",
        type=int,
        default=20,
        help="Frequency (in steps) of printing training metrics.",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=5,
        help="Frequency (in steps) of logging training metrics to TensorBoard.",
    )
    parser.add_argument(
        "--per_class_frequency",
        type=int,
        default=3,
        help="Frequency (in epochs) to print per-class accuracy during training/validation.",
    )
    args = parser.parse_args()

    main(args)
