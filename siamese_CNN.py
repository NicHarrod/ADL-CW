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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import argparse
import pathlib
import random 
from sklearn.metrics import confusion_matrix



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
SEED = 42 # You can choose any integer
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
        
        print(f"Building Siamese CNN with input shape {self.input_shape} and {self.class_count} output classes.")

        self.activation = nn.LeakyReLU(negative_slope=0.01)

        # convolutional layers:
        # ((conv, batch norm) x 2 , pool) x 4
        # named as conv_(layer number)_(conv number in layer)
        # and batch_norm_(layer number)_(conv number in layer)

        # first conv layer pair:
        
        self.conv_1_0 = nn.Conv2d(in_channels=self.input_shape.channels,out_channels=64,kernel_size=(3, 3),padding=(1, 1),)
        self.batch_norm_1_0 = nn.BatchNorm2d(64)
        self.initialise_layer(self.conv_1_0)

        self.conv_1_1 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3, 3),padding=(1, 1),)
        self.batch_norm_1_1 = nn.BatchNorm2d(64)
        self.initialise_layer(self.conv_1_1)

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # second conv layer pair:

        self.conv2_0 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3, 3),padding=(1, 1),)
        self.batch_norm2_0 = nn.BatchNorm2d(128)
        self.initialise_layer(self.conv2_0)

        self.conv2_1 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3, 3),padding=(1, 1),)
        self.batch_norm2_1 = nn.BatchNorm2d(128)
        self.initialise_layer(self.conv2_1)

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        
        # third conv layer pair:

        self.conv3_0 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3, 3),padding=(1, 1),)
        self.batch_norm3_0 = nn.BatchNorm2d(256)
        self.initialise_layer(self.conv3_0)

        self.conv3_1 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=(3, 3),padding=(1,1),)
        self.batch_norm3_1 = nn.BatchNorm2d(256)
        self.initialise_layer(self.conv3_1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # fourth conv layer pair:

        self.conv4_0 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3, 3),padding=(1, 1),)
        self.batch_norm4_0 = nn.BatchNorm2d(512)
        self.initialise_layer(self.conv4_0)

        self.conv4_1 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=(3, 3),padding=(1, 1),)
        self.batch_norm4_1 = nn.BatchNorm2d(512)
        self.initialise_layer(self.conv4_1)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # self.dropout_conv4 = nn.Dropout2d(p=0.3)

        

    

        # fully connected layers:
              
        # global avg pooling to reduce spatial dimension from 14 x14 to 1 x 1
        self.gap = nn.AdaptiveAvgPool2d((1,1))

        # first FC layer
        self.fc1 = nn.Linear (1024, 512)
        self.fc2 = nn.Linear (512,3)
        self.dropout = nn.Dropout(p=0.5)

    def convForward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.activation(self.batch_norm_1_0(self.conv_1_0(x)))
        x = self.activation(self.batch_norm_1_1(self.conv_1_1(x))) 
        #x = self.dropout_conv1 (x)
        x= self.pool1(x)  


        x= self.activation(self.batch_norm2_0(self.conv2_0(x)))
        x= self.activation(self.batch_norm2_1(self.conv2_1(x)))
        #x = self.dropout_conv2 (x)
        x= self.pool2(x)   


        x= self.activation(self.batch_norm3_0(self.conv3_0(x)))
        x= self.activation(self.batch_norm3_1(self.conv3_1(x)))
        #x = self.dropout_conv3 (x)
        x= self.pool3(x)
        

        x= self.activation(self.batch_norm4_0(self.conv4_0(x)))
        x= self.activation(self.batch_norm4_1(self.conv4_1(x)))
        # x = self.dropout_conv4 (x)
        x= self.pool4(x)

        

        return x



    def forward(self, images: torch.Tensor) -> torch.Tensor:


        # do conv for each image in the pair
        conv_outputs = []
        # [anchor,comparator]
        for image in images:
            x = self.convForward(image)
            conv_outputs.append(x)
        

        # concatenate the outputs to give 1x1x1024 - here x and y are results of convolution at layer 7 and 8 
        feat_map_1 = self.gap (conv_outputs[0])
        feat_map_2 = self.gap (conv_outputs[1])

        # flatten the feature maps to turn (B,512,1,1) into (B,512)
        flat_feat_map_1 = feat_map_1.flatten(start_dim=1)
        flat_feat_map_2 = feat_map_2.flatten(start_dim=1)

        concatenated_features = torch.cat ((flat_feat_map_1, flat_feat_map_2), dim=1)
        # x = self.dropout(concatenated_features)
        # print (f"concatenated shape: {concatenated_features.shape}")
        # Pass through fully connected layers with ReLU and dropout
        x = self.activation(self.fc1(concatenated_features))
        x = self.dropout(x)
        output = self.fc2(x)
    
        return output
    

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


def test_CNN():
    model = SiameseCNN(height=512, width=512, channels=3, class_count=3)
    # generate 2 batches of 1 image each
    sample_inputs = [torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512)]
    # resize inputs to 224x224 using transforms
    resize = transforms.Resize((224, 224))
    sample_inputs = [resize(img) for img in sample_inputs]
    output = model(sample_inputs)

def test_forward():
    model = SiameseCNN(height=512, width=512, channels=3, class_count=3)
    # generate 2 batches of 1 image each
    sample_inputs = [torch.randn(1, 3, 512, 512), torch.randn(1, 3, 512, 512)]
    # resize inputs to 224x224 using transforms
    resize = transforms.Resize((224, 224))
    sample_inputs = [resize(img) for img in sample_inputs]
    output = model(sample_inputs)
    # print(output.shape)
def main(args):
    torch.backends.cudnn.benchmark = True
    """
        Initialize the ProgressionDataset.

        Parameters
        ----------
        root_dir : str
            Root directory containing recipe folders or image pairs.
        transform : callable, optional
            Optional torchvision transform for image preprocessing.
        mode : str, default='train'
            Operation mode: 'train', 'val', or 'test'.
        recipe_ids_list : list of str, optional
            List of recipe folder names (required for 'train' mode).
        epoch_size : int, optional
            Number of samples per epoch (required for 'train' mode).
        label_file : str, optionaltransforms.CenterCrop(int(224 * 0.4)),
            Path to text file containing image pair indices and labels 
            (required for 'val'/'test' mode).
        """


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_transform = transforms.Compose([
        # Optionally apply random brightness/contrast adjustments
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
        ], p=0.5),  # Apply with 50% probability

        # Randomly apply greyscale conversion
        transforms.RandomGrayscale(p=0.1),  # 10% chance to convert to grayscale


        # Resize image to ensure consistency in size (224x224)
        transforms.Resize(256),  # Resize to a larger size to allow cropping
        
        # Crop the bottom half of the image with random resize to 224x224
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.0)),
        
        # Convert the image to a tensor and normalize
        transforms.ToTensor(),
    ])



    
    train_dataset = ProgressionDataset(
        root_dir=os.path.join(args.dataset_root, "train"), transform=train_transform, mode="train", epoch_size = 2000,  recipe_ids_list=[os.path.basename(p) for p in (args.dataset_root / "train").glob("*")]
    )
    test_dataset = ProgressionDataset(
        root_dir=os.path.join(args.dataset_root, "test"), transform=transform, mode="test",  label_file=str(args.dataset_root / "test_labels.txt")
    )


    val_dataset = ProgressionDataset(   
        root_dir=os.path.join(args.dataset_root, "val"), transform=transform, mode="val",  label_file=str(args.dataset_root / "val_labels.txt")
    )
    # ---- Loaders ----
    train_loader = DataLoader(
        train_dataset,
        shuffle = True,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
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
    weights = torch.tensor([1.0, 1.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    trainer = Trainer(
        model, train_loader, val_loader, test_loader, criterion, optimizer, summary_writer, DEVICE,resume_from_checkpoint=args.resume_from_checkpoint
    )

    trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    #summary_writer.close()
    trainer.test()



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
        resume_from_checkpoint: str = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

        if resume_from_checkpoint:
            self.load_model(resume_from_checkpoint)
    def load_model(self, checkpoint_path: str):
        """
        Loads the model's weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded from checkpoint.")

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for img_a, img_b, labels in self.train_loader:
                batch = [img_a, img_b]
                batch = [b.to(self.device) for b in batch]
                labels = labels.to(self.device)
                data_load_end_time = time.time()



                logits = self.model.forward(batch)
                # print(logits.shape)
             

                loss = self.criterion(logits,labels)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
                with torch.no_grad():
                    preds = logits.argmax(-1)
                    accuracy = compute_accuracy(labels, preds)

                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time

                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()

                if ((epoch + 1) % 5) == 0:
                    self.compute_and_print_confusion_matrix(self.val_loader, split_name="val")
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()
    def compute_and_print_confusion_matrix(self, loader, split_name="val"):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for img_a, img_b, labels in loader:
                batch = [img_a.to(self.device), img_b.to(self.device)]
                labels = labels.to(self.device)

                logits = self.model(batch)
                preds = logits.argmax(dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        print(f"\nConfusion Matrix ({split_name}):")
        print(cm)


    def print_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )
    def save_model(self, path="best_models/siamese_CNN_model.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model parameters saved to: {path}")


    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for img_a, img_b, labels in self.val_loader:
                batch = [img_a, img_b]
                batch = [b.to(self.device) for b in batch]
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"validation": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"validation": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")

    def test(self):
        """
        Evaluates the model on the final, unseen test dataset and prints the results.
        """
        results = {"preds": [], "labels": []}
        total_loss = 0
        # Set the model to evaluation mode
        self.model.eval()

        # Disable gradient calculations
        with torch.no_grad():
            # Use the dedicated test_loader for final evaluation
            for img_a, img_b, labels in self.test_loader:
                batch = [img_a, img_b]
                batch = [b.to(self.device) for b in batch]
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        # Calculate final metrics
        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        if accuracy >= 0.45:
            self.save_model(path=f"best_models/{accuracy}.pth")

        # Confusion Matrix
        cm = confusion_matrix(np.array(results["labels"]), np.array(results["preds"]))
        print("\n" + "---" * 20)
        print(f"FINAL TEST RESULTS")
        print(f"Test Accuracy: {accuracy * 100:2.2f}%")
        print(f"Confusion Matrix on Test Set:")
        print(cm)
        print("---" * 20 + "\n")

def compute_accuracy(
    labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    if args.save_dir_name != "":
        tb_log_dir_prefix = args.save_dir_name
    else:
        # tb_log_dir_prefix =f'CNN_bn_bs={args.batch_size}_lr={args.learning_rate}_wd={args.weight_decay}_ep{args.epochs}_run_'
        tb_log_dir_prefix =f'CNN_bs={args.batch_size}_lr={args.learning_rate}_wd={args.weight_decay}_ep{args.epochs}_run_'
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
        default=50,
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
        "--weight_decay",
        type=float,
        default=0,
        help="Weight decay (L2 penalty) for optimizer.",
    )

    parser.add_argument(
        "--save_dir_name",
        type=str,
        default="",
        help="Name of directory to save best model weights.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint to resume training from.",
    )
    args = parser.parse_args()

    main(args)