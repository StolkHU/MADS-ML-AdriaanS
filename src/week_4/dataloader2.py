from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import requests
import zipfile
from pathlib import Path


def get_hymenoptera():
    """Download Hymenoptera dataset (bees vs ants)"""
    datadir = Path.home() / ".cache/mads_datasets/hymenoptera_data"
    url = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"

    if not datadir.exists():
        print(f"Creating directory {datadir}")
        datadir.mkdir(parents=True)

        print("Downloading Hymenoptera dataset...")
        response = requests.get(url)
        zip_file_path = datadir / "hymenoptera_data.zip"

        with open(zip_file_path, "wb") as f:
            f.write(response.content)

        print(f"Extracting {zip_file_path}")
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(datadir)
        zip_file_path.unlink()
    else:
        print("Hymenoptera data already exists")

    return datadir / "hymenoptera_data"


class HymenopteraDataLoader:
    """Eenvoudige DataLoader voor Hymenoptera dataset (bees vs ants)"""

    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.augment = config.get("augment", True)
        self.resize_size = config.get("resize_size", 256)
        self.crop_size = config.get("crop_size", 224)
        self.image_size = config.get("image_size", 224)

    def load_data(self):
        """Load Hymenoptera dataset"""
        print(f"Loading Hymenoptera dataset with batch_size={self.batch_size}")

        # Download data
        data_dir = get_hymenoptera()

        # Define transforms
        if self.augment:
            train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        val_transform = transforms.Compose(
            [
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        # Create datasets
        train_dataset = datasets.ImageFolder(
            data_dir / "train", transform=train_transform
        )

        val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transform)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        print("✅ Hymenoptera dataset loaded:")
        print(f"  - Train: {len(train_dataset)} samples")
        print(f"  - Val: {len(val_dataset)} samples")
        print(f"  - Classes: {train_dataset.classes}")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")

        return train_loader, val_loader
