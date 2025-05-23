import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import ray
from ray import train, tune


class AdriaanNet(nn.Module):
    """
    A simple configurable neural network for images.
    Easy to understand and expand with more layers.
    """

    def __init__(self, config):
        super().__init__()

        # Store config for easy access
        self.config = config

        # First layer: always from image to hidden size
        layers = [
            nn.Conv2d(3, config["hidden_size"], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]

        # Add more conv layers based on config
        current_size = config["hidden_size"]
        for i in range(config["num_layers"] - 1):
            layers.extend(
                [
                    nn.Conv2d(current_size, current_size * 2, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            current_size = current_size * 2

        self.features = nn.Sequential(*layers)

        # Calculate size after convolutions
        # Use the actual image size from config
        image_size = config.get("image_size", 64)
        final_size = image_size // (2 ** config["num_layers"])
        flat_features = current_size * final_size * final_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 128),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(128, config["num_classes"]),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def load_flowers_data(batch_size=32, image_size=224):
    """
    Load Flowers dataset - 102 flower categories
    Downloads automatically if not present
    """
    # Data transforms
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # Resize to fixed size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet standards
        ]
    )

    # Download and load dataset
    # Flowers102 has train, val, and test splits
    train_data = datasets.Flowers102(
        root="./data", split="train", download=True, transform=transform
    )

    val_data = datasets.Flowers102(
        root="./data", split="val", download=True, transform=transform
    )

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(config):
    """
    Train function that ray tune will call with different configs
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with config
    model = AdriaanNet(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Load data with smaller images to save memory
    image_size = config.get("image_size", 64)  # Default 64x64 for speed
    train_loader, val_loader = load_flowers_data(config["batch_size"], image_size)

    # Training loop
    for epoch in range(5):  # Less epochs for flowers (more complex)
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        # Calculate metrics
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Report to ray tune
        train.report(
            {"loss": avg_val_loss, "accuracy": val_acc, "train_accuracy": train_acc}
        )

        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")


def main():
    """
    Main function to run hyperparameter tuning
    """
    # Initialize ray
    ray.init()

    # Fase 1: Brede verkenning
    search_space = {
        "hidden_size": tune.choice([16, 32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3]),
        "dropout": tune.uniform(0.0, 0.5),
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "num_classes": 102,
        "image_size": 64,
    }

    analysis = tune.run(
        train_model,
        config=search_space,
        num_samples=20,
        metric="accuracy",
        mode="max",
        verbose=1,
        resources_per_trial={"cpu": 1, "gpu": 0.5 if torch.cuda.is_available() else 0},
    )

    # Beste configuratie ophalen
    best_config = analysis.best_config

    # Fase 2: Verfijning
    refined_space = {
        "hidden_size": tune.choice(
            [best_config["hidden_size"], best_config["hidden_size"] * 2]
        ),
        "num_layers": tune.choice(
            [best_config["num_layers"], best_config["num_layers"] + 1]
        ),
        "dropout": tune.uniform(
            max(0.0, best_config["dropout"] - 0.1),
            min(0.5, best_config["dropout"] + 0.1),
        ),
        "lr": tune.loguniform(best_config["lr"] / 2, best_config["lr"] * 2),
        "batch_size": tune.choice([best_config["batch_size"]]),
        "num_classes": 102,
        "image_size": 64,
    }

    analysis = tune.run(
        train_model,
        config=refined_space,
        num_samples=10,
        metric="accuracy",
        mode="max",
        verbose=1,
        resources_per_trial={"cpu": 1, "gpu": 0.5 if torch.cuda.is_available() else 0},
    )

    # Print best config
    best_config = analysis.best_config
    print("\nBest config found:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")

    print(f"\nBest validation accuracy: {analysis.best_result['accuracy']:.2f}%")

    ray.shutdown()


if __name__ == "__main__":
    main()
