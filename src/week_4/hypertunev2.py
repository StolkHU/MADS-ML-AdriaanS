import torch
import torch.nn as nn
import ray
from ray import train, tune
import toml
from my_own_models import AdriaanNet, AdriaanGRU, AdriaanTransfer
from dataloader import FlowerDataLoader


class ModelSearchConfig:
    """
    Class to handle the configuration of the model search space.
    """

    def __init__(self, config_file):
        self.config = toml.load(config_file)["model"]

    def get_search_space(self):
        """
        Define the search space for the model.
        """
        search_space = {
            "model_type": tune.choice(self.config["model_type"]),
            "hidden_size": tune.choice(self.config["hidden_size"]),
            "num_layers": tune.choice(self.config["num_layers"]),
            "dropout": tune.uniform(
                self.config["dropout"][0], self.config["dropout"][1]
            ),
            "lr": tune.loguniform(self.config["lr"][0], self.config["lr"][1]),
            "batch_size": tune.choice(self.config["batch_size"]),
            "num_classes": self.config["num_classes"],
            "image_size": self.config["image_size"],
            "transfer_custom_layers": tune.choice(
                self.config["transfer_custom_layers"]
            ),
        }
        return search_space


def train_model(config):
    """
    Train function that ray tune will call with different configs.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with config
    if config["model_type"] == "gru":
        model = AdriaanGRU(config).to(device)
    elif config["model_type"] == "transfer":
        model = AdriaanTransfer(config).to(device)
    else:
        model = AdriaanNet(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Load data with smaller images to save memory
    data_loader = FlowerDataLoader(
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        augment=config.get("augment", True),
        resize_size=config.get("resize_size", 256),
        crop_size=config.get("crop_size", 224),
    )

    train_loader, val_loader = data_loader.load_data()

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
    ray.init()

    config_file = "config.toml"
    full_config = toml.load(config_file)
    experiment_name = full_config.get("experiment", {}).get(
        "name", "default_experiment"
    )

    search_config = ModelSearchConfig(config_file)
    search_space = search_config.get_search_space()

    analysis = tune.run(
        train_model,
        config=search_space,
        num_samples=20,
        metric="accuracy",
        mode="max",
        verbose=1,
        name=experiment_name,
        resources_per_trial={"cpu": 1, "gpu": 0.5 if torch.cuda.is_available() else 0},
    )

    best_config = analysis.best_config
    print("\nBest config found:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")

    print(f"\nBest validation accuracy: {analysis.best_result['accuracy']:.2f}%")
    ray.shutdown()


if __name__ == "__main__":
    main()
