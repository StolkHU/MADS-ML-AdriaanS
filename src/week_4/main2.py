import torch
import torch.nn as nn
import ray
from ray import train, tune
import toml
from models import AdriaanNet, AdriaanGRU, AdriaanTransfer
from dataloader2 import HymenopteraDataLoader


class ModelSearchConfig:
    """Config loader voor hyperparameter search"""

    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.config = toml.load(f)["model"]

    def get_search_space(self):
        """Maak search space voor Ray Tune"""
        return {
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


def train_model(config):
    """Train model op Hymenoptera dataset"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        if config["model_type"] == "gru":
            model = AdriaanGRU(config).to(device)
        elif config["model_type"] == "transfer":
            model = AdriaanTransfer(config).to(device)
        else:
            model = AdriaanNet(config).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # Load Hymenoptera data
        data_loader = HymenopteraDataLoader(config)
        train_loader, val_loader = data_loader.load_data()

        # Training
        num_epochs = config.get("epochs", 3)
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            # Normale PyTorch DataLoader iteratie
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

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

            # Calculate results
            train_acc = 100.0 * correct / total if total > 0 else 0
            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 999.0

            print(f"Epoch {epoch + 1}: Train {train_acc:.1f}%, Val {val_acc:.1f}% 🐝🐜")

            train.report(
                {"loss": avg_val_loss, "accuracy": val_acc, "train_accuracy": train_acc}
            )

    except Exception as e:
        print(f"Training failed: {e}")
        train.report({"loss": 999.0, "accuracy": 0.0, "train_accuracy": 0.0})


def main():
    """Main functie voor Hymenoptera classificatie"""
    print("🐝🐜 Starting Hymenoptera (Bees vs Ants) Classification!")

    # Ray setup
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)

    try:
        # Load config
        config_file = "config_hymenoptera.toml"
        with open(config_file, "r") as f:
            full_config = toml.load(f)

        # Setup search space
        search_config = ModelSearchConfig(config_file)
        search_space = search_config.get_search_space()

        # Add other configs
        search_space.update(full_config.get("data", {}))
        search_space.update(full_config.get("training", {}))

        # Run search
        num_trials = full_config.get("tune", {}).get("num_samples", 5)
        experiment_name = full_config.get("experiment", {}).get("name", "hymenoptera")

        print(f"Starting {num_trials} trials for {experiment_name}...")

        analysis = tune.run(
            train_model,
            config=search_space,
            num_samples=num_trials,
            metric="accuracy",
            mode="max",
            name=experiment_name,
            max_concurrent_trials=1,
            resources_per_trial={"cpu": 1, "gpu": 0},
        )

        # Results
        if analysis.best_config:
            print(
                f"\n🎉 Best Bees vs Ants accuracy: {analysis.best_result['accuracy']:.2f}%"
            )
            print("Best config:")
            for key, value in analysis.best_config.items():
                print(f"  {key}: {value}")
        else:
            print("❌ No successful trials")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
