import torch
import torch.nn as nn
import ray
from ray import train, tune
import toml
from models import AdriaanNet, AdriaanGRU, AdriaanTransfer
from dataloader import FlowerDataLoader


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
    """Train model met MADS datasets"""
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

        # Load data
        data_loader = FlowerDataLoader(config)
        train_streamer, val_streamer = data_loader.load_data()

        # Training
        num_epochs = config.get("epochs", 3)
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            # MADS datasets: gebruik .stream() en next()
            train_stream = train_streamer.stream()
            batch_idx = 0

            try:
                while True:
                    inputs, targets = next(train_stream)
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
                    batch_idx += 1

            except StopIteration:
                pass

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                val_stream = val_streamer.stream()
                try:
                    while True:
                        inputs, targets = next(val_stream)
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

                except StopIteration:
                    pass

            # Report results
            train_acc = 100.0 * correct / total if total > 0 else 0
            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
            avg_val_loss = (
                val_loss / len(val_streamer) if len(val_streamer) > 0 else 999.0
            )

            print(f"Epoch {epoch + 1}: Train {train_acc:.1f}%, Val {val_acc:.1f}%")

            train.report(
                {"loss": avg_val_loss, "accuracy": val_acc, "train_accuracy": train_acc}
            )

    except Exception as e:
        print(f"Training failed: {e}")


def main():
    """Main functie met ModelSearchConfig"""
    # Ray setup
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)

    try:
        # Load config
        config_file = "config.toml"
        with open(config_file, "r") as f:
            full_config = toml.load(f)

        # Gebruik ModelSearchConfig
        search_config = ModelSearchConfig(config_file)
        search_space = search_config.get_search_space()

        # Voeg andere configs toe
        search_space.update(full_config.get("data", {}))
        search_space.update(full_config.get("training", {}))

        # Run search
        num_trials = full_config.get("tune", {}).get("num_samples", 3)
        experiment_name = full_config.get("experiment", {}).get("name", "flower_search")

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
            print(f"\n🎉 Best accuracy: {analysis.best_result['accuracy']:.2f}%")
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
