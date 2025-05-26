import torch
import torch.nn as nn
import ray
from ray import train, tune
import toml
import pandas as pd
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
    """Train model met learning rate scheduler en early stopping"""
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

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("scheduler_step", 3),  # Elke 3 epochs
            gamma=config.get("scheduler_gamma", 0.5),  # Halveer learning rate
        )

        best_val_acc = 0.0
        patience = config.get("patience", 5)  # Stop na 5 epochs zonder verbetering
        patience_counter = 0

        # Load data
        data_loader = HymenopteraDataLoader(config)
        train_loader, val_loader = data_loader.load_data()

        # Training loop
        num_epochs = config.get("epochs", 10)
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0

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

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0  # Reset counter als er verbetering is
                print(f"Epoch {epoch + 1}: Nieuwe beste accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                print(
                    f"Epoch {epoch + 1}: Geen verbetering ({patience_counter}/{patience})"
                )

            # Report to Ray Tune
            train.report(
                {
                    "loss": avg_val_loss,
                    "accuracy": val_acc,
                    "train_accuracy": train_acc,
                    "learning_rate": current_lr,
                    "epoch": epoch + 1,
                }
            )

            if patience_counter >= patience:
                print(f"Early stopping! Geen verbetering na {patience} epochs.")
                print(f"Beste accuracy was: {best_val_acc:.2f}%")
                break

    except Exception as e:
        print(f"Error tijdens training: {e}")
        train.report({"loss": 999.0, "accuracy": 0.0, "train_accuracy": 0.0})


def main():
    if ray.is_initialized():
        ray.shutdown()

    ray.init(ignore_reinit_error=True, dashboard_host="0.0.0.0", dashboard_port=8265)

    try:
        config_file = "config_hymenoptera.toml"
        with open(config_file, "r") as f:
            full_config = toml.load(f)

        search_config = ModelSearchConfig(config_file)
        search_space = search_config.get_search_space()

        search_space.update(full_config.get("data", {}))
        search_space.update(full_config.get("training", {}))

        num_trials = full_config.get("tune", {}).get("num_samples", 5)
        experiment_name = full_config.get("experiment", {}).get("name", "hymenoptera")

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

        if analysis.trials:
            results_data = []

            for trial in analysis.trials:
                if trial.last_result:
                    row = {
                        "trial_id": trial.trial_id,
                        "accuracy": trial.last_result.get("accuracy", 0),
                        "loss": trial.last_result.get("loss", 999),
                        "train_accuracy": trial.last_result.get("train_accuracy", 0),
                    }
                    row.update(trial.config)
                    results_data.append(row)

            df = pd.DataFrame(results_data)
            df = df.sort_values("accuracy", ascending=False)

            csv_filename = f"{experiment_name}_results.csv"
            df.to_csv(csv_filename, index=False)

    except Exception:
        import traceback

        traceback.print_exc()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
