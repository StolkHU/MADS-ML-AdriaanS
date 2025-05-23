from ray import tune
import toml


class ModelSearchConfig:
    def __init__(self, config_file):
        self.config = toml.load(config_file)["model"]

    def get_search_space(self):
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
