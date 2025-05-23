import torch.nn as nn
import torchvision.models as models


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


class AdriaanGRU(nn.Module):
    def __init__(self, config):
        image_size = config.get("image_size", 64)
        super().__init__()
        self.config = config
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.gru = nn.GRU(
            input_size=16 * (image_size // 2) * (image_size // 2),
            hidden_size=config["hidden_size"],
            batch_first=True,
        )
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])

    def forward(self, x):
        batch_size = x.size(0)
        x = self.feature_extractor(x)
        x = x.view(batch_size, 1, -1)  # [batch, seq_len=1, features]
        _, h_n = self.gru(x)
        out = self.classifier(h_n.squeeze(0))
        return out


class AdriaanTransfer(nn.Module):
    """
    Transfer learning model gebaseerd op een voorgetraind netwerk (bijv. ResNet18).
    Alleen de laatste lagen worden aangepast voor de specifieke classificatietaak.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Laad een voorgetraind ResNet18-model
        self.base_model = models.resnet18(pretrained=True)

        # Vries alle lagen behalve de laatste
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Pas de laatste volledig verbonden laag aan
        num_features = self.base_model.fc.in_features
        layers = [
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.5)),
        ]

        # Voeg extra lagen toe op basis van transfer_custom_layers
        for _ in range(config.get("transfer_custom_layers", 1) - 1):
            layers.extend(
                [nn.Linear(128, 128), nn.ReLU(), nn.Dropout(config.get("dropout", 0.5))]
            )

        layers.append(nn.Linear(128, config["num_classes"]))
        self.base_model.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.base_model(x)
