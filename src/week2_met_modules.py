from datetime import datetime
from pathlib import Path
from typing import Iterator, Dict, Any, List, Optional

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import Trials, fmin, tpe, hp
from hyperopt.pyll.base import scope
from loguru import logger
from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor


def get_fashion_streamers(batchsize: int) -> tuple[Iterator, Iterator]:
    """Fashion MNIST datastreamer ophalen."""
    fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = fashionfactory.create_datastreamer(
        batchsize=batchsize, preprocessor=preprocessor
    )
    return streamers["train"].stream(), streamers["valid"].stream()


def get_device() -> str:
    """Beste beschikbare rekendevice bepalen."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
        logger.info("Using MPS")
    elif torch.cuda.is_available():
        device = "cuda:0"
        logger.info("Using cuda")
    else:
        device = "cpu"
        logger.info("Using cpu")
    return device


def setup_mlflow(experiment_path: str) -> None:
    """MLflow configureren."""
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_path)


class FlexibleCNN(nn.Module):
    """Flexibel CNN model met vaste blokken maar flexibel aantal lagen."""

    def __init__(
        self,
        num_conv_blocks: int = 3,
        num_dense_blocks: int = 2,
        filters: List[int] = None,
        units: List[int] = None,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.25,
        input_channels: int = 1,
        input_size: tuple = (32, 1, 28, 28),
        num_classes: int = 10,
    ):
        """
        Initializeer FlexibleCNN met een configureerbaar aantal blokken.

        Args:
            num_conv_blocks: Aantal convolutionele blokken
            num_dense_blocks: Aantal dense blokken
            filters: Lijst met aantal filters voor elke conv laag
            units: Lijst met aantal units voor elke dense laag
            use_batch_norm: Of batch normalization gebruikt moet worden
            dropout_rate: Dropout rate (0 voor geen dropout)
            input_channels: Aantal invoerkanalen (1 voor grijsschaal, 3 voor RGB)
            input_size: Grootte van de invoer (batch, channels, height, width)
            num_classes: Aantal uitvoerklassen
        """
        super().__init__()

        # Basisparameters
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Default waarden voor filters en units
        if filters is None:
            filters = [32 * (2**i) for i in range(num_conv_blocks)]  # 32, 64, 128, ...
        if units is None:
            units = [
                128 // (2**i) if i > 0 else 128 for i in range(num_dense_blocks)
            ]  # 128, 64, 32, ...

        # Zorg ervoor dat de lijsten lang genoeg zijn
        filters = filters[:num_conv_blocks] + [filters[-1]] * (
            num_conv_blocks - len(filters)
        )
        units = units[:num_dense_blocks] + [units[-1]] * (num_dense_blocks - len(units))

        # Opbouwen van convolutionele blokken
        self.conv_blocks = nn.ModuleList()
        in_channels = self.input_channels

        for i in range(num_conv_blocks):
            block = self._create_conv_block(in_channels, filters[i])
            self.conv_blocks.append(block)
            in_channels = filters[i]

        # Bereken feature size en bouw dense blokken
        self.feature_size = self._compute_feature_size()
        logger.info(f"Feature size after convolutions: {self.feature_size}")

        # Opbouwen van dense blokken
        self.dense_blocks = nn.ModuleList()
        in_features = self.feature_size

        for i in range(num_dense_blocks):
            block = self._create_dense_block(in_features, units[i])
            self.dense_blocks.append(block)
            in_features = units[i]

        # Output laag
        self.output_layer = nn.Linear(in_features, self.num_classes)

    def _create_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Standaard convolutioneel blok met batch norm en dropout."""
        layers = []

        # Convolutielaag met padding=1 om feature map grootte te behouden
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        # Batch Normalization (optioneel)
        if self.use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        # Activatiefunctie
        layers.append(nn.ReLU())

        # Pooling laag
        layers.append(nn.MaxPool2d(kernel_size=2))

        # Dropout (optioneel)
        if self.dropout_rate > 0:
            layers.append(nn.Dropout2d(self.dropout_rate))

        return nn.Sequential(*layers)

    def _create_dense_block(self, in_features: int, out_features: int) -> nn.Sequential:
        """Standaard dense blok met batch norm en dropout."""
        layers = []

        # Lineaire laag
        layers.append(nn.Linear(in_features, out_features))

        # Batch Normalization (optioneel)
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(out_features))

        # Activatiefunctie
        layers.append(nn.ReLU())

        # Dropout (optioneel)
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))

        return nn.Sequential(*layers)

    def _compute_feature_size(self) -> int:
        """Bereken de grootte van de output feature map voor flattening."""
        x = torch.ones(self.input_size)

        for i, block in enumerate(self.conv_blocks):
            try:
                x = block(x)
                logger.info(f"Feature map na conv blok {i}: {x.shape}")
            except RuntimeError as e:
                logger.error(f"Error in conv blok {i}: {e}")
                # Pas het model dynamisch aan bij fouten
                self.conv_blocks = nn.ModuleList(list(self.conv_blocks)[:i])
                x = torch.ones(self.input_size)
                for j in range(i):
                    x = self.conv_blocks[j](x)
                logger.info(
                    f"Model aangepast: {i} convolutionele blokken, feature map: {x.shape}"
                )
                break

        # Bereken het aantal features na flattening
        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass voor het model."""
        # Convolutionele blokken
        for block in self.conv_blocks:
            x = block(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense blokken
        for block in self.dense_blocks:
            x = block(x)

        # Output laag
        return self.output_layer(x)


def objective(params: Dict[str, Any]) -> float:
    """
    Objective functie voor hyperparameter optimalisatie.

    Args:
        params: Dictionary met hyperparameters

    Returns:
        Negatieve validatie nauwkeurigheid (voor minimalisatie)
    """
    # Model directory aanmaken indien nodig
    modeldir = Path("models").resolve()
    if not modeldir.exists():
        modeldir.mkdir(parents=True)
        logger.info(f"Created {modeldir}")

    # Datastreamer ophalen
    batchsize = params.get("batchsize", 64)
    trainstreamer, validstreamer = get_fashion_streamers(batchsize)

    # Trainer settings
    accuracy = metrics.Accuracy()
    settings = TrainerSettings(
        epochs=params.get("epochs", 20),
        metrics=[accuracy],
        logdir=Path("modellog"),
        train_steps=937,
        valid_steps=937,
        reporttypes=[ReportTypes.MLFLOW],
    )

    # Device ophalen
    device = get_device()

    # Training met MLflow tracking
    with mlflow.start_run():
        # MLflow metadata
        mlflow.set_tag("model", "flexible_convnet")
        mlflow.set_tag("dev", "Adriaan")
        mlflow.log_params(params)
        mlflow.log_param("batchsize", f"{batchsize}")

        # Model parameters
        num_conv_blocks = params.get("num_conv_blocks", 3)
        num_dense_blocks = params.get("num_dense_blocks", 2)

        # Filter en unit configuraties
        filters = [
            params.get(f"filters{i + 1}", 32 * (2**i)) for i in range(num_conv_blocks)
        ]
        units = [
            params.get(f"units{i + 1}", 128 // (2**i if i > 0 else 1))
            for i in range(num_dense_blocks)
        ]

        # Model, optimizer en loss
        model = FlexibleCNN(
            num_conv_blocks=num_conv_blocks,
            num_dense_blocks=num_dense_blocks,
            filters=filters,
            units=units,
            use_batch_norm=params.get("use_batch_norm", True),
            dropout_rate=params.get("dropout_rate", 0.25),
        )
        model.to(device)

        optimizer_class = getattr(optim, params.get("optimizer", "Adam"))
        loss_fn = torch.nn.CrossEntropyLoss()

        # Training
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer_class,  # type: ignore
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
        trainer.loop()

        # Model opslaan
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = modeldir / (tag + "model.pt")
        logger.info(f"Saving model to {modelpath}")
        torch.save(model, modelpath)

        # Beste validatie nauwkeurigheid teruggeven (negatief voor minimalisatie)
        best_val_accuracy = max(trainer.val_metrics.get("accuracy", [0.0]))
        return -best_val_accuracy


def main():
    """Hoofdfunctie voor training en hyperparameter optimalisatie."""
    setup_mlflow("flexible_cnn_experiment")

    # Hyperparameter zoekruimte
    search_space = {
        # Structuur hyperparameters
        "num_conv_blocks": scope.int(hp.quniform("num_conv_blocks", 2, 4, 1)),
        "num_dense_blocks": scope.int(hp.quniform("num_dense_blocks", 1, 3, 1)),
        # Convolutionele lagen hyperparameters
        "filters1": scope.int(hp.quniform("filters1", 16, 64, 8)),
        "filters2": scope.int(hp.quniform("filters2", 32, 128, 8)),
        "filters3": scope.int(hp.quniform("filters3", 64, 256, 8)),
        # Dense lagen hyperparameters
        "units1": scope.int(hp.quniform("units1", 64, 256, 8)),
        "units2": scope.int(hp.quniform("units2", 32, 128, 8)),
        # Regularisatie hyperparameters
        "use_batch_norm": hp.choice("use_batch_norm", [True, False]),
        "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.5),
        # Training hyperparameters
        "batchsize": scope.int(hp.quniform("batchsize", 32, 128, 32)),
        "epochs": scope.int(hp.quniform("epochs", 10, 30, 5)),
        "optimizer": hp.choice("optimizer", ["Adam", "SGD", "RMSprop"]),
    }

    # Hyperparameter optimalisatie
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=10,
        trials=Trials(),
    )
    logger.info(f"Beste hyperparameters: {best_result}")

    # Alternatief: Enkele run met vaste parameters
    # fixed_params = {
    #     'num_conv_blocks': 3,
    #     'num_dense_blocks': 2,
    #     'filters1': 32,
    #     'filters2': 64,
    #     'filters3': 128,
    #     'units1': 128,
    #     'units2': 64,
    #     'use_batch_norm': True,
    #     'dropout_rate': 0.25,
    #     'batchsize': 64,
    #     'epochs': 20,
    #     'optimizer': 'Adam',
    # }
    # objective(fixed_params)


if __name__ == "__main__":
    main()

# MLflow server starten met:
# mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
