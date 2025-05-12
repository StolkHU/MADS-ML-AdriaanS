from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer.preprocessors import BasePreprocessor
from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics

import torch
import torch.optim as optim
from torch import nn
from loguru import logger

# Dataset inladen - Fashion MNIST heeft afbeeldingen van 28x28 met 1 kanaal
fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
preprocessor = BasePreprocessor()
streamers = fashionfactory.create_datastreamer(batchsize=64, preprocessor=preprocessor)
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()

# Metrics en loss functie definiëren
accuracy = metrics.Accuracy()
loss_fn = torch.nn.CrossEntropyLoss()

# Hyperparameters
units = [256, 128, 64]
batchsize = 64


# Define CNN model
class CNN(nn.Module):
    def __init__(
        self, filters: int, units1: int, units2: int, units3: int, input_size: tuple
    ):
        super().__init__()
        self.in_channels = input_size[1]  # Aantal kanalen (1 voor Fashion MNIST)
        self.input_size = input_size
        self.filters = filters
        self.units1 = units1
        self.units2 = units2
        self.units3 = units3

        self.convolutions = nn.Sequential(
            nn.Conv2d(self.in_channels, filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Test om de uitvoergrootte van de convoluties te bepalen
        activation_map_size = self._conv_test(self.input_size)
        logger.info(f"Aggregating activationmap with size {activation_map_size}")
        self.agg = nn.AvgPool2d(activation_map_size)

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters, units1),
            nn.ReLU(),
            nn.Linear(units1, units2),
            nn.ReLU(),
            nn.Linear(units2, units3),
            nn.ReLU(),
            nn.Linear(units3, 10),  # 10 klassen voor Fashion MNIST
        )

    def _conv_test(self, input_size):
        x = torch.ones(input_size, dtype=torch.float32)
        x = self.convolutions(x)
        return x.shape[-2:]

    def forward(self, x):
        x = self.convolutions(x)
        x = self.agg(x)
        logits = self.dense(x)
        return logits


# Trainer instellingen
settings = TrainerSettings(
    epochs=20,
    metrics=[accuracy],
    logdir="modellogs",
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
)

# Fashion MNIST heeft 28x28 afbeeldingen met 1 kanaal (grijswaarden)
# Batchgrootte is 64, dus input_size = (64, 1, 28, 28)
input_size = (batchsize, 1, 28, 28)

# Grid search over verschillende unit groottes
for unit1 in units:
    for unit2 in units:
        if unit2 <= unit1:
            for unit3 in units:
                if unit3 <= unit2:
                    model = CNN(
                        filters=128,
                        units1=unit1,
                        units2=unit2,
                        units3=unit3,
                        input_size=input_size,
                    )

                    trainer = Trainer(
                        model=model,
                        settings=settings,
                        loss_fn=loss_fn,
                        optimizer=optim.Adam,
                        traindataloader=trainstreamer,
                        validdataloader=validstreamer,
                        scheduler=optim.lr_scheduler.ReduceLROnPlateau,
                    )
                    trainer.loop()
