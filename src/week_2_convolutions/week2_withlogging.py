import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from mltrainer import metrics, Trainer, TrainerSettings, ReportTypes
from mltrainer.imagemodels import CNNConfig, CNNblocks
from mltrainer.preprocessors import BasePreprocessor
from mads_datasets import DatasetFactoryProvider, DatasetType

warnings.simplefilter("ignore", UserWarning)

experiment_path = "mlflow_test"

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_path)


####### INPUT #######
optimizer = optim.Adam
loss_fn = torch.nn.CrossEntropyLoss()
accuracy = metrics.Accuracy()
batchsize = 64
fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
preprocessor = BasePreprocessor()
streamers = fashionfactory.create_datastreamer(
    batchsize=batchsize, preprocessor=preprocessor
)
train = streamers["train"]
valid = streamers["valid"]
trainstreamer = train.stream()
validstreamer = valid.stream()

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using MPS")
elif torch.cuda.is_available():
    device = "cuda:0"
    print("using cuda")
else:
    device = "cpu"
    print("using cpu")

modeldir = Path("models").resolve()
if not modeldir.exists():
    modeldir.mkdir()
    print(f"Created {modeldir}")

# Define the hyperparameter search space
settings = TrainerSettings(
    epochs=10,
    metrics=[accuracy],
    logdir=modeldir,
    train_steps=len(train),
    valid_steps=len(valid),
    reporttypes=[ReportTypes.MLFLOW, ReportTypes.TOML],
)


# Define the objective function for hyperparameter optimization
def objective(params):
    # Eenvoudige check voor kernel_size en num_layers combinatie
    # Grote kernel sizes met veel lagen kunnen problemen veroorzaken
    if params["kernel_size"] >= 5 and params["num_layers"] > 3:
        print(
            f"Ongeldige combinatie: kernel_size={params['kernel_size']} en num_layers={params['num_layers']} is mogelijk te groot voor 28x28 beelden"
        )
        # Return een hoge verlieswaarde om deze combinatie te ontmoedigen
        return {"loss": 9999.0, "status": STATUS_OK}

    # Start a new MLflow run for tracking the experiment
    with mlflow.start_run():
        # Set MLflow tags to record metadata about the model and developer
        mlflow.set_tag("model", "convnet")
        mlflow.set_tag("dev", "adriaan")
        # Log hyperparameters to MLflow
        mlflow.log_params(params)
        mlflow.log_param("batchsize", f"{batchsize}")

        # Initialize the optimizer, loss function, and accuracy metric
        optimizer = optim.Adam
        loss_fn = torch.nn.CrossEntropyLoss()

        # Gebruik de hyperparameters uit de zoekruimte
        config = CNNConfig(
            matrixshape=(28, 28),  # every image is 28x28
            batchsize=batchsize,
            input_channels=1,  # we have black and white images, so only one channel
            hidden=params["filters"],  # number of filters
            kernel_size=params["kernel_size"],  # kernel size of the convolution
            maxpool=3,  # kernel size of the maxpool
            num_layers=params["num_layers"],  # number of convolutional blocks
            num_classes=10,
        )

        # Instantiate the CNN model with the given hyperparameters
        model = CNNblocks(config)

        # Pas hier handmatig de dense_units aan als dat mogelijk is
        # Opmerking: Dit werkt alleen als het CNNblocks model een manier heeft om dense_units aan te passen
        # Als dit niet werkt, moet je een aangepaste versie van CNNblocks maken
        try:
            # Probeer om de dense layer aan te passen na initialisatie
            # (Dit is een voorbeeld; de exacte implementatie hangt af van de CNNblocks klasse)
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    print(
                        f"Aangepaste dense layer gevonden: {name}, originele grootte: {module.out_features}"
                    )
                    # Als je de grootte van de dense layer wilt aanpassen, moet je de module vervangen
                    # Dit is een voorbeeld en werkt mogelijk niet direct in jouw model
                    if "fc" in name or "classifier" in name:
                        in_features = module.in_features
                        # Vervang alleen de laatste dense layer vóór de classifier (output) layer
                        if (
                            module.out_features != 10
                        ):  # Geen aanpassing van de output layer
                            parent_module = model
                            path = name.split(".")
                            for part in path[:-1]:
                                parent_module = getattr(parent_module, part)
                            setattr(
                                parent_module,
                                path[-1],
                                nn.Linear(in_features, params["dense_units"]),
                            )
                            print(
                                f"  -> Aangepast naar grootte: {params['dense_units']}"
                            )
        except Exception as e:
            print(f"Waarschuwing: Kon de dense layer niet aanpassen: {e}")

        # Print model summary to see the architecture
        print(
            f"\nModel with filters={params['filters']}, kernel_size={params['kernel_size']}, num_layers={params['num_layers']}"
        )
        # Zet de summary in een try-except blok om fouten te vangen
        try:
            summary(model, input_size=(batchsize, 1, 28, 28), verbose=1)
        except Exception as e:
            print(f"Kon model summary niet tonen: {e}")
            print("Doorgaan met de training...")

        # Train the model using a custom train loop
        trainer = Trainer(
            model=model,
            settings=settings,
            loss_fn=loss_fn,
            optimizer=optimizer,
            traindataloader=trainstreamer,
            validdataloader=validstreamer,
            scheduler=optim.lr_scheduler.ReduceLROnPlateau,
            device=device,
        )
        trainer.loop()

        # Save the trained model with a timestamp
        tag = datetime.now().strftime("%Y%m%d-%H%M")
        modelpath = (
            modeldir
            / f"{tag}_filters{params['filters']}_kernel{params['kernel_size']}_layers{params['num_layers']}_model.pt"
        )
        torch.save(model, modelpath)

        # Log the saved model as an artifact in MLflow
        mlflow.log_artifact(local_path=str(modelpath), artifact_path="pytorch_models")

        # Log performance metrics to MLflow
        mlflow.log_metric("test_loss", trainer.test_loss)
        mlflow.log_metric("test_accuracy", trainer.test_accuracy)

        return {"loss": trainer.test_loss, "status": STATUS_OK}


# Hyperparameter zoekruimte definitie
search_space = {
    "filters": scope.int(hp.quniform("filters", 16, 128, 8)),
    "kernel_size": scope.int(
        hp.quniform("kernel_size", 2, 3, 1)
    ),  # Beperkt tot kleinere kernels (2-3)
    "num_layers": scope.int(
        hp.quniform("num_layers", 1, 5, 1)
    ),  # Beperkt tot minder lagen (1-5)
}

# Start de hyperparameter optimalisatie
if __name__ == "__main__":
    print("Starting hyperparameter optimization...")
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=30,  # Aantal experimenten dat uitgevoerd wordt
        trials=trials,
    )

    print("\n\nBeste hyperparameters gevonden:")
    print(f"Filters: {best['filters']}")
    print(f"Kernel Size: {best['kernel_size']}")
    print(f"Number of Layers: {best['num_layers']}")

    # Toon ook de beste prestaties
    best_trial_idx = trials.best_trial["tid"]
    best_loss = trials.results[best_trial_idx]["loss"]
    print(f"Best validation loss: {best_loss}")

    # Optioneel: toon alle resultaten gesorteerd op prestatie
    print("\nAlle experimenten gesorteerd op prestatie:")
    sorted_trials = sorted(trials.results, key=lambda x: x["loss"])
    for i, trial in enumerate(sorted_trials):
        print(f"Experiment {i + 1}: Loss = {trial['loss']}")
