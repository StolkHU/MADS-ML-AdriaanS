# from mads_datasets import DatasetFactoryProvider, DatasetType
# from mltrainer.preprocessors import BasePreprocessor
# from mltrainer import Trainer, TrainerSettings, ReportTypes, metrics
# from mltrainer.imagemodels import CNNConfig, CNNblocks
# from torchinfo import summary
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# from hyperopt.pyll import scope

# from pathlib import Path
# import torch
# import torch.optim as optim
# from torch import nn
# import mlflow

# fashionfactory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
# preprocessor = BasePreprocessor()
# streamers = fashionfactory.create_datastreamer(batchsize=64, preprocessor=preprocessor)
# train = streamers["train"]
# valid = streamers["valid"]
# trainstreamer = train.stream()
# validstreamer = valid.stream()

# accuracy = metrics.Accuracy()
# loss_fn = torch.nn.CrossEntropyLoss()
# units = [256, 128, 64]
# batchsize = 64
# optimizer = optim.Adam

# if torch.backends.mps.is_available() and torch.backends.mps.is_built():
#     device = torch.device("mps")
#     print("Using MPS")
# elif torch.cuda.is_available():
#     device = "cuda:0"
#     print("using cuda")
# else:
#     device = "cpu"
#     print("using cpu")

# modeldir = Path("models").resolve()
# if not modeldir.exists():
#     modeldir.mkdir()
#     print(f"Created {modeldir}")


# class NeuralNetwork(nn.Module):
#     def __init__(
#         self, num_classes: int, units1: int, units2: int, units3: int = None
#     ) -> None:
#         super().__init__()
#         self.num_classes = num_classes
#         self.units1 = units1
#         self.units2 = units2
#         self.units3 = units3
#         self.flatten = nn.Flatten()

#         # Als units3 None is, maak netwerk met 2 hidden layers
#         if units3 is None:
#             self.linear_relu_stack = nn.Sequential(
#                 nn.Linear(28 * 28, units1),
#                 nn.BatchNorm1d(units1),
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(units1, units2),
#                 nn.BatchNorm1d(units2),
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(units2, num_classes),
#             )
#         # Anders maak netwerk met 3 hidden layers
#         else:
#             self.linear_relu_stack = nn.Sequential(
#                 nn.Linear(28 * 28, units1),
#                 nn.BatchNorm1d(units1),
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(units1, units2),
#                 nn.BatchNorm1d(units2),
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(units2, units3),
#                 nn.BatchNorm1d(units3),
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(units3, num_classes),
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


# settings = TrainerSettings(
#     epochs=20,
#     metrics=[accuracy],
#     logdir="modellogs",
#     train_steps=len(train),
#     valid_steps=len(valid),
#     reporttypes=[ReportTypes.TENSORBOARD, ReportTypes.TOML],
# )

# # for unit1 in units:
# #     for unit2 in units:
# #         if unit2 <= unit1:
# #             for unit3 in units:
# #                 if unit3 <= unit2:

# #                     model = NeuralNetwork(num_classes=10, units1=unit1, units2=unit2, units3=unit3)

# #                     trainer = Trainer(
# #                                     model=model,
# #                                     settings=settings,
# #                                     loss_fn=loss_fn,
# #                                     optimizer=optim.Adam,
# #                                     traindataloader=trainstreamer,
# #                                     validdataloader=validstreamer,
# #                                     scheduler=optim.lr_scheduler.ReduceLROnPlateau
# #                                 )
# #                     trainer.loop()


# config = CNNConfig(
#     matrixshape=(28, 28),  # every image is 28x28
#     batchsize=batchsize,
#     input_channels=1,  # we have black and white images, so only one channel
#     hidden=32,  # number of filters
#     kernel_size=3,  # kernel size of the convolution
#     maxpool=3,  # kernel size of the maxpool
#     num_layers=4,  # we will stack 4 Convolutional blocks, each with two Conv2d layers
#     num_classes=10,
# )

# model = CNNblocks(config)
# model.config
# summary(model, input_size=(32, 1, 28, 28))

# experiment_path = "mlflow_test"


# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# mlflow.set_experiment(experiment_path)


# # Define the hyperparameter search space
# settings = TrainerSettings(
#     epochs=3,
#     metrics=[accuracy],
#     logdir=modeldir,
#     train_steps=100,
#     valid_steps=100,
#     reporttypes=[ReportTypes.MLFLOW, ReportTypes.TOML],
# )


# # Define the objective function for hyperparameter optimization
# def objective(params):
#     # Start a new MLflow run for tracking the experiment
#     with mlflow.start_run():
#         # Set MLflow tags to record metadata about the model and developer
#         mlflow.set_tag("model", "convnet")
#         mlflow.set_tag("dev", "raoul")
#         # Log hyperparameters to MLflow
#         mlflow.log_params(params)
#         mlflow.log_param("batchsize", f"{batchsize}")

#         config = CNNConfig(
#             matrixshape=(28, 28),  # every image is 28x28
#             batchsize=batchsize,
#             input_channels=1,  # we have black and white images, so only one channel
#             hidden=params["filters"],  # number of filters
#             kernel_size=3,  # kernel size of the convolution
#             maxpool=3,  # kernel size of the maxpool
#             num_layers=4,  # we will stack 4 Convolutional blocks, each with two Conv2d layers
#             num_classes=10,
#         )

#         # Instantiate the CNN model with the given hyperparameters
#         model = CNNblocks(config)
#         # Train the model using a custom train loop
#         trainer = Trainer(
#             model=model,
#             settings=settings,
#             loss_fn=loss_fn,
#             optimizer=optimizer,
#             traindataloader=trainstreamer,
#             validdataloader=validstreamer,
#             scheduler=optim.lr_scheduler.ReduceLROnPlateau,
#             device=device,
#         )
#         trainer.loop()

#         # Save the trained model with a timestamp
#         tag = datetime.now().strftime("%Y%m%d-%H%M")
#         modelpath = modeldir / (tag + "model.pt")
#         torch.save(model, modelpath)

#         # Log the saved model as an artifact in MLflow
#         mlflow.log_artifact(local_path=modelpath, artifact_path="pytorch_models")
#         return {"loss": trainer.test_loss, "status": STATUS_OK}


# search_space = {
#     "filters": scope.int(hp.quniform("filters", 16, 128, 8)),
#     "kernel_size": scope.int(hp.quniform("kernel_size", 2, 5, 1)),
#     "num_layers": scope.int(hp.quniform("num_layers", 1, 10, 1)),
# }

# best_result = fmin(
#     fn=objective, space=search_space, algo=tpe.suggest, max_evals=3, trials=Trials()
# )

# best_result
